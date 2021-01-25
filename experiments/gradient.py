import jax
import jax.numpy as jnp
import lab as B
import numpy as np
import wbml.out as out
from stheno.input import MultiInput
from stheno.jax import Measure, GP, EQ, Delta, cross
from varz.jax import Vars
from wbml.experiment import WorkingDirectory

from psa import psa, cos_sim

# Initialise experiment.
wd = WorkingDirectory("_experiments", "gradient")
out.report_time = True
B.epsilon = 1e-6
B.default_dtype = jnp.float32

# Settings of experiment:
x = B.linspace(0, 10, 1000)
m = 2
p = 4
h = 1.5

sims_m1 = []
sims_m3 = []
sims_uc = []


for i in range(30):
    out.kv("Repetition", i + 1)

    # Sample a true basis.
    true_basis = Vars(jnp.float32).orthogonal(shape=(p, p))

    # Build a model for the data.
    prior = Measure()
    z_model = [GP(EQ() + 0.05 * Delta(), measure=prior) for _ in range(m)]
    z_model += [GP(0.1 * Delta(), measure=prior) for _ in range(p - m)]
    y_model = [
        sum([true_basis[j, i] * z_model[i] for i in range(p)], 0) for j in range(p)
    ]

    # Sample some data.
    y = B.concat(*prior.sample(*[p(x) for p in y_model]), axis=1)

    def model(vs, z):
        """No model. Just check the entropy estimator."""
        return 0

    vs = Vars(jnp.float32)

    # Construct PSA estimator.
    kl_m3_estimator = psa(
        model,
        vs,
        y,
        m,
        h,
        markov=3,
        kl_estimator=True,
    )
    kl_m3_estimator(vs)  # Initialise variables.

    # Construct second PSA estimator.
    kl_m1_estimator = psa(
        model,
        vs,
        y,
        m,
        h,
        markov=1,
        kl_estimator=True,
    )
    kl_m1_estimator(vs)  # Initialise variables.

    # Construct UC PSA estimator.
    kl_uc_estimator = psa(
        model,
        vs,
        y,
        m,
        h,
        entropy_conditional=False,
        kl_estimator=True,
    )
    kl_uc_estimator(vs)  # Initialise variables.

    def kl_true(vs):
        """Model likelihood and the true entropy."""
        basis = vs["basis"]
        z_model = [
            sum([basis[i, j] * y_model[i] for i in range(p)], 0) for j in range(m)
        ]
        entropy = cross(*z_model)(MultiInput(*[p(x) for p in z_model])).entropy()
        return (-entropy - model(vs, y @ basis)) / m / B.shape(y)[1]

    def grad_basis(f, vs):
        """Compute the gradient with respect to the basis for an estimator."""
        basis_latent = vs.get_vars("basis")[0]
        vs_copy = vs.copy()

        def to_diff(basis_latent_):
            vs_copy.assign("basis", basis_latent_, differentiable=True)
            return f(vs_copy)

        return jax.grad(to_diff)(basis_latent)

    def norm(x):
        """L2 norm."""
        return B.sqrt(B.sum(x ** 2))

    # Estimate gradient with PSA and true KL.
    grad_psa_m3 = grad_basis(kl_m3_estimator, vs)
    grad_psa_m1 = grad_basis(kl_m1_estimator, vs)
    grad_psa_uc = grad_basis(kl_uc_estimator, vs)
    grad_true = grad_basis(kl_true, vs)

    # Report results.
    out.kv("PSA (M3)", grad_psa_m3)
    out.kv("PSA (M1)", grad_psa_m1)
    out.kv("PSA (UC)", grad_psa_uc)
    out.kv("True", grad_true)
    out.kv("Cosine sim. (M3)", cos_sim(grad_psa_m3, grad_true))
    out.kv("Cosine sim. (M1)", cos_sim(grad_psa_m1, grad_true))
    out.kv("Cosine sim. (UC)", cos_sim(grad_psa_uc, grad_true))

    sims_m3.append(cos_sim(grad_psa_m3, grad_true))
    sims_m1.append(cos_sim(grad_psa_m1, grad_true))
    sims_uc.append(cos_sim(grad_psa_uc, grad_true))


def report(name, x):
    with out.Section(name):
        out.kv("Mean", np.mean(x))
        out.kv("Error", 1.96 * np.std(x) / np.sqrt(len(x)))
        out.kv("Lower error bound", np.mean(x) - 1.96 * np.std(x) / np.sqrt(len(x)))
        out.kv("Upper error bound", np.mean(x) + 1.96 * np.std(x) / np.sqrt(len(x)))


# Report averages.
with out.Section("Average cosine similarities"):
    report("M3", sims_m3)
    report("M1", sims_m1)
    report("UC", sims_uc)
