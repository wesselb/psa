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
x = B.linspace(0, 10, 300)
m = 2
p = 4

h_x = 1.0
h_y = 1.0
h_ce = 0.2

out.kv("h_x", h_x)
out.kv("h_y", h_y)
out.kv("h_ce", h_ce)

sims_m1 = []
sims_m3 = []
sims_uc = []


for i in range(10):
    out.kv("Rep.", i + 1)

    # Sample a true basis.
    true_basis = Vars(jnp.float32).orthogonal(shape=(p, p))

    # Build a model for the data.
    prior = Measure()
    z_model = [GP(EQ() + 0.1 * Delta(), measure=prior) for _ in range(m)]
    z_model += [GP(Delta(), measure=prior) for _ in range(p - m)]
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
        markov=3,
        h_x=h_x,
        h_y=h_y,
        h_ce=h_ce,
        return_kl_estimator=True,
    )
    kl_m3_estimator(vs)  # Initialise variables.

    # Construct second PSA estimator.
    kl_m1_estimator = psa(
        model,
        vs,
        y,
        m,
        markov=1,
        h_x=h_x,
        h_y=h_y,
        h_ce=h_ce,
        return_kl_estimator=True,
    )
    kl_m1_estimator(vs)  # Initialise variables.

    # Construct UC PSA estimator.
    kl_uc_estimator = psa(
        model,
        vs,
        y,
        m,
        h_x=h_x,
        entropy_conditional=False,
        return_kl_estimator=True,
    )
    kl_uc_estimator(vs)  # Initialise variables.

    def kl_true(vs):
        """Model likelihood and the true entropy."""
        basis = vs["basis"]
        z_model = [
            sum([basis[i, j] * y_model[i] for i in range(p)], 0) for j in range(m)
        ]
        entropy = cross(*z_model)(MultiInput(*[p(x) for p in z_model])).entropy()
        return -entropy - model(vs, y @ basis)

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

# Report averages.
out.kv("Avg. cosine sim. (M3)", np.mean(sims_m3))
out.kv("Avg. cosine sim. (M1)", np.mean(sims_m1))
out.kv("Avg. cosine sim. (UC)", np.mean(sims_uc))
