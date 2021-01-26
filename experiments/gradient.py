import jax
import jax.numpy as jnp
import lab as B
import numpy as np
import wbml.out as out
from jax.lax import stop_gradient
from stheno.input import MultiInput
from stheno.jax import Measure, GP, Exp, Delta, cross
from varz.jax import Vars
from wbml.experiment import WorkingDirectory

from psa import psa, cos_sim

# Initialise experiment.
wd = WorkingDirectory("_experiments", "gradient")
out.report_time = True
B.epsilon = 1e-5
B.default_dtype = jnp.float32

# Settings of experiment:
x = B.linspace(0, 20, 1000)
m = 2
p = 4
h = 1.0
# More noise makes the conditional estimator less useful!
noise = 0.05

sims_m1 = []
sims_uc = []

mse_m1 = []
mse_uc = []


for i in range(100):
    out.kv("Repetition", i + 1)

    # Sample a true basis.
    true_basis = Vars(jnp.float32).get(shape=(p, p))

    # Build a model for the data.
    prior = Measure()
    z_model = [GP(Exp() + noise * Delta(), measure=prior) for i in range(m)]
    z_model += [GP(noise * Delta(), measure=prior) for _ in range(p - m)]
    y_model = [
        sum([true_basis[j, i] * z_model[i] for i in range(p)], 0) for j in range(p)
    ]

    # Sample some data.
    y = B.concat(*prior.sample(*[p(x) for p in y_model]), axis=1)

    def model(vs, z):
        """No model. Just check the entropy estimator."""
        return 0

    vs = Vars(jnp.float32)
    vs.get(shape=(p, m), name="basis")  # Initialise to random basis.

    # Construct PSA estimator.
    kl_m1_estimator = psa(
        model,
        vs,
        y,
        m,
        h,
        kl_estimator=True,
    )

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

    def kl_true(vs):
        """Model likelihood and the true entropy."""
        basis = vs["basis"]
        basis_no_grad = stop_gradient(basis)
        z_model = [
            sum([basis_no_grad[i, j] * y_model[i] for i in range(p)], 0)
            for j in range(m)
        ]
        z = y @ basis
        z_flat = B.reshape(z.T, -1)
        entropy = -cross(*z_model)(MultiInput(*[p(x) for p in z_model])).logpdf(z_flat)
        return (-entropy - model(vs, z)) / m / B.shape(y)[1]

    def grad_basis(f, vs):
        """Compute the gradient with respect to the basis for an estimator."""
        basis_latent = vs.get_vars("basis")[0]
        vs_copy = vs.copy()

        def to_diff(basis_latent_):
            vs_copy.assign("basis", basis_latent_, differentiable=True)
            return f(vs_copy)

        return jax.grad(to_diff)(basis_latent)

    def mse(x, y):
        """MSE."""
        return B.mean((x - y) ** 2) / B.mean(y ** 2)

    # Estimate gradient with PSA and true KL.
    grad_psa_m1 = grad_basis(kl_m1_estimator, vs)
    grad_psa_uc = grad_basis(kl_uc_estimator, vs)
    grad_true = grad_basis(kl_true, vs)

    # Report results.
    out.kv("PSA (M1)", grad_psa_m1)
    out.kv("PSA (UC)", grad_psa_uc)
    out.kv("True", grad_true)
    out.kv("Cosine sim. (M1)", cos_sim(grad_psa_m1, grad_true))
    out.kv("Cosine sim. (UC)", cos_sim(grad_psa_uc, grad_true))
    out.kv("MSE (M1)", mse(grad_psa_m1, grad_true))
    out.kv("MSE (UC)", mse(grad_psa_uc, grad_true))

    # Save results.
    sims_m1.append(cos_sim(grad_psa_m1, grad_true))
    sims_uc.append(cos_sim(grad_psa_uc, grad_true))
    mse_m1.append(mse(grad_psa_m1, grad_true))
    mse_uc.append(mse(grad_psa_uc, grad_true))


# Report average results.
out.kv("Mean cos. sim. (M1)", np.mean(sims_m1))
out.kv("Mean cos. sim. (UC)", np.mean(sims_uc))
with out.Section("Difference in cosine similarity"):
    diffs = np.array(sims_m1) - np.array(sims_uc)
    out.kv("Mean", np.mean(diffs))
    out.kv(
        "Lower confidence bound",
        np.mean(diffs) - 1.96 * np.std(diffs) / np.sqrt(len(diffs)),
    )
out.kv("Mean MSE (M1)", np.mean(mse_m1))
out.kv("Mean MSE (UC)", np.mean(mse_uc))
with out.Section("Difference in MSE"):
    diffs = np.array(mse_m1) - np.array(mse_uc)
    out.kv("Mean", np.mean(diffs))
    out.kv(
        "Upper confidence bound",
        np.mean(diffs) + 1.96 * np.std(diffs) / np.sqrt(len(diffs)),
    )
