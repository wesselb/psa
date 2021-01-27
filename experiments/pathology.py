import sys

import jax.numpy as jnp
import lab as B
import matplotlib.pyplot as plt
import wbml.out as out
from stheno.jax import GP, EQ, Delta, SparseObs, Measure
from varz.jax import Vars
from varz.spec import parametrised, Positive
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

from psa import psa, pair_signals

# Get seed from command line settings.
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 0

# Initialise experiment.
wd = WorkingDirectory("_experiments", "pathology", f"{seed}", seed=seed)
out.report_time = True

# General settings:
B.default_dtype = jnp.float64

# Settings of data:
x = B.linspace(0, 20, 3000)
m = 2
p = 4
kernels = [EQ()] * m  # Kernels of latent processes.
noise = 0.01

# Settings of model approximation:
n_inducing = 200

# Settings of PSA:
orthogonal = False
h = None  # `None` means that the bandwidth is automatically determined.
eta = 1e-2  # Regularisation of PSA
rate = 5e-2  # Learning rate
iters = 500  # Number of optimisation iterations
batch_size = 200  # Nystrom approximation

# Define a true model and sample some data.
true_basis = Vars(B.default_dtype).get(shape=(p, p))
z_model = [GP(kernels[i] + noise * Delta()) for i in range(m)]
z_model += [GP(Delta()) for _ in range(p - m)]
z = B.concat(*[p(x).sample() for p in z_model], axis=1)
y = z @ true_basis.T

# Determine locations of inducing points for GPs.
x_ind = B.linspace(min(x), max(x), n_inducing)


@parametrised
def model(
    vs,
    z,
    variances: Positive = B.ones(m),
    scales: Positive = B.ones(m),
    noises: Positive = B.ones(m),
):
    logpdf = 0
    for i in range(m):
        prior = Measure()
        f = GP(variances[i] * kernels[i].stretch(scales[i]), measure=prior)
        e = GP(noises[i] * Delta(), measure=prior)
        logpdf += SparseObs(f(x_ind), e, f(x), z[:, i]).elbo(prior)
    return logpdf


basis_init = Vars(B.default_dtype).orthogonal(shape=(p, m))

# Estimate with entropy term.
vs = Vars(B.default_dtype)
basis_psa = psa(
    model,
    vs,
    y,
    m,
    h,
    eta=eta,
    batch_size=batch_size,
    iters=iters,
    rate=rate,
    basis_init=basis_init,
    entropy=True,
    entropy_conditional=True,
    orthogonal=orthogonal,
)

# # Estimate with unconditional entropy term.
# vs = Vars(jnp.float32)
# basis_psa_uc = psa(
#     model,
#     vs,
#     y,
#     m,
#     h,
#     eta=eta,
#     iters=iters,
#     rate=rate,
#     basis_init=basis_init,
#     entropy=True,
#     entropy_conditional=False,
#     orthogonal=orthogonal,
# )
#
# # Estimate without entropy term.
# vs = Vars(jnp.float32)
# basis_mle = psa(
#     model,
#     vs,
#     y,
#     m,
#     h,
#     eta=eta,
#     iters=iters,
#     rate=rate,
#     basis_init=basis_init,
#     entropy=False,
#     orthogonal=orthogonal,
# )

out.kv("Basis PSA", basis_psa)
# out.kv("Basis PSA (UC)", basis_psa_uc)
# out.kv("Basis MLE", basis_mle)
# out.kv("Diff.", basis_psa - basis_mle)

plt.figure(figsize=(12, 4))

# Plot PSA result.
z_est, z_true = pair_signals(y @ basis_psa, z[:, :m])
plt.subplot(1, 3, 1)
plt.title("PSA")
cmap = plt.get_cmap("tab10")
for i in range(m):
    plt.plot(x, z_est[:, i], c=cmap(i))
    plt.plot(x, z_true[:, i], alpha=0.5, c=cmap(i))
tweak(legend=False)

# # Plot UC PSA result.
# z_est, z_true = pair_signals(y @ basis_psa_uc, z[:, :m])
# plt.subplot(1, 3, 2)
# plt.title("PSA (UC)")
# for i in range(m):
#     plt.plot(x, z_est[:, i], c=cmap(i))
#     plt.plot(x, z_true[:, i], alpha=0.5, c=cmap(i))
# tweak(legend=False)
#
# # Plot MLE result.
# z_est, z_true = pair_signals(y @ basis_mle, z[:, :m])
# plt.subplot(1, 3, 3)
# plt.title("MLE")
# for i in range(m):
#     plt.plot(x, z_est[:, i], c=cmap(i))
#     plt.plot(x, z_true[:, i], alpha=0.5, c=cmap(i))
# tweak(legend=False)

plt.savefig(wd.file("result.png"))
plt.show()
