import jax.numpy as jnp
import lab as B
import matplotlib.pyplot as plt
import wbml.out as out
from stheno.jax import GP, EQ, Delta
from varz.jax import minimise_adam, Vars
from varz.spec import parametrised, Positive
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

from psa import psa_kl_estimator, pair_signals

# Initialise experiment.
wd = WorkingDirectory("_experiments", "toy")
out.report_time = True
B.epsilon = 1e-6
B.default_dtype = jnp.float32

x = B.linspace(0, 10, 250)
m = 2
p = 4
true_basis = Vars(jnp.float32).orthogonal(shape=(p, p))

z_model = [GP(0.95 * EQ() + 0.05 * Delta()) for _ in range(m)]
z_model += [GP(Delta()) for _ in range(p - m)]
z = B.concat(*[p(x).sample() for p in z_model], axis=1)
y = z @ true_basis.T


@parametrised
def model(
    vs,
    z,
    variances: Positive = 0.5 * B.ones(m),
    scales: Positive = 1 * B.ones(m),
    noises: Positive = 0.5 * B.ones(m),
):
    logpdf = 0
    for i in range(m):
        kernel = variances[i] * EQ().stretch(scales[i]) + noises[i] * Delta()
        logpdf += GP(kernel)(x).logpdf(z[:, i])
    return logpdf


basis_init = Vars(jnp.float32).orthogonal(shape=(p, m), name="basis")
iters = 2000

# Estimate with entropy term.
vs = Vars(jnp.float32)
objective = psa_kl_estimator(
    model, y, m, h=1.0, h_ce=0.2, eta=1e-2, basis_init=basis_init, entropy=True
)
minimise_adam(objective, vs, iters=iters, trace=True, jit=True, rate=5e-2)
basis_psa = vs["basis"]

# Estimate without entropy term.
vs = Vars(jnp.float32)
objective = psa_kl_estimator(
    model, y, m, h=1.0, h_ce=0.2, eta=1e-2, basis_init=basis_init, entropy=False
)
minimise_adam(objective, vs, iters=iters, trace=True, jit=True, rate=5e-2)
basis_mle = vs["basis"]

out.kv("Basis PSA", basis_psa)
out.kv("Basis MLE", basis_mle)
out.kv("Diff.", basis_psa - basis_mle)

plt.figure(figsize=(10, 4))

# Plot PSA result.
z_est, z_true = pair_signals(y @ basis_psa, z[:, :m])
plt.subplot(1, 2, 1)
plt.title("With Entropy Term")
for i in range(m):
    plt.plot(x, z_est[:, i], ls="--")
    plt.plot(x, z_true[:, i], ls="-")
tweak(legend=False)

# Plot MLE result.
z_est, z_true = pair_signals(y @ basis_mle, z[:, :m])
plt.subplot(1, 2, 2)
plt.title("Without Entropy Term")
for i in range(m):
    plt.plot(x, z_est[:, i], ls="--")
    plt.plot(x, z_true[:, i], ls="-")
tweak(legend=False)

plt.savefig(wd.file("result.png"))
plt.show()
