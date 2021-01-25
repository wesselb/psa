import jax.numpy as jnp
import lab as B
import matplotlib.pyplot as plt
import wbml.out as out
from stheno.jax import GP, EQ, Delta
from varz.jax import Vars
from varz.spec import parametrised, Positive
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

from psa import psa, pair_signals

# Initialise experiment.
wd = WorkingDirectory("_experiments", "pathology", seed=1)
out.report_time = True
B.epsilon = 1e-6
B.default_dtype = jnp.float32

# Settings of experiment:
x = B.linspace(0, 10, 1000)
m = 2
p = 4
markov = 1
h = 1.0
rate = 5e-2
iters = 500

# Sample some data.
true_basis = Vars(jnp.float32).orthogonal(shape=(p, p))
z_model = [GP(EQ() + 0.01 * Delta()) for _ in range(m)]
z_model += [GP(0.1 * Delta()) for _ in range(p - m)]
z = B.concat(*[p(x).sample() for p in z_model], axis=1)
y = z @ true_basis.T


@parametrised
def model(
    vs,
    z,
    variances: Positive = 0.5 * B.ones(m),
    scales: Positive = B.ones(m),
    noises: Positive = 0.5 * B.ones(m),
):
    logpdf = 0
    for i in range(m):
        kernel = variances[i] * EQ().stretch(scales[i]) + noises[i] * Delta()
        logpdf += GP(kernel)(x).logpdf(z[:, i])
    return logpdf


basis_init = Vars(jnp.float32).orthogonal(shape=(p, m))

# Estimate with entropy term.
vs = Vars(jnp.float32)
basis_psa = psa(
    model,
    vs,
    y,
    m,
    h,
    iters=iters,
    rate=rate,
    markov=markov,
    basis_init=basis_init,
    entropy=True,
    orthogonal=False,
)

# Estimate with unconditional entropy term.
vs = Vars(jnp.float32)
basis_psa_uc = psa(
    model,
    vs,
    y,
    m,
    h,
    iters=iters,
    rate=rate,
    markov=markov,
    basis_init=basis_init,
    entropy=True,
    entropy_conditional=False,
    orthogonal=False,
)

# Estimate without entropy term.
vs = Vars(jnp.float32)
basis_mle = psa(
    model,
    vs,
    y,
    m,
    h,
    iters=iters,
    rate=rate,
    markov=markov,
    basis_init=basis_init,
    entropy=False,
    orthogonal=False,
)

out.kv("Basis PSA", basis_psa)
out.kv("Basis PSA (UC)", basis_psa_uc)
out.kv("Basis MLE", basis_mle)
out.kv("Diff.", basis_psa - basis_mle)

plt.figure(figsize=(12, 4))

# Plot PSA result.
z_est, z_true = pair_signals(y @ basis_psa, z[:, :m])
plt.subplot(1, 3, 1)
plt.title("With Conditional Entropy Term")
cmap = plt.get_cmap("tab10")
for i in range(m):
    plt.plot(x, z_est[:, i], c=cmap(i))
    plt.plot(x, z_true[:, i], alpha=0.5, c=cmap(i))
tweak(legend=False)

# Plot PSA result.
z_est, z_true = pair_signals(y @ basis_psa_uc, z[:, :m])
plt.subplot(1, 3, 2)
plt.title("With Unconditional Entropy Term")
for i in range(m):
    plt.plot(x, z_est[:, i], c=cmap(i))
    plt.plot(x, z_true[:, i], alpha=0.5, c=cmap(i))
tweak(legend=False)

# Plot MLE result.
z_est, z_true = pair_signals(y @ basis_mle, z[:, :m])
plt.subplot(1, 3, 3)
plt.title("Without Entropy Term")
for i in range(m):
    plt.plot(x, z_est[:, i], c=cmap(i))
    plt.plot(x, z_true[:, i], alpha=0.5, c=cmap(i))
tweak(legend=False)

plt.savefig(wd.file("result.png"))
plt.show()
