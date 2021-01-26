import jax.numpy as jnp
import lab as B
import matplotlib.pyplot as plt
import numpy as np
import wbml.out as out
from stheno.jax import GP, Matern32, Delta
from varz.jax import Vars
from varz.spec import parametrised, Positive, Unbounded
from wbml.data.eeg import load
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

from psa import psa

# Initialise experiment.
wd = WorkingDirectory("_experiments", "eeg")
out.report_time = True
B.epsilon = 1e-6
B.default_dtype = jnp.float32

# Load data.
data = load()[0]

# Settings of experiment:
x = jnp.array(data.index)
y = jnp.array(data)
m = 3
p = data.shape[1]
h = 1.0

orthogonal = False

rate = 1e-2
iters = 2000


@parametrised
def model(
    vs,
    z,
    means: Unbounded(shape=(m,)),
    variances: Positive = 1 * B.ones(m),
    scales: Positive = np.array([0.5, 0.1, 0.05]),
    noises: Positive = 1e-2 * B.ones(m),
):
    logpdf = 0
    for i in range(m):
        kernel = variances[i] * Matern32().stretch(scales[i]) + noises[i] * Delta()
        logpdf += GP(means[i], kernel)(x).logpdf(z[:, i])
    return logpdf


basis_init = Vars(jnp.float32).orthogonal(shape=(p, m))

# Estimate with entropy term.
basis_psa = psa(
    model,
    Vars(jnp.float32),
    y,
    m,
    h,
    iters=iters,
    rate=rate,
    basis_init=basis_init,
    entropy=True,
    orthogonal=orthogonal,
)

# Estimate with unconditional entropy term.
basis_psa_uc = psa(
    model,
    Vars(jnp.float32),
    y,
    m,
    h,
    iters=iters,
    rate=rate,
    basis_init=basis_init,
    entropy=True,
    entropy_conditional=False,
    orthogonal=orthogonal,
)

# Estimate without entropy term.
basis_mle = psa(
    model,
    Vars(jnp.float32),
    y,
    m,
    h,
    iters=iters,
    rate=rate,
    basis_init=basis_init,
    entropy=False,
    orthogonal=orthogonal,
)

out.kv("Basis PSA", basis_psa)
out.kv("Basis PSA (UC)", basis_psa_uc)
out.kv("Basis MLE", basis_mle)
out.kv("Diff.", basis_psa - basis_mle)

plt.figure(figsize=(12, 4))

# Plot PSA result.
z_est = y @ basis_psa
plt.subplot(1, 3, 1)
plt.title("With Conditional Entropy Term")
cmap = plt.get_cmap("tab10")
for i in range(m):
    plt.plot(x, z_est[:, i], c=cmap(i))
tweak(legend=False)

# Plot PSA result.
z_est = y @ basis_psa_uc
plt.subplot(1, 3, 2)
plt.title("With Unconditional Entropy Term")
for i in range(m):
    plt.plot(x, z_est[:, i], c=cmap(i))
tweak(legend=False)

# Plot MLE result.
z_est = y @ basis_mle
plt.subplot(1, 3, 3)
plt.title("Without Entropy Term")
for i in range(m):
    plt.plot(x, z_est[:, i], c=cmap(i))
tweak(legend=False)

plt.savefig(wd.file("result.png"))
plt.show()
