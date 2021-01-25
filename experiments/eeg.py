import numpy as np
import jax.numpy as jnp
import lab as B
import matplotlib.pyplot as plt
import wbml.out as out
from psa import psa, pair_signals
from stheno.jax import GP, Matern32, Delta
from varz.jax import Vars
from varz.spec import parametrised, Positive
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak
from wbml.data.eeg import load

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

orthogonal = True
markov = 1
h_x = 1.0
h_y = 1.0
h_ce = 0.2


@parametrised
def model(
    vs,
    z,
    variances: Positive = 1 * B.ones(m),
    scales: Positive = np.array([0.1, 0.05, 0.01]),
    noises: Positive = 1e-2 * B.ones(m),
):
    logpdf = 0
    for i in range(m):
        kernel = variances[i] * Matern32().stretch(scales[i]) + noises[i] * Delta()
        logpdf += GP(kernel)(x).logpdf(z[:, i])
    return logpdf


basis_init = Vars(jnp.float32).orthogonal(shape=(p, m))
rate = 5e-2
iters = 1000
batch_size = 100

# Estimate with entropy term.
vs = Vars(jnp.float32)
basis_psa = psa(
    model,
    vs,
    y,
    m,
    iters=iters,
    rate=rate,
    batch_size=batch_size,
    markov=markov,
    h_x=h_x,
    h_y=h_y,
    h_ce=h_ce,
    basis_init=basis_init,
    entropy=True,
    orthogonal=orthogonal,
)

# Estimate with unconditional entropy term.
vs = Vars(jnp.float32)
basis_psa_uc = psa(
    model,
    vs,
    y,
    m,
    iters=iters,
    rate=rate,
    markov=markov,
    h_x=h_x,
    h_y=h_y,
    h_ce=h_ce,
    basis_init=basis_init,
    entropy=True,
    entropy_conditional=False,
    orthogonal=orthogonal,
)

# Estimate without entropy term.
vs = Vars(jnp.float32)
basis_mle = psa(
    model,
    vs,
    y,
    m,
    iters=iters,
    rate=rate,
    markov=markov,
    h_x=h_x,
    h_y=h_y,
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