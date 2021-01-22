import jax.numpy as jnp
import lab as B
import matplotlib.pyplot as plt
from stheno.jax import Measure, GP, EQ, Delta
from varz.jax import minimise_adam, Vars
from varz.spec import parametrised, Positive
from wbml.plot import tweak

from psa import psa_kl_estimator

B.epsilon = 1e-6
B.set_random_seed(1)
B.default_dtype = jnp.float32


x = B.linspace(0, 10, 500)
m = 2
p = 5
true_basis = Vars(jnp.float32).orthogonal(shape=(p, p))

prior = Measure()
z_model = [GP(0.95 * EQ() + 0.05 * Delta(), measure=prior) for _ in range(m)]
z_model += [GP(Delta(), measure=prior) for _ in range(p - m)]
z = B.concat(*prior.sample(*[p(x) for p in z_model]), axis=1)
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


vs = Vars(jnp.float32)
psa_objective = psa_kl_estimator(model, y, m)
minimise_adam(psa_objective, vs, iters=200, trace=True, jit=True, rate=5e-2)

plt.figure()
plt.plot(y @ vs["basis"], label="Estimated", ls="--")
plt.plot(z[:, :m], label="True", ls="-")
tweak(legend=True)
plt.show()
