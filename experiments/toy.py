import jax.numpy as jnp
import lab as B
import matplotlib.pyplot as plt
from stheno.jax import Measure, GP, EQ
from varz.jax import minimise_adam, Vars
from varz.spec import parametrised, Positive
from wbml.plot import tweak

from psa import psa_kl_estimator

B.epsilon = 1e-6
B.set_random_seed(0)
B.default_dtype = jnp.float32


x = B.linspace(0, 10, 100)
z = GP(EQ())(x).sample(2)
h = Vars(jnp.float32).orthogonal(shape=(10, 2))
y = z @ h.T + 0.1 * B.randn(100, 10) @ (B.eye(10) - h @ h.T)


@parametrised
def model(
    vs,
    z,
    var1: Positive = 1,
    var2: Positive = 1,
    scale1: Positive = 1,
    scale2: Positive = 1,
):
    prior = Measure()
    y1 = GP(var1 * EQ().stretch(scale1), measure=prior)
    y2 = GP(var2 * EQ().stretch(scale2), measure=prior)
    return y1(x).logpdf(z[:, 0]) + y2(x).logpdf(z[:, 1])


vs = Vars(jnp.float32)

# Initialise to a bad basis
vs.orthogonal((B.eye(10) - h @ h.T)[:, :2], name="h")

minimise_adam(
    psa_kl_estimator(model, y, 2), vs, iters=2000, trace=True, jit=True, rate=5e-2
)

plt.figure()
plt.plot(y @ vs["h"])
plt.plot(z, label="True", ls="--")
tweak(legend=True)
plt.show()
