import jax
import jax.numpy as jnp
import lab.jax as B
from stheno import Normal

from psa import construct_gradient_estimator


def test_gradient_estimator():
    estimator = construct_gradient_estimator()

    d = Normal(jnp.array([[1.0, -0.4], [-0.4, 0.8]]))
    x = B.concat(*[d.sample().T for _ in range(1000)], axis=0)

    h = 10
    h_ce = 0.2

    @jax.jit
    @jax.grad
    def true_grad(x):
        d_0given1 = Normal(
            (d.var[0, 1] / d.var[1, 1] * x[1])[None, None],
            (d.var[0, 0] - d.var[0, 1] * d.var[1, 0] / d.var[1, 1])[None, None],
        )
        return d_0given1.logpdf(x[0])

    true_grads = B.stack(*[true_grad(xi) for xi in x], axis=0)
    est_grads = B.concat(*estimator(x[:, 0:1], x[:, 1:2], h, h_ce), axis=1)

    # Assert that the MAE for both is less than 0.1.
    mae = max(B.mean(B.abs(true_grads - est_grads), axis=0))
    assert mae < 0.1
