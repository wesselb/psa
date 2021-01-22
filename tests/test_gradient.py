import pytest
import jax
import jax.numpy as jnp
import lab.jax as B
from stheno import Normal

from psa import entropy_gradient_estimator


@pytest.mark.parametrize("correlation", [-0.4, 0, 0.4])
def test_entropy_gradient_estimator(correlation):
    estimator = entropy_gradient_estimator()

    d = Normal(jnp.array([[1.0, correlation], [correlation, 0.8]]))
    x = d.sample(1000).T

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

    # Assert that the MAE for both is less than 0.2.
    mae = max(B.mean(B.abs(true_grads - est_grads), axis=0))
    assert mae < 0.2
