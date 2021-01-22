import pytest
import jax
import jax.numpy as jnp
import lab.jax as B
from stheno import Normal

from psa import entropy_gradient_estimator, cos_sim


@pytest.mark.parametrize(
    "n, min_cos_sim", [(100, 0.4), (250, 0.6), (750, 0.7), (1500, 0.8)]
)
@pytest.mark.parametrize("correlation", [-0.4, 0, 0.4])
def test_entropy_gradient_estimator_correlated_gaussian(n, min_cos_sim, correlation):
    estimator = entropy_gradient_estimator()

    d = Normal(jnp.array([[1.0, correlation], [correlation, 0.8]]))
    x = d.sample(n).T

    @jax.jit
    @jax.grad
    def true_grad(x):
        conditional = Normal(
            B.uprank(d.var[0, 1] / d.var[1, 1] * x[1]),
            B.uprank(d.var[0, 0] - d.var[0, 1] * d.var[1, 0] / d.var[1, 1]),
        )
        return conditional.logpdf(x[0])

    true_grads = B.stack(*[true_grad(xi) for xi in x], axis=0)
    est_grads = B.concat(*estimator(x[:, 0:1], x[:, 1:2], 1, 0.2), axis=1)
    assert cos_sim(true_grads, est_grads) > min_cos_sim
