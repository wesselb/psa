import jax
import jax.numpy as jnp
import lab.jax as B
import numpy as np
import pytest
from stheno.jax import Normal


from psa import cos_sim, stein_conditional


@pytest.mark.parametrize(
    "n, min_cos_sim", [(100, 0.4), (250, 0.55), (750, 0.7), (1500, 0.8)]
)
@pytest.mark.parametrize("correlation", [-0.4, 0, 0.4])
@pytest.mark.parametrize("nystrom", [True, False])
def test_entropy_gradient_estimator_correlated_gaussian(
    n, min_cos_sim, correlation, nystrom
):
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

    if nystrom:
        inducing_inds = np.arange(n)
    else:
        inducing_inds = None

    true_grads = B.stack(*[true_grad(xi) for xi in x], axis=0)
    est_grads = B.concat(
        *stein_conditional(
            x[:, 0:1], x[:, 1:2], h=1.0, h_joint=1.0, inducing_inds=inducing_inds
        ),
        axis=1
    )

    assert cos_sim(true_grads, est_grads) > min_cos_sim
