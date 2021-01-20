import lab.jax as B
from jax.lax import stop_gradient

from .gradient import entropy_gradient_estimator

__all__ = ["psa_kl_estimator"]


def psa_kl_estimator(model_loglik, y, m):
    estimator = entropy_gradient_estimator()

    def kl(vs):
        h = vs.get(shape=(B.shape(y)[1], m), name="h")
        x = y @ h
        x_no_grad = stop_gradient(x)
        g1, g2 = estimator(x_no_grad[1:], x_no_grad[:-1])
        entropy_proxy = B.sum(g1 * x[1:]) + B.sum(g2 * x[:-1])
        return entropy_proxy - model_loglik(vs, x)

    return kl
