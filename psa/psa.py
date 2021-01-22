import lab.jax as B
from jax.lax import stop_gradient

from .gradient import entropy_gradient_estimator

__all__ = ["psa_kl_estimator"]


def psa_kl_estimator(model_loglik, y, m, orthogonal=True):
    """Construct an estimator of the KL for PSA.

    Args:
        model_loglik (function): Function that takes in a variable container and the
            projected data and computes the likelihood under the model.
        y (matrix): Data.
        m (int): Number of components.
        orthogonal (bool, optional): Use an orthogonal basis. Defaults to `True`.

    Returns
        function: Function that takes in a variable container and estimates the KL.
    """
    estimator = entropy_gradient_estimator()

    def kl(vs):
        # Construct the basis.
        if orthogonal:
            get_h = vs.orthogonal
        else:
            get_h = vs.get
        h = get_h(shape=(B.shape(y)[1], m), name="h")

        # Perform projection.
        x = y @ h

        # Compute proxy objective for the entropy.
        x_no_grad = stop_gradient(x)
        g1, g2 = estimator(x_no_grad[1:], x_no_grad[:-1])
        entropy_proxy = B.sum(g1 * x[1:]) + B.sum(g2 * x[:-1])

        # Assemble KL.
        return entropy_proxy - model_loglik(vs, x)

    return kl
