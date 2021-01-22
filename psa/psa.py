import lab.jax as B
from jax.lax import stop_gradient

from .gradient import entropy_gradient_estimator

__all__ = ["psa_kl_estimator"]


def psa_kl_estimator(model_loglik, y, m, orthogonal=True, h=None, h_ce=None, eta=1e-2):
    """Construct an estimator of the KL for PSA.

    Args:
        model_loglik (function): Function that takes in a variable container and the
            projected data and computes the likelihood under the model.
        y (matrix): Data.
        m (int): Number of components.
        orthogonal (bool, optional): Use an orthogonal basis. Defaults to `True`.
        h (float, optional): Length scale for the kernels over the arguments.
            Defaults to a median-based value.
        h_ce (float, optional):  Length scale for the kernel for the conditional
            expectation. Defaults to a median-based value.
        eta (float, optional): L2 regulariser. Defaults to `1e-2`.

    Returns
        function: Function that takes in a variable container and estimates the KL.
    """
    estimator = entropy_gradient_estimator()

    def kl(vs):
        # Construct the basis.
        if orthogonal:
            get_basis = vs.orthogonal
        else:
            get_basis = vs.get
        basis = get_basis(shape=(B.shape(y)[1], m), name="basis")

        # Perform projection.
        x = y @ basis

        # Compute proxy objective for the entropy.
        x_no_grad = stop_gradient(x)
        g1, g2 = estimator(x_no_grad[1:], x_no_grad[:-1], h, h_ce, eta)
        entropy_proxy = -B.sum(g1 * x[1:]) - B.sum(g2 * x[:-1])

        # Assemble KL.
        return -entropy_proxy - model_loglik(vs, x)

    return kl
