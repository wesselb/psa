import jax
import lab.jax as B
from matrix import LowRank

__all__ = ["stein", "stein_conditional"]


def estimate_scale(x):
    """Estimate the bandwidth.

    Args:
        x (matrix): Inputs of observations.

    Returns:
        vector: Bandwidths.
    """
    return B.sqrt(B.mean((x - B.mean(x, axis=0)) ** 2, axis=0))


def _k(x, h):
    gram = B.exp(-0.5 * B.pw_dists2(x / h))

    def dk(i):
        if B.isscalar(h):
            hi = h
        else:
            hi = h[i]
        xi = x[:, i]
        return -(xi[None, :] - xi[:, None]) / hi ** 2 * gram

    return gram, dk


def _k_cross(x, y, h):
    return B.exp(-0.5 * B.pw_dists2(x / h, y / h))


@jax.jit
def stein(x, h=None, eta=1e-2, inducing_inds=None):
    """Gradient estimator of the unconditional density.

    Args:
        x (matrix): Samples.
        h (float, optional): Length scale for the kernel. Defaults to using
            :func:`.stein.estimate_scale`.
        eta (float, optional): Regulariser. Defaults to `1e-2`.
        inducing_inds (vector, optional): Indices of the data points to use as
            inducing points for the Nystrom approximation.

    Returns:
        matrix: Estimate of the gradients of `log p(x)` with respect to `x`.
    """
    if h is None:
        h = estimate_scale(x)

    gram, dk = _k(x, h)
    inds = range(B.shape(x)[1])  # Get gradient w.r.t. all dimensions.
    y = B.stack(*[B.sum(dk(i), axis=1) for i in inds], axis=1)

    if inducing_inds is None:
        # Don't use any approximation.
        chol = B.cholesky(B.reg(gram, diag=eta))
        return -B.cholsolve(chol, y)
    else:
        # Use Nystrom approximation.
        x_ind = x[inducing_inds]
        gram_ind, _ = _k(x_ind, h)
        chol = B.cholesky(B.reg(gram_ind))
        # Be sure to move `eta` outside for better numerical stability.
        lr = LowRank(B.transpose(B.trisolve(chol, _k_cross(x_ind, x, h))) / B.sqrt(eta))
        wb = lr + B.fill_diag(B.cast(B.dtype(lr), 1), B.shape(lr)[0])
        return -B.dense(B.mm(B.pd_inv(wb), y)) / eta


@jax.jit
def stein_conditional(x, y, h=None, h_joint=None, eta=1e-2, inducing_inds=None):
    """Gradient estimator of the unconditional density.

    Args:
        x (matrix): Samples of the first variable.
        y (matrix): Samples of the second variable.
        h (float, optional): Length scale for the kernel. Defaults to using
            :func:`.stein.estimate_scale`.
        h_joint (float, optional): Length scale for the kernel on the joint density.
            Defaults to using :func:`.stein.estimate_scale`.
        eta (float, optional): Regulariser. Defaults to `1e-2`.
        inducing_inds (vector, optional): Indices of the data points to use as
            inducing points for the Nystrom approximation.

    Returns:
        tuple[matrix,matrix]: Estimates of the gradients of `log p(x|y)` with respect
            to `x` and `y`.
    """
    if h is None:
        h = estimate_scale(x)
    if h_joint is None:
        h_joint = estimate_scale(B.concat(x, y, axis=1))

    grad_joint = stein(
        B.concat(x, y, axis=1), h=h_joint, eta=eta, inducing_inds=inducing_inds
    )
    grad_joint_x = grad_joint[:, : B.shape(x)[1]]
    grad_joint_y = grad_joint[:, B.shape(x)[1] :]
    grad_marg_y = stein(y, h=h, eta=eta, inducing_inds=inducing_inds)
    return grad_joint_x, grad_joint_y - grad_marg_y
