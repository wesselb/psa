import jax
import lab.jax as B


__all__ = ["stein", "stein_conditional"]


def _k(x, h):
    gram = B.exp(-0.5 * B.pw_dists2(x) / h ** 2)

    def dk(i):
        xi = x[:, i]
        return (xi[:, None] - xi[None, :]) / h ** 2 * gram

    return gram, dk


@jax.jit
def stein(x, h, eta=1e-2):
    """Gradient estimator of the unconditional density.

    Args:
        x (matrix): Samples.
        h (float): Length scale for the kernel.
        eta (float, optional): Regulariser. Defaults to `1e-2`.

    Returns:
        matrix: Estimate of the gradients of `log p(x)` with respect to `x`.
    """
    gram, dk = _k(x, h)
    chol = B.cholesky(B.reg(gram, diag=eta))
    inds = range(B.shape(x)[1])  # Get gradient w.r.t. all dimensions.
    return -B.cholsolve(chol, B.stack(*[B.sum(dk(i), axis=1) for i in inds], axis=1))


@jax.jit
def stein_conditional(x, y, h, h_joint=None, eta=1e-2):
    """Gradient estimator of the unconditional density.

    Args:
        x (matrix): Samples of the first variable.
        y (matrix): Samples of the second variable.
        h (float): Length scale for the kernel.
        h_joint (float, optional): Length scale for the kernel on the joint density.
            Defaults to `h` times the square root of one plus the ratio of the
            dimensionality of `y` and `x`.
        eta (float, optional): Regulariser. Defaults to `1e-2`.

    Returns:
        tuple[matrix,matrix]: Estimates of the gradients of `log p(x|y)` with respect
            to `x` and `y`.
    """
    if h_joint is None:
        h_joint = h * B.sqrt(1 + B.shape(y)[1] / B.shape(x)[1])
    grad_joint = stein(B.concat(x, y, axis=1), h=h_joint, eta=eta)
    grad_joint_x = grad_joint[:, : B.shape(x)[1]]
    grad_joint_y = grad_joint[:, B.shape(x)[1] :]
    grad_marg_y = stein(y, h=h, eta=eta)
    return grad_joint_x, grad_joint_y - grad_marg_y
