import jax
import lab.jax as B
from .util import select_bandwidth


__all__ = ["stein", "stein_conditional"]

def _k(x, h):
    gram = B.exp(-0.5 * B.pw_dists2(x / (h ** 2)))

    def dk(i):
        xi = x[:, i]
        return (xi[:, None] - xi[None, :]) / h[i] ** 2 * gram

    return gram, dk

control_flow_cache = B.ControlFlowCache() # trying to use https://wesselb.github.io/2021/01/19/linear-models-with-stheno-and-jax.html
# but failing

@jax.jit
def _k_jitted(x, h):
    with control_flow_cache:
        return _k(x, h)

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
    scale = select_bandwidth(x, None) # ignoring h in an ugly way
    gram, dk = _k_jitted(x, scale)
    chol = B.cholesky(B.reg(gram, diag=eta))
    inds = range(B.shape(x)[1])  # Get gradient w.r.t. all dimensions.
    return -B.cholsolve(chol, B.stack(*[B.sum(dk(i), axis=1) for i in inds], axis=1))


@jax.jit
def stein_conditional(x, y, h, eta=1e-2):
    """Gradient estimator of the unconditional density.

    Args:
        x (matrix): Samples of the first variable.
        y (matrix): Samples of the second variable.
        h (float): Length scale for the kernel.
        eta (float, optional): Regulariser. Defaults to `1e-2`.

    Returns:
        tuple[matrix,matrix]: Estimates of the gradients of `log p(x|y)` with respect
            to `x` and `y`.
    """
    scale = select_bandwidth(B.concat(x, y, axis=1), None) # ignoring h in an ugly way
    grad_joint = stein(B.concat(x, y, axis=1), h=scale, eta=eta)
    grad_joint_x = grad_joint[:, : B.shape(x)[1]]
    grad_joint_y = grad_joint[:, B.shape(x)[1] :]
    scale = select_bandwidth(y, None)
    grad_marg_y = stein(y, h=scale, eta=eta)
    return grad_joint_x, grad_joint_y - grad_marg_y
