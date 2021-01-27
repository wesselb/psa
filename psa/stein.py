import jax
import lab.jax as B
import jax.random as rnd 

key = rnd.PRNGKey(666)

__all__ = ["stein", "stein_conditional", "stein_nystrom", "stein_conditional_nystrom"]


def _k(x, h):
    gram = B.exp(-0.5 * B.pw_dists2(x) / h ** 2)

    def dk(i):
        xi = x[:, i]
        return (xi[:, None] - xi[None, :]) / h ** 2 * gram

    return gram, dk

# Adapting from https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf
def _nystrom_k(x, h, n_samples, m, eta=1e-2):  # m is the rank of the approx.
    perm = rnd.permutation(key, x)

    # This is likely a terrible way of doing dynamic indexing, but I don't know a better one
    # def slice_1(M, i):
    #     return M[:i, :]

    # def slice_2(M, i):
    #     return M[i:, :]

    # def slice_3(M, i):
    #     return M[:, :i]

    # def slice_4(v, i):
    #     return v[:i]

    samples = perm[:n_samples, :] # randomly draw m samples
    # samples = jax.jit(slice_1, static_argnums=(0,))(perm, n_samples) # randomly draw m samples

    lr_gram = B.exp(-0.5 * B.pw_dists2(samples) / h ** 2) # Build low-rank Gram

    U, S, V = B.svd(B.reg(lr_gram, diag=eta)) 

    # rest = jax.jit(slice_2, static_argnums=(0,))(perm, n_samples)
    # gram_offd = B.exp(-0.5 * B.pw_dists2(rest, samples) / h ** 2)
    gram_offd = B.exp(-0.5 * B.pw_dists2(perm[n_samples:, :], samples) / h ** 2)

    # Um = jax.jit(slice_3, static_argnums=(0,))(U, m)
    # Sm = jax.jit(slice_4, static_argnums=(0,))(S, M)
    U_ = B.sqrt(n_samples / len(x)) * gram_offd @ U[:, :m] / S[:m]
    S_ = len(x) / n_samples * S[:m]
    # U_ = B.sqrt(n_samples / len(x)) * gram_offd @ Um / Sm
    # S_ = len(x) / n_samples * Sm

    def dk(i):
        xi = x[:, i]
        return (xi[:, None] - xi[None, :]) / h ** 2 * U_ @ S_ @ B.transpose(U_)
    
    return U_, S_, dk


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

# Duplicating functions just so I avoid using `if`s because of JAX
@jax.jit
def stein_nystrom(x, h, n_samples, m, eta=1e-2):
    """Gradient estimator of the unconditional density using the Nyström approximation.

    Args:
        x (matrix): Samples.
        h (float): Length scale for the kernel.
        n_samples (int): Number of samples to be used to approximate the kernel.
        m (int): Rank of the kernel approximation. Must be smaller than n_samples.
        eta (float, optional): Regulariser. Defaults to `1e-2`.

    Returns:
        matrix: Estimate of the gradients of `log p(x)` with respect to `x`.
    """
    jitted = jax.jit(_nystrom_k, static_argnums=(2, 3))
    # U, S, dk = _nystrom_k(x, h, n_samples, m, eta=eta)
    U, S, dk = jitted(x, h, n_samples, m, eta=eta)
    chol = U @ B.sqrt(S)
    inds = range(B.shape(x)[1])  # Get gradient w.r.t. all dimensions.
    return -B.cholsolve(chol, B.stack(*[B.sum(dk(i), axis=1) for i in inds], axis=1))


@jax.jit
def stein_conditional(x, y, h, h_joint=None, eta=1e-2):
    """Gradient estimator of the conditional density.

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

@jax.jit
def stein_conditional_nystrom(x, y, h, n_samples, m, h_joint=None, eta=1e-2):
    """Gradient estimator of the conditional density using the Nyström approximation.

    Args:
        x (matrix): Samples of the first variable.
        y (matrix): Samples of the second variable.
        h (float): Length scale for the kernel.
        n_samples (int): Number of samples to be used to approximate the kernel.
        m (int): Rank of the kernel approximation. Must be smaller than n_samples.
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
    grad_joint = stein_nystrom(B.concat(x, y, axis=1), h_joint, n_samples, m, eta=eta)
    grad_joint_x = grad_joint[:, : B.shape(x)[1]]
    grad_joint_y = grad_joint[:, B.shape(x)[1] :]
    grad_marg_y = stein_nystrom(y, h, n_samples, m, eta=eta)
    return grad_joint_x, grad_joint_y - grad_marg_y
