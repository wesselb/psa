import numpy as np
import lab.jax as B
from jax.lax import stop_gradient

from .gradient import entropy_gradient_estimator

__all__ = ["pair_signals", "psa_kl_estimator"]


def pair_signals(x, y):
    """Pair two collections of signals in the best way. Flipping signs is allowed.

    Args:
        x (matrix): First collection of signals. Different columns correspond to
            different signals.
        y (matrix): Second collection of signals. Different columns correspond to
            different signals.

    Returns:
        tuple[matrix, matrix]: Reordering of `x` and `y`, corresponding to the produced
            pairing.
    """
    if B.shape(x)[1] != B.shape(y)[1]:
        raise ValueError("An equal number of signals must be given.")

    # Signals are given as columns. Make them rows.
    x = B.transpose(x)
    y = B.transpose(y)

    # We'll work in NumPy.
    x = B.to_numpy(x)
    y = B.to_numpy(y)

    n = B.shape(x)[0]

    # First, check if it helps to flip the signs of `x`.
    for i in range(n):
        if B.min(B.pw_dists(-x[i : i + 1], y)) < B.min(B.pw_dists(x[i : i + 1], y)):
            x[i] = -x[i]

    # Do a greedy matching. Not optimal, but sufficient for our purposes.
    x_matched = []
    y_matched = []
    for i in range(n):
        x_left = sorted(set(range(n)) - set(x_matched))
        y_left = sorted(set(range(n)) - set(y_matched))
        x_left_i, y_left_i = np.unravel_index(
            np.argmin(B.pw_dists(x[x_left], y[y_left])), (n - i, n - i)
        )
        x_matched.append(x_left[x_left_i])
        y_matched.append(y_left[y_left_i])

    # Returns matched signals as columns.
    return B.transpose(x[x_matched]), B.transpose(y[y_matched])


def psa_kl_estimator(
    model_loglik,
    y,
    m,
    orthogonal=True,
    entropy=True,
    basis_init=None,
    h=None,
    h_ce=None,
    eta=1e-2,
):
    """Construct an estimator of the KL for PSA.

    Args:
        model_loglik (function): Function that takes in a variable container and the
            projected data and computes the likelihood under the model.
        y (matrix): Data.
        m (int): Number of components.
        orthogonal (bool, optional): Use an orthogonal basis. Defaults to `True`.
        entropy (bool, optional): Estimate the entropy. Defaults to `True`.
        basis_init (matrix, optional): Initialisation for the basis.
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
        basis = get_basis(init=basis_init, shape=(B.shape(y)[1], m), name="basis")

        # Perform projection.
        x = y @ basis

        if entropy:
            # Compute proxy objective for the entropy.
            x_no_grad = stop_gradient(x)
            g1, g2 = estimator(x_no_grad[1:], x_no_grad[:-1], h, h_ce, eta)
            entropy_proxy = -B.sum(g1 * x[1:]) - B.sum(g2 * x[:-1])

            # Assemble KL.
            return -entropy_proxy - model_loglik(vs, x)
        else:
            # Not using the entropy term.
            return -model_loglik(vs, x)

    return kl
