import jax
import jax.numpy as jnp
import lab.jax as B
import numpy as np
import wbml.out as out
from jax.lax import stop_gradient
from varz import ADAM

from .stein import stein, stein_conditional

__all__ = ["pair_signals", "psa"]


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
    x, y = B.transpose(x[x_matched]), B.transpose(y[y_matched])

    # Sort by magnitudes of `y`.
    perm = np.argsort(B.sum(y ** 2, axis=0))
    return x[:, perm], y[:, perm]


def _sample_batch_indices(y, batch_size):
    if batch_size is None:
        return jnp.arange(B.shape(y)[0])
    else:
        return jnp.array(np.sort(np.random.permutation(B.shape(y)[0])[:batch_size]))


def psa(
    model_loglik,
    vs,
    y,
    m,
    h=None,
    eta=1e-2,
    batch_size=None,
    iters=500,
    rate=5e-2,
    kl_estimator=False,
    orthogonal=True,
    entropy=True,
    entropy_conditional=True,
    basis_init=None,
    markov=1,
):
    """Perform predictable subspace analysis (PSA).

    Args:
        model_loglik (function): Function that takes in a variable container and the
            projected data and computes the likelihood under the model.
        vs (:class:`varz.Vars`): Variable container.
        y (matrix): Data.
        m (int): Number of components.
        h (float, optional): Length scale for the kernel. Defaults to using
            :func:`.stein.estimate_scale`.
        eta (float, optional): L2 regulariser. Defaults to `1e-2`.
        batch_size (int, optional): Number of data points to subsample for the
            Nystrom approximation. Defaults to not using the Nystrom approximation.
        iters (int, optional): Number of optimisation iterations. Defaults to `500`.
        rate (float, optional): Learning rate. Defaults to `5e-2`.
        kl_estimator (bool, optional): Return the KL estimator instead of performing
            the optimisation. Defaults to `False`.
        orthogonal (bool, optional): Use an orthogonal basis. Defaults to `True`.
        entropy (bool, optional): Estimate the entropy. Defaults to `True`.
        entropy_conditional (bool, optional): Estimate the entropy of the conditional
            densities. Defaults to `True`.
        basis_init (matrix, optional): Initialisation for the basis.
        markov (int, optional): Order of Markov assumption. Defaults to `1`.

    Returns
        matrix: Estimated basis.
    """

    def kl(vs, inducing_inds):
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
            splits = [x[i : B.shape(x)[0] - markov + i] for i in range(markov)]
            x_condition = B.concat(*splits, axis=1)
            if entropy_conditional:
                g1, g2 = stein_conditional(
                    stop_gradient(x[markov:]),
                    stop_gradient(x_condition),
                    h=h,
                    eta=eta,
                    inducing_inds=inducing_inds,
                )
                entropy_proxy = -B.sum(g1 * x[markov:]) - B.sum(g2 * x_condition)
            else:
                g = stein(stop_gradient(x), h=h, eta=eta, inducing_inds=inducing_inds)
                entropy_proxy = -B.sum(g * x)

            # Assemble KL.
            return (-entropy_proxy - model_loglik(vs, x)) / m / B.shape(y)[1]
        else:
            # Not using the entropy term.
            return -model_loglik(vs, x) / m / B.shape(y)[1]

    # See if we just require the KL estimator.
    if kl_estimator:
        return kl

    # Vectorise parameters of objective.
    kl(vs, None)  # Initialise variables.
    x = vs.get_vector()  # Initialise vector packer.
    vs_copy = vs.copy()  # Copy variable container for differentiable assignment.

    def f_vectorised(x, inducing_inds):
        vs_copy.set_vector(x)
        return kl(vs_copy, inducing_inds)

    f_value_and_grad = jax.jit(jax.value_and_grad(f_vectorised))

    def f_value_and_grad_subsampled(x):
        if batch_size:
            inducing_inds = np.random.permutation(B.shape(y)[0])[:batch_size]
            inducing_inds = np.sort(inducing_inds)
        else:
            inducing_inds = None
        return f_value_and_grad(x, inducing_inds)

    # Perform optimisation.
    adam = ADAM(rate)
    with out.Progress("Fitting PSA", total=iters) as progress:
        for i in range(iters):
            obj_value, grad = B.to_numpy(f_value_and_grad_subsampled(x))
            progress({"Objective value": obj_value})
            x = adam.step(x, grad)
    vs.set_vector(x)

    # The result of PSA is the learned basis.
    return vs["basis"]
