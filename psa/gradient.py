import jax
import lab.jax as B
from jax.lax import scan
from stheno import EQ

__all__ = ["construct_gradient_estimator"]


def _map_sum(f, *xs):
    acc0 = f(*(x[0] for x in xs))

    def scan_f(acc, arg):
        return acc + f(*arg), None

    res, _ = scan(scan_f, acc0, tuple(x[1:] for x in xs))
    return res


def construct_gradient_estimator(k_x=EQ(), k_y=EQ(), k_ce=EQ()):
    """Construct the gradient estimator.

    Args:
        k_x (:class:`stheno.Kernel`, optional): Kernel for the dependence on the first
            argument. Defaults to the EQ kernel.
        k_x (:class:`stheno.Kernel`, optional): Kernel for the dependence on the second
            argument. Defaults to the EQ kernel.
        k_x (:class:`stheno.Kernel`, optional): Kernel for the conditional expectation.
            Defaults to the EQ kernel.

    Returns:
        function: Gradient estimator.
    """

    def f_x(x, x_ref, h):
        if B.rank(x) < B.rank(x_ref):
            x = B.expand_dims(x, axis=0)
        return B.dense(k_x.stretch(h)(x_ref, x))

    def f_y(y, y_ref, h):
        if B.rank(y) < B.rank(y_ref):
            y = B.expand_dims(y, axis=0)
        return B.dense(k_y.stretch(h)(y_ref, y))

    def f_ce(xi, yi, x, y, h, h_ce):
        w = B.dense(k_ce.stretch(h_ce)(B.expand_dims(yi, axis=0), y))
        f0_ce = B.sum((f_x(xi, x, h) - f_x(x, x, h)) * w, axis=1) / B.sum(w)
        return f0_ce[:, None] * f_y(yi, y, h)

    def f_x_grad_x(xi, x, h):
        def to_diff(xi_):
            return B.flatten(f_x(xi_, x, h))

        return jax.jacfwd(to_diff)(xi)

    def f_ce_dy(xi, yi, x, y, h, h_ce):
        def to_diff(yi_):
            return B.flatten(f_ce(xi, yi_, x, y, h, h_ce))

        return jax.jacfwd(to_diff)(yi)

    def f_dx(x, y, h):
        def to_map(xi, yi):
            return f_x_grad_x(xi, x, h) * f_y(yi, y, h)

        return _map_sum(to_map, x, y)

    def f_dy(x, y, h, h_ce):
        def to_map(xi, yi):
            return f_ce_dy(xi, yi, x, y, h, h_ce)

        return _map_sum(to_map, x, y)

    @jax.jit
    def estimator(x, y, h, h_ce, eta=1e-2):
        """Gradient estimator.

        Args:
            x (matrix): Samples of the argument of the logpdf.
            y (matrix): Samples of the conditional argument of the logpdf.
            h (float): Length scale for the kernels over the arguments.
            h_ce (float):  Length scale for the kernel for the conditional expectation.
            eta (float, optional): L2 regulariser. Defaults to `1e-2`.

        Returns:
            tuple[matrix, matrix]: Estimates of the gradients of the conditional logpdf.

        """
        chol = B.chol(B.reg(f_x(x, x, h) * f_y(y, y, h), diag=eta))
        return (
            -B.cholsolve(chol, f_dx(x, y, h)),
            -B.cholsolve(chol, f_dy(x, y, h, h_ce)),
        )

    return estimator
