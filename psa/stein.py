import lab as B
import logging

__all__ = ["stein_grad"]

log = logging.getLogger(__name__)


def k(x, y, h):
    return B.exp(-0.5 * B.pw_dists2(x, y) / h ** 2)


def dk(x, y, h, i):
    return -(x[:, i : i + 1] - y[:, i : i + 1].T) / h ** 2 * k(x, y, h)


def stein_grad(x, h, reg=1e-2, i=0):
    return B.cholsolve(
        B.cholesky(B.reg(k(x, x, h), diag=reg)), B.sum(dk(x, x, h, i), axis=1)[:, None]
    )
