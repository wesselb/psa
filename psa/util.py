import lab as B
import jax.numpy as jnp
import jax

__all__ = ["cos_sim", "select_bandwidth"]


def cos_sim(x, y):
    """Cosine similarity between two tensors.

    Args:
        x (tensor): First tensor.
        y (tensor): Second tensor.

    Returns:
        scalar: Cosine similarity.
    """
    norm_x = B.sqrt(B.sum(x ** 2))
    norm_y = B.sqrt(B.sum(y ** 2))
    return B.sum(x * y) / norm_x / norm_y

#### Bandwidth estimation
# Adapted from https://www.statsmodels.org/stable/_modules/statsmodels/nonparametric/bandwidths.html

@jax.jit
def _select_sigma(x, percentile=25):
    """
    Returns the smaller of std(X, ddof=1).

    References
    ----------
    Silverman (1986) p.47
    """
    normalize = 1.349
    IQR = (jnp.percentile(x, 75) - jnp.percentile(x, 25)) / normalize
    std_dev = jnp.std(x, axis=0, ddof=1)
    return jnp.minimum(std_dev, IQR)


## Univariate Rule of Thumb Bandwidths ##
@jax.jit
def bw_scott(x, kernel=None):
    """
    Scott's Rule of Thumb

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Unused

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns 1.059 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = jnp.subtract.reduce(jnp.percentile(x, [75,25]))

    References
    ----------

    Scott, D.W. (1992) Multivariate Density Estimation: Theory, Practice, and
        Visualization.
    """
    A = _select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** (-0.2)

@jax.jit
def bw_silverman(x, kernel=None):
    """
    Silverman's Rule of Thumb

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Unused

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns .9 * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = jnp.subtract.reduce(jnp.percentile(x, [75,25]))

    References
    ----------

    Silverman, jnp.W. (1986) `Density Estimation.`
    """
    A = _select_sigma(x)
    n = len(x)
    return .9 * A * n ** (-0.2)


@jax.jit
def bw_normal_reference(x, kernel=None):
    """
    Plug-in bandwidth with kernel specific constant based on normal reference.

    This bandwidth minimizes the mean integrated square error if the true
    distribution is the normal. This choice is an appropriate bandwidth for
    single peaked distributions that are similar to the normal distribution.

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    kernel : CustomKernel object
        Used to calculate the constant for the plug-in bandwidth.
        The default is a Gaussian kernel.

    Returns
    -------
    bw : float
        The estimate of the bandwidth

    Notes
    -----
    Returns C * A * n ** (-1/5.) where ::

       A = min(std(x, ddof=1), IQR/1.349)
       IQR = jnp.subtract.reduce(jnp.percentile(x, [75,25]))
       C = constant from Hansen (2009)

    When using a Gaussian kernel this is equivalent to the 'scott' bandwidth up
    to two decimal places. This is the accuracy to which the 'scott' constant is
    specified.

    References
    ----------

    Silverman, jnp.W. (1986) `Density Estimation.`
    Hansen, jnp.E. (2009) `Lecture Notes on Nonparametrics.`
    """
    C = 1.0592238410488122
    A = _select_sigma(x)
    n = len(x)
    return C * A * n ** (-0.2)

@jax.jit
def select_bandwidth(x, kernel):
    """
    Selects bandwidth for a selection rule bw

    this is a wrapper around existing bandwidth selection rules

    Parameters
    ----------
    x : array_like
        Array for which to get the bandwidth
    bw : str
        name of bandwidth selection rule, currently supported are:
        %s
    kernel : not used yet

    Returns
    -------
    bw : float
        The estimate of the bandwidth
    """
    bandwidth = bw_normal_reference(x, kernel) # hardcoding it here because JAX is annoying
    # there are two other options: bw_silverman and bw_scott, but passing this as an argument
    # was making it break
    return bandwidth
