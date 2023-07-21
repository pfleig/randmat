import numpy as np
from scipy.stats import beta as beta_distr
nax = np.newaxis


def Gaussian(sig):
    """Computes the curve of a centered Gaussian with standard deviation sig.

    Args:
        sig: standard deviation of the Gaussian.

    Returns:
        x: discrete grid on which Gaussian is computed.
        gauss: function values of the Gaussian curve.
    """

    x = np.linspace(-3*sig, 3*sig, 1000)
    gauss = lambda x: np.exp(-x**2/(2*sig**2)) / np.sqrt(2.*np.pi*sig**2)

    return x, gauss


def Beta_distribution_GGmodel(m):
    """Returns the (symmetric) Beta distribution of pairwise correlations of the GG model.

    Args:
        m: latent dimension of the GG model.

    Returns:
        x_corr: discrete grid on which the distribution is computed.
        Beta: function values of the Beta distribution.
    """

    loc = -1.
    scale = 2.
    alpha = (m-1.)/2.
    beta = alpha

    x_corr = np.linspace(-.999, .999, 1000)

    Beta = beta_distr.pdf(x_corr, alpha, beta, loc, scale)

    return x_corr, Beta


def MP(q, sig=1.):
    """Computes the curve of the Marcenko-Pastur (MP) distribution.

    Args:
        q: ratio of variables to observations.
        sig: standard deviation of random variables in the probabilistic matrix.

    Returns:
        lam: discrete grid over which the MP distribution is computed.
        rho: function values of the MP distribution.
        lambda_min: lower MP bound.
        lambda_max: upper MP bound.
    """

    sig2 = sig**2

    lambda_min = sig2 * np.square(1 - np.sqrt(q))
    lambda_max = sig2 * np.square(1 + np.sqrt(q))

    lam = np.linspace(lambda_min, lambda_max, 250, endpoint=True)
    lambda_min = lambda_min*np.ones(lam.size)
    lambda_max = lambda_max*np.ones(lam.size)

    if q < 1.:
        rho = np.sqrt(np.multiply((lambda_max-lam),(lam-lambda_min)))/(2*np.pi*lam*q*sig2)
    elif q > 1.:
        rho = np.sqrt(np.multiply((lambda_max - lam), (lam - lambda_min))) / (2 * np.pi * lam)
        lam = np.insert(lam, 0, 0., axis=0)

    return lam, rho, lambda_min, lambda_max