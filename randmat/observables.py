import numpy as np
from sklearn import preprocessing
from numpy import linalg as la
nax = np.newaxis


def correlation_matrix(X, scale_data):
    """Returns the correlation matrix.

    Args:
        X: data matrix.
        scale_data: boolean variable indictating whether the data is to be scaled.

    Returns:
        corr_mat: correlation matrix of X.
    """

    # compute correlations
    X = preprocessing.scale(X, axis=0, with_mean=True, with_std=scale_data)  # standardise matrix
    n_rows = X.shape[0]

    corr_mat = np.dot(np.transpose(X), X) / n_rows

    return corr_mat


def corr_density(X, scale_data):
    """Returns density of correlation values.

    Args:
        X: data matrix.
        scale_data: boolean variable indicating whether the data is to be scaled.

    Returns:
        density: density of correlation values of X.
    """

    # compute correlations
    X = preprocessing.scale(X, axis=0, with_mean=True, with_std=scale_data)  # standardise matrix
    (n_rows, n_cols) = np.shape(X)
    corr_X = np.dot(np.transpose(X), X) / n_rows  # covariance matrix of the data

    # extract and sort correlations from the correlation matrix
    correlations = corr_X[np.triu_indices(n_cols,1)]
    correlations[::-1].sort()

    # set precision of correlations; important because otherwise it might happen that correlations are
    # found to be >1. due to numerical inaccuracy at the 12 decimal place
    correlations = np.around(correlations, 9)

    # compute the density
    density, bins = np.histogram(correlations, 100, range=(-1., 1.), density='True')
    density = np.flipud(density)

    return density


def sorted_correlations(X, scale_data):
    """Returns the correlation values.

    Args:
        X: data matrix.
        scale_data: boolean variable indictating whether the data is to be scaled.

    Returns:
        correlations: array of sorted entries of the correlation matrix X.
    """

    X = preprocessing.scale(X, axis=0, with_mean=True, with_std=scale_data)  # standardise columns of the data matrix
    (n_rows, n_cols) = np.shape(X)
    corr_X = np.dot(np.transpose(X), X) / n_rows  # covariance matrix of the data
    correlations = corr_X[np.triu_indices(n_cols,1)]
    # correlations = corr_X[np.triu_indices(N,0)]
    correlations[::-1].sort()

    # set precision of correlations; this is important because otherwise it might happen that correlations are
    # found to be >1. due to some numerical inaccuracy at the 12 decimal place
    correlations = np.around(correlations, 9)

    return correlations


def sorted_eigenvalues(X, scale_data):
    """Returns the eigenvalues of the correlation matrix.

    Args:
        X: data matrix.
        scale_data: boolean variable indictating whether the data is to be scaled.

    Returns:
        eigenvalues: array of sorted eigenvalues of the correlation matrix of X.
    """

    X = preprocessing.scale(X, axis=0, with_mean=True, with_std=scale_data)  # preprocess data matrix colums

    n_rows = X.shape[0]
    corr_X = np.dot(np.transpose(X), X) / n_rows  # covariance matrix of the data
    evals, evecs = la.eig(corr_X)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    eigenvalues = np.real(evals)
    eigenvalues[::-1].sort()

    return eigenvalues