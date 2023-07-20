import numpy as np
from scipy.stats import bernoulli
nax = np.newaxis

def G_model(T, N):
    """Returns a Gaussian random matrix.

    Args:
        N: number of variables.
        T: number of observations.

    Returns:
        X: TxN matrix with iid zero-mean, unit Gaussian random numbers as entries.
    """

    X = np.random.normal(0, 1., (T,N))

    return X

def GG_model(N, T, m):
    """Returns a random matrix with Gaussian latent feature mixing.

    Args:
        N: number of variables.
        T: number of observations.
        m: number of latent features.

    Returns:
        X: TxN matrix given by the matrix product of two iid, zero-mean, unity variance Gaussian matrices.
    """

    U = np.random.normal(0., 1., (m, N))
    V = np.random.normal(0., 1., (m, T))

    X = np.dot(np.transpose(V), U)

    return X

def dirichlet_draw(K, beta):
    """Returns a Dirichlet random vector.

    Args:
        K: number of bins.
        beta: Dirichlet distribution hyperparameter.

    Returns:
        X: length K vector drawn from Dirichlet distribution with hyperparameter beta.
    """

    q = np.zeros(K)
    for i in range(K):
        a = beta
        if i == K-1:
            b = (K-(i+1)) * beta + 1E-20
        else:
            b = (K - (i + 1)) * beta
        r = np.random.beta(a,b)
        q[i] = r * (1. - np.sum(q[0:i]))

    return q

def Dirichlet_matrix(K, beta, N):
    """Returns a matrix with Dirichlet random vectors as rows.

    Args:
        K: number of bins.
        beta: Dirichlet distribution hyperparameter.
        N: number of rows.

    Returns:
        X: NxK matrix with rows drawn from Dirichlet distribution with hyperparameter beta.
    """

    U = np.zeros((N,K))
    for n in range(N):
        r = dirichlet_draw(K, beta)
        U[n, :] = r

    return U

def SDV_model(beta, N, T, m, p, S_type, V_type):
    """Returns a random matrix with Dirichlet latent feature mixing.

    Args:
        N: number of variables.
        T: number of observations.
        m: number of latent features.
        beta: Dirichlet distribution hyperparameter.
        p : if S_type is 'Bernoulli', p controls the evnet probability.
        S_type : sets the type of modulation matrix.
        V_type : set the type of latent feature matrix.

    Returns:
        X: TxN matrix given by Dirichlet mixture of latent features.
    """

    U = Dirichlet_matrix(m, beta, T)

    if S_type == 'ones':
        S = np.ones((T, m))
    elif S_type == 'Bernoulli':  # +1 with probability p, -1 with probability 1-p
        S = 2 * bernoulli.rvs(p, size=(T, m)) - np.ones((T, m))
    elif S_type == 'exp':
        S = np.random.exponential(1, (T, m))

    if V_type == 'Gaussian':
        V = np.random.normal(0., 1., (m, N))
    elif V_type == 'binary':
        V = 2 * bernoulli.rvs(0.5, size=(m, N)) - np.ones((m, N)) # +1 with probability 0.5, -1 with probability 0.5
    elif V_type == 'Dirichlet':
        V = Dirichlet_matrix(N, 0.1, m)
    elif V_type == 'exp_corr':
        pass

    X = np.dot(np.multiply(S, U), V)

    return X