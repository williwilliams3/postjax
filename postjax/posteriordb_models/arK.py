import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior
import numpy as np


class arK_backup:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 7
        self.name = "arK-arK"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        # data values
        data = self.data
        y = jnp.array(data["y"])
        K = data["K"]
        T = data["T"]
        if len(x) != K + 2:
            raise (f"Input must be {K+2} dim vector")

        # Reparametrizations
        alpha = x[0]
        beta = x[1 : (K + 1)]
        sigma = jnp.exp(x[-1])

        log_target = 0.0
        for t in range(K, T):
            mu = alpha + jnp.dot(beta, jnp.flip(y[(t - K) : (t)]))
            # likelihood
            log_target += jss.norm.logpdf(y[t], loc=mu, scale=sigma)
        # prior distributions
        log_target += jss.norm.logpdf(alpha, loc=0.0, scale=10.0)
        log_target += jss.norm.logpdf(beta, loc=0.0, scale=10.0).sum()
        log_target += jss.cauchy.logpdf(sigma, loc=0.0, scale=2.5)
        # change of variables
        log_target += x[-1]
        return log_target


class arK:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 7
        self.name = "arK-arK"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        # data values
        data = self.data
        y = np.array(data["y"])
        K = data["K"]
        T = data["T"]
        if len(x) != K + 2:
            raise (f"Input must be {K+2} dim vector")
        matrix_lags = create_matrix_lags(y, K)
        matrix_lags = matrix_lags[:-1]
        # Reparametrizations
        alpha = x[0]
        beta = x[1 : (K + 1)]
        sigma = jnp.exp(x[-1])

        log_target = 0.0
        # Likelihood
        mu = jnp.dot(matrix_lags, jnp.flip(beta))
        log_target = jss.norm.logpdf(y[K:], loc=mu, scale=sigma).sum()

        # prior distributions
        log_target += jss.norm.logpdf(alpha, loc=0.0, scale=10.0)
        log_target += jss.norm.logpdf(beta, loc=0.0, scale=10.0).sum()
        log_target += jss.cauchy.logpdf(sigma, loc=0.0, scale=2.5)
        # change of variables
        log_target += x[-1]
        return log_target


def create_matrix_lags(arr, k):
    T = arr.size
    return np.lib.stride_tricks.as_strided(
        arr, shape=(T - k + 1, k), strides=(arr.strides[0], arr.strides[0])
    )
