import numpy as np
import jax.scipy.stats as jss
from jax.nn import sigmoid, softplus
import jax.numpy as jnp
from .utils import get_posterior


class dogs_dogs:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 3
        self.name = "dogs-dogs"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()
        self.clipping_tolerance = 1e-5

    def logp(self, x):
        if len(x) != 3:
            raise ("Input must be 3 dim vector")
        data = self.data
        y = np.array(data["y"])
        n_avoid = np.zeros((data["n_dogs"], data["n_trials"]))
        n_shock = np.zeros((data["n_dogs"], data["n_trials"]))
        n_avoid[:, 1:] += np.cumsum(1 - y[:, :-1], axis=1)
        n_shock[:, 1:] += np.cumsum(y[:, :-1], axis=1)

        p = x[0] + x[1] * n_avoid + x[2] * n_shock

        sigma = sigmoid(p)
        sigma_clip = jnp.clip(
            sigma, self.clipping_tolerance, 1.0 - self.clipping_tolerance
        )

        return (
            jss.bernoulli.logpmf(y, sigma_clip).sum()
            + jss.norm.logpdf(x, loc=0.0, scale=100).sum()
        )


class dogs_log:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 2
        self.name = "dogs-dogs_log"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()
        self.clipping_tolerance = 1e-5
        self.xlim = [-3000, -400]
        self.ylim = [-1700, -500]

    def logp(self, x):
        if len(x) != 2:
            raise ("Input must be 2 dim vector")
        data = self.data
        y = np.array(data["y"])
        n_avoid = np.zeros((data["n_dogs"], data["n_trials"]))
        n_shock = np.zeros((data["n_dogs"], data["n_trials"]))
        n_avoid[:, 1:] += np.cumsum(1 - y[:, :-1], axis=1)
        n_shock[:, 1:] += np.cumsum(y[:, :-1], axis=1)
        # Transformation
        y0 = (x[0] - 100) / 100
        y1 = x[1] / 100
        sigma0 = sigmoid(y0)
        sigma1 = sigmoid(y1)
        p = sigmoid(sigma0 * n_avoid + sigma1 * n_shock)
        p_clip = jnp.clip(p, self.clipping_tolerance, 1.0 - self.clipping_tolerance)
        return (
            # likelihood
            jss.bernoulli.logpmf(y, p=p_clip).sum()
            # prior is always zero
            # + jss.uniform.logpdf(sigma1)
            # + jss.uniform.logpdf(sigma2)
            # change of variables
            - softplus(-y0)
            - softplus(y0)
            - softplus(-y1)
            - softplus(y1)
        )
