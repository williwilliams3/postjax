import numpy as np
import jax.scipy.stats as jss
from jax.nn import sigmoid
from .utils import get_posterior


class dogs_dogs:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 3
        self.name = "dogs-dogs"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        # ['mu.1', 'mu.2', 'sigma.1', 'sigma.2', 'theta']
        if len(x) != 3:
            raise ("Input must be 3 dim vector")
        data = self.data
        y = np.array(data["y"])
        n_avoid = np.zeros((data["n_dogs"], data["n_trials"]))
        n_shock = np.zeros((data["n_dogs"], data["n_trials"]))
        n_avoid[:, 1:] += np.cumsum(1 - y[:, :-1], axis=1)
        n_shock[:, 1:] += np.cumsum(y[:, :-1], axis=1)

        p = x[0] + x[1] * n_avoid + x[2] * n_shock
        return (
            jss.bernoulli.logpmf(y, p=sigmoid(p)).sum()
            + jss.norm.logpdf(x, loc=0.0, scale=100).sum()
        )


class dogs_log:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 2
        self.name = "dogs-dogs_log"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        # ['mu.1', 'mu.2', 'sigma.1', 'sigma.2', 'theta']
        if len(x) != 2:
            raise ("Input must be 2 dim vector")
        data = self.data
        y = np.array(data["y"])
        n_avoid = np.zeros((data["n_dogs"], data["n_trials"]))
        n_shock = np.zeros((data["n_dogs"], data["n_trials"]))
        n_avoid[:, 1:] += np.cumsum(1 - y[:, :-1], axis=1)
        n_shock[:, 1:] += np.cumsum(y[:, :-1], axis=1)

        p = sigmoid(x[0] * n_avoid + x[1] * n_shock)
        return (
            # likelihood
            jss.bernoulli.logpmf(y, p=p).sum()
            # prior
            + jss.uniform.logpdf(x[0], loc=-100, scale=100)
            + jss.uniform.logpdf(x[1], loc=0, scale=100)
        )
