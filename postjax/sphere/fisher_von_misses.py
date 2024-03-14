import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
import scipy.stats as ss


class FishervonMises:
    """
    fisher_von_misses
    https://github.com/microscopic-image-analysis/geosss/blob/main/geosss/distributions.py
    """

    def __init__(self, D=2, mu=jnp.ones(2), kappa=1.0):
        self.D = D
        self.name = "FishervonMises"
        self.mu = mu
        self.kappa = kappa

    def logp(self, x):
        return self.kappa * jnp.dot(self.mu, x)

    def dlogp(self, x):
        return self.kappa * self.mu

    def generate_samples_numpy(self, N=10000, rng=np.random.default_rng()):
        D = self.D
        mu = self.mu
        kappa = self.kappa
        samples = ss.vonmises_fisher(mu, kappa).rvs(N, random_state=rng)
        return samples
