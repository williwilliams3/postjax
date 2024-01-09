import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class blr:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 5 + 1
        self.name = "sblrc-blr"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()  # the data must be supplied

    def logp(self, x):
        # data values
        data = self.data
        y = jnp.array(data["y"])
        X = jnp.array(data["X"])
        D = data["D"]
        N = data["N"]

        if len(x) != D + 1:
            raise (f"Input must be {D+1} dim vector")
        # parameters
        beta = x[0:D]
        sigma = jnp.exp(x[-1])

        # transformed data

        log_target = 0.0
        # likelihood
        log_target += jss.norm.logpdf(y, loc=jnp.dot(X, beta), scale=sigma).sum()
        # priors
        log_target += jss.norm.logpdf(beta, 0.0, 10.0).sum()
        log_target += jss.norm.logpdf(sigma, 0.0, 10.0)
        # change of variables
        log_target += x[-1]
        return log_target
