import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class arma11:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 4
        self.name = "arma-arma11"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        if len(x) != 4:
            raise ("Input must be 4 dim vector")
        data = self.data
        y = jnp.array(data["y"])

        T = data["T"]
        mu = x[0]
        phi = x[1]
        theta = x[2]
        sigma = x[3]

        # likelihood
        nu = mu + phi * mu
        err = y[0] - nu
        log_target = 0.0
        log_target += jss.norm.logpdf(err, 0, scale=jnp.exp(sigma))
        for t in range(1, T):
            nu = mu + phi * y[t - 1] + theta * err
            err = y[t] - nu
            log_target += jss.norm.logpdf(err, 0, scale=jnp.exp(sigma))
        # prior
        log_target += (
            jss.norm.logpdf(mu, 0, scale=10)
            + jss.norm.logpdf(phi, 0, scale=2)
            + jss.norm.logpdf(theta, 0, scale=2)
            + jss.cauchy.logpdf(jnp.exp(sigma), 0, scale=2.5)
        )
        # Change of variable
        return log_target + sigma
