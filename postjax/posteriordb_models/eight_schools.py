import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class eight_schools_centered:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 10
        self.name = "eight_schools-eight_schools_centered"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        # x[0:8] are theta, x[8] mu, x[9] tau,
        # tau>0 in stan is halfcauchy distribution
        if len(x) != 10:
            raise ("Input must be 10 dim vector")
        data = self.data
        return (
            jnp.sum(
                jss.norm.logpdf(
                    jnp.array(data["y"]), loc=x[0:8], scale=jnp.array(data["sigma"])
                )
            )
            + jnp.sum(jss.norm.logpdf(x[0:8], loc=x[8], scale=jnp.exp(x[9])))
            + jss.norm.logpdf(x[8], 0, scale=5)
            + jss.cauchy.logpdf(jnp.exp(x[9]), 0, scale=5)
            + x[9]
        )


class eight_schools_noncentered:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 10
        self.name = "eight_schools-eight_schools_noncentered"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        # x[0:8] are theta, x[8] mu, x[9] tau,
        # tau>0 in stan is halfcauchy distribution
        if len(x) != 10:
            raise ("Input must be 10 dim vector")
        data = self.data
        theta_trans = x[0:8]
        mu = x[8]
        tau = jnp.exp(x[9])
        theta = tau * theta_trans + mu
        return (
            # log-likelihood
            jnp.sum(
                jss.norm.logpdf(
                    jnp.array(data["y"]),
                    loc=theta,
                    scale=jnp.array(data["sigma"]),
                )
            )
            # prior distribution
            + jnp.sum(jss.norm.logpdf(theta_trans, loc=0.0, scale=1.0))
            + jss.norm.logpdf(mu, 0, scale=5)
            + jss.cauchy.logpdf(tau, 0, scale=5)
            # change of variables
            + x[9]
        )
