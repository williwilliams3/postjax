import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class nes:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 10
        self.name = "nes2000-nes"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        # data values
        data = self.data
        N = data["N"]
        age_discrete = jnp.array(data["age_discrete"])
        educ1 = jnp.array(data["educ1"])
        gender = jnp.array(data["gender"])
        income = jnp.array(data["income"])
        partyid7 = jnp.array(data["partyid7"])
        race_adj = jnp.array(data["race_adj"])
        real_ideo = jnp.array(data["real_ideo"])

        if len(x) != 10:
            raise (f"Input must be {10} dim vector")
        # parameters
        beta = x[0:9]
        sigma = jnp.exp(x[-1])

        # transformed data
        age30_44 = age_discrete == 2
        age45_64 = age_discrete == 3
        age65up = age_discrete == 4

        log_target = 0.0
        # likelihood
        mu = (
            beta[0]
            + beta[1] * real_ideo
            + beta[2] * race_adj
            + beta[3] * age30_44
            + beta[4] * age45_64
            + beta[5] * age65up
            + beta[6] * educ1
            + beta[7] * gender
            + beta[8] * income
        )
        log_target += jss.norm.logpdf(partyid7, loc=mu, scale=sigma).sum()
        # priors

        # change of variables
        log_target += x[-1]
        return log_target
