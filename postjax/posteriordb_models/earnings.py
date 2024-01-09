import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class logearn_interaction:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 5
        self.name = "earnings-logearn_interaction"
        self.alpha = 1.0
        posterior = get_posterior(self.name, pdb_path)
        self.data = posterior.data.values()  # the data must be supplied

    def logp(self, x):
        # data values
        data = self.data
        earn = jnp.array(data["earn"])
        height = jnp.array(data["height"])
        male = jnp.array(data["male"])
        if len(x) != 5:
            raise (f"Input must be {5} dim vector")
        # transformed data
        log_earn = jnp.log(earn)
        inter = jnp.multiply(height, male)

        # Reparametrizations
        sigma = jnp.exp(x[4])

        log_target = 0.0

        # likelihood
        mu = x[0] + x[1] * height + x[2] * male + x[3] * inter
        log_target += jss.norm.logpdf(log_earn, loc=mu, scale=sigma).sum()

        # change of variables
        log_target += x[4]
        return log_target
