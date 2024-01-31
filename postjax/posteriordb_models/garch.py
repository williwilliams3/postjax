import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior
from jax.nn import sigmoid, softplus


class garch11:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 4
        self.name = "garch-garch11"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        if len(x) != 4:
            raise ("Input must be 4 dim vector")
        # data values
        data = self.data
        y = jnp.array(data["y"])
        sigma20 = data["sigma1"] * 2
        T = data["T"]
        # Reparametrizations
        mu = x[0]
        alpha0 = jnp.exp(x[1])
        alpha1 = sigmoid(x[2])
        beta1 = (1 - alpha1) * sigmoid(x[3])

        # Change for loop for lax.scan
        garch_drift = jnp.concatenate(
            [jnp.array([sigma20]), alpha0 + alpha1 * (y[:-1] - mu) ** 2]
        )
        _, sigma2_vector = geometric_series_accumulation(garch_drift, beta1)

        # Likelihood
        log_target = jss.norm.logpdf(y, mu, scale=jnp.sqrt(sigma2_vector)).sum()

        # change of variables
        log_target += x[1]
        log_target += -softplus(-x[2]) - softplus(x[2])
        log_target += -softplus(-x[3]) - softplus(x[3]) + jnp.log((1 - alpha1))
        return log_target


def geometric_series_accumulation(a_array, b):
    def update_accumulation(prev_accumulation, a_i):
        update_value = prev_accumulation * b + a_i
        return update_value, update_value

    initial_accumulation = jnp.zeros_like(a_array[0], dtype=jnp.float32)
    geometric_series, info = jax.lax.scan(
        update_accumulation, initial_accumulation, a_array
    )
    return geometric_series, info


class garch11_backup:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 4
        self.name = "garch-garch11"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

    def logp(self, x):
        if len(x) != 4:
            raise ("Input must be 4 dim vector")
        # data values
        data = self.data
        y = jnp.array(data["y"])
        sigma1 = data["sigma1"]
        T = data["T"]
        # Reparametrizations
        mu = x[0]
        alpha0 = jnp.exp(x[1])
        alpha1 = sigmoid(x[2])
        beta1 = (1 - alpha1) * sigmoid(x[3])
        sigmat = sigma1
        log_target = 0.0
        log_target += jss.norm.logpdf(y[0], mu, scale=sigmat)
        for t in range(1, T):
            sigmat = jnp.sqrt(
                alpha0 + alpha1 * (y[t - 1] - mu) ** 2 + beta1 * sigmat**2
            )
            # likelihood
            log_target += jss.norm.logpdf(y[t], mu, scale=sigmat)
        # change of variables
        log_target += x[1]
        log_target += -softplus(-x[2]) - softplus(x[2])
        log_target += -softplus(-x[3]) - softplus(x[3]) + jnp.log((1 - alpha1))
        return log_target
