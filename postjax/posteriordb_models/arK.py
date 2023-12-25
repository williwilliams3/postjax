import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class arK:
    """
    data {
      int<lower=0> K;
      int<lower=0> T;
      real y[T];
    }

    parameters {
      real alpha;
      real beta[K];
      real<lower=0> sigma;
    }

    model {
      alpha ~ normal(0, 10);
      beta ~ normal(0, 10);
      sigma ~ cauchy(0, 2.5);

      for (t in (K+1):T) {
        real mu;
        mu = alpha;

        for (k in 1:K)
          mu = mu + beta[k] * y[t - k];

        y[t] ~ normal(mu, sigma);
      }
    }
    """

    def __init__(self, pdb_path="posteriordb/posterior_database"):
        self.D = 7
        self.name = "arK-arK"
        self.alpha = 1.0
        posterior = get_posterior(self.name, pdb_path)
        self.data = posterior.data.values()  # the data must be supplied

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
