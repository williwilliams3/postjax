import jax.numpy as jnp
import jax.scipy.stats as jss
from jax.nn import sigmoid
from .utils import get_posterior


class dogs_dogs:
    """
    data {
      int<lower=0> n_dogs;
      int<lower=0> n_trials;
      int<lower=0,upper=1> y[n_dogs,n_trials];
    }
    parameters {
      vector[3] beta;
    }
    transformed parameters {
      matrix[n_dogs,n_trials] n_avoid;
      matrix[n_dogs,n_trials] n_shock;
      matrix[n_dogs,n_trials] p;

      for (j in 1:n_dogs) {
        n_avoid[j,1] = 0;
        n_shock[j,1] = 0;
        for (t in 2:n_trials) {
          n_avoid[j,t] = n_avoid[j,t-1] + 1 - y[j,t-1];
          n_shock[j,t] = n_shock[j,t-1] + y[j,t-1];
        }
        for (t in 1:n_trials)
          p[j,t] = beta[1] + beta[2] * n_avoid[j,t] + beta[3] * n_shock[j,t];
      }
    }
    model {
      beta ~ normal(0, 100);
      for (i in 1:n_dogs) {
        for (j in 1:n_trials)
          y[i,j] ~ bernoulli_logit(p[i,j]);
      }
    }
    """

    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 3
        self.name = "dogs-dogs"
        self.alpha = 1.0
        posterior = get_posterior(self.name, pdb_path)
        self.data = posterior.data.values()  # the data must be supplied

    def logp(self, x):
        # ['mu.1', 'mu.2', 'sigma.1', 'sigma.2', 'theta']
        if len(x) != 3:
            raise ("Input must be 3 dim vector")
        data = self.data
        y = jnp.array(data["y"])
        n_avoid = jnp.zeros((data["n_dogs"], data["n_trials"]))
        n_shock = jnp.zeros((data["n_dogs"], data["n_trials"]))
        n_avoid[:, 1:] += jnp.cumsum(1 - y[:, :-1], axis=1)
        n_shock[:, 1:] += jnp.cumsum(y[:, :-1], axis=1)

        p = x[0] + x[1] * n_avoid + x[2] * n_shock
        return (
            jss.bernoulli.logpmf(y, p=sigmoid(p)).sum()
            + jss.norm.logpdf(x, loc=0.0, scale=100).sum()
        )


class dogs_dogs_log:
    """
    data {
      int<lower=0> n_trials;
      int<lower=0> n_dogs;
      int<lower=0,upper=1> y[n_dogs,n_trials];
    }
    parameters {
      vector[2] beta;
    }
    transformed parameters {
      matrix[n_dogs,n_trials] n_avoid;
      matrix[n_dogs,n_trials] n_shock;
      matrix[n_dogs,n_trials] p;

      for (j in 1:n_dogs) {
        n_avoid[j,1] = 0;
        n_shock[j,1] = 0;
        for (t in 2:n_trials) {
          n_avoid[j,t] = n_avoid[j,t-1] + 1 - y[j,t-1];
          n_shock[j,t] = n_shock[j,t-1] + y[j,t-1];
        }
        for (t in 1:n_trials)
          p[j,t] = inv_logit(beta[1] * n_avoid[j,t] + beta[2] * n_shock[j,t]);
      }
    }
    model {
      beta[1] ~ uniform(-100, 0);
      beta[2] ~ uniform(0, 100);
      for (i in 1:n_dogs) {
        for (j in 1:n_trials)
          y[i,j] ~ bernoulli(p[i,j]);
      }
    }
    """

    def __init__(self):
        self.D = 2
        self.name = "dogs-dogs_log"
        self.alpha = 1.0
        posterior = get_posterior(self.name)
        self.data = posterior.data.values()  # the data must be supplied

    def logp(self, x):
        # ['mu.1', 'mu.2', 'sigma.1', 'sigma.2', 'theta']
        if len(x) != 2:
            raise ("Input must be 2 dim vector")
        data = self.data
        y = jnp.array(data["y"])
        n_avoid = jnp.zeros((data["n_dogs"], data["n_trials"]))
        n_shock = jnp.zeros((data["n_dogs"], data["n_trials"]))
        n_avoid[:, 1:] += jnp.cumsum(1 - y[:, :-1], axis=1)
        n_shock[:, 1:] += jnp.cumsum(y[:, :-1], axis=1)

        p = sigmoid(x[0] * n_avoid + x[1] * n_shock)
        return (
            # likelihood
            jss.bernoulli.logpmf(y, p=p).sum()
            # prior
            + jss.uniform.logpdf(x[0], loc=-100, scale=100)
            + jss.uniform.logpdf(x[1], loc=0, scale=100)
        )
