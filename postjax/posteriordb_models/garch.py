import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class garch11:
    """
    data {
      int<lower=0> T;
      real y[T];
      real<lower=0> sigma1;
    }

    parameters {
      real mu;
      real<lower=0> alpha0;
      real<lower=0, upper=1> alpha1;
      real<lower=0, upper=(1-alpha1)> beta1;
    }

    model {
      real sigma[T];
      sigma[1] = sigma1;
      for (t in 2:T)
        sigma[t] = sqrt(  alpha0
                        + alpha1 * square(y[t - 1] - mu)
                        + beta1 * square(sigma[t - 1]));

      y ~ normal(mu, sigma);
    }
    """

    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 4
        self.name = "garch-garch11"
        self.alpha = 1.0
        posterior = get_posterior(self.name, pdb_path)
        self.data = posterior.data.values()

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
