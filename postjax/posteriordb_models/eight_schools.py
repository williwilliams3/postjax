import jax.numpy as jnp
import jax.scipy.stats as jss

class eight_schools_centered:
    """
    data {
        int <lower=0> J; // number of schools
        real y[J]; // estimated treatment
        real<lower=0> sigma[J]; // std of estimated effect
    }
    parameters {
        real theta[J]; // treatment effect in school j
        real mu; // hyper-parameter of mean
        real<lower=0> tau; // hyper-parameter of sdv
    }
    model {
        tau ~ cauchy(0, 5); // a non-informative prior
        theta ~ normal(mu, tau);
        y ~ normal(theta, sigma);
        mu ~ normal(0, 5);
    }
    """

    def __init__(self):
        self.D = 10
        self.name = "eight_schools-eight_schools_centered"
        self.alpha = 1.0
        self.data = {
            "J": 8,
            "y": [28, 8, -3, 7, -1, 1, 18, 12],
            "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
        }

    def logp(self, x):
        # x[0:8] are theta, x[8] mu, x[9] tau,
        # tau>0 in stan is halfcauchy distribution
        if len(x) != 10:
            raise ("Input must be 10 dim vector")
        data = self.data
        return (
            jnp.sum(
                jss.norm.logpdf(
                    np.array(data["y"]), loc=x[0:8], scale=jnp.array(data["sigma"])
                )
            )
            + jnp.sum(jss.norm.logpdf(x[0:8], loc=x[8], scale=jnp.exp(x[9])))
            + jss.norm.logpdf(x[8], 0, scale=5)
            + jss.cauchy.logpdf(jnp.exp(x[9]), 0, scale=5)
            + x[9]
        )

class eight_schools_noncentered:
    """
    data {
      int <lower=0> J; // number of schools
      real y[J]; // estimated treatment
      real<lower=0> sigma[J]; // std of estimated effect
    }
    parameters {
      vector[J] theta_trans; // transformation of theta
      real mu; // hyper-parameter of mean
      real<lower=0> tau; // hyper-parameter of sd
    }
    transformed parameters{
      vector[J] theta;
      // original theta
      theta=theta_trans*tau+mu;
    }
    model {
      theta_trans ~ normal (0,1);
      y ~ normal(theta , sigma);
      mu ~ normal(0, 5); // a non-informative prior
      tau ~ cauchy(0, 5);
    }
        }
    """

    def __init__(self):
        self.D = 10
        self.name = "eight_schools-eight_schools_noncentered"
        self.alpha = 1.0
        self.data = {
            "J": 8,
            "y": [28, 8, -3, 7, -1, 1, 18, 12],
            "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
        }

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