import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class arma11:
    """
    // ARMA(1, 1)

    data {
    int<lower=1> T; // number of observations
    real y[T];      // observed outputs
    }

    parameters {
    real mu;             // mean coefficient
    real phi;            // autoregression coefficient
    real theta;          // moving average coefficient
    real<lower=0> sigma; // noise scale
    }

    model {
    vector[T] nu;  // prediction for time t
    vector[T] err; // error for time t

    mu ~ normal(0, 10);
    phi ~ normal(0, 2);
    theta ~ normal(0, 2);
    sigma ~ cauchy(0, 2.5);

    nu[1] = mu + phi * mu; // assume err[0] == 0
    err[1] = y[1] - nu[1];
    for (t in 2:T) {
        nu[t] = mu + phi * y[t - 1] + theta * err[t - 1];
        err[t] = y[t] - nu[t];
    }

    err ~ normal(0, sigma);
    }
    """

    def __init__(self, pdb_path="posteriordb/posterior_database"):
        self.D = 4
        self.name = "arma-arma11"
        self.alpha = 1.0
        posterior = get_posterior(self.name, pdb_path)
        self.data = posterior.data.values()  # the data must be supplied

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

    def logp2(self, x):
        if len(x) != 4:
            raise ("Input must be 4 dim vector")
        data = self.data
        y = jnp.array(data["y"])
        T = data["T"]
        mu = x[0]
        phi = x[1]
        theta = x[2]
        sigma = x[3]
        nu = []
        err = []
        # NOT vectorized likelihood

        nu.append(mu + phi * mu)
        err.append(y[0] - nu)

        for t in range(1, T):
            nu.append(mu + phi * y[t - 1] + theta * err[-1])
            err = y[t] - nu
        log_target = jss.norm.logpdf(err, 0, scale=jnp.exp(sigma))
        log_target += jss.norm.logpdf(err, 0, scale=jnp.exp(sigma))
        log_target += (
            jss.norm.logpdf(mu, 0, scale=10)
            + jss.norm.logpdf(phi, 0, scale=2)
            + jss.norm.logpdf(theta, 0, scale=2)
            + jss.cauchy.logpdf(jnp.exp(sigma), 0, scale=2.5)
            + sigma
        )
        return log_target
