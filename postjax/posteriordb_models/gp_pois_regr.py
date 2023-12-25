import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


def exponentiated_quadratic(x_, alpha, rho):
    """
    Exponentiated Quadratic (EQ) covariance function.

    Parameters:
        x: Input array of shape (d) d is the dimensionality of the input.
        alpha: Scaling parameter of the covariance function.
        rho: Length-scale parameter of the covariance function.

    Returns:
        Covariance matrix of shape (d, d).
    """
    x_ /= rho
    pairwise_sq_dists = x_[:, None] - x_
    cov_matrix = alpha**2 * jnp.exp(-0.5 * pairwise_sq_dists**2)
    return cov_matrix


class gp_regr:
    """
    data {
      int<lower=1> N;
      real x[N];
      vector[N] y;
    }

    parameters {
      real<lower=0> rho;
      real<lower=0> alpha;
      real<lower=0> sigma;
    }

    model {
      matrix[N, N] cov =   cov_exp_quad(x, alpha, rho)
                         + diag_matrix(rep_vector(sigma, N));
      matrix[N, N] L_cov = cholesky_decompose(cov);

      rho ~ gamma(25, 4);
      alpha ~ normal(0, 2);
      sigma ~ normal(0, 1);

      y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
    }
    """

    def __init__(self, pdb_path="posteriordb/posterior_database"):
        self.D = 3
        self.name = "gp_pois_regr-gp_regr"
        self.alpha = 1.0
        posterior = get_posterior(self.name, pdb_path)
        self.data = posterior.data.values()  # the data must be supplied

    def logp(self, x):
        # data values
        data = self.data
        N = data["N"]
        x_ = jnp.array(data["x"])
        y_ = jnp.array(data["y"])

        if len(x) != 3:
            raise (f"Input must be {3} dim vector")
        # parameters
        rho = jnp.exp(x[0])
        alpha = jnp.exp(x[1])
        sigma = jnp.exp(x[2])

        # transformed data
        cov = exponentiated_quadratic(x_, alpha, rho) + jnp.diag(sigma * jnp.ones(N))

        log_target = 0.0
        # likelihood
        log_target += jss.multivariate_normal.logpdf(
            y_, mean=jnp.zeros(N), cov=cov
        ).sum()
        # priors
        log_target += jss.gamma.logpdf(rho, a=25.0, scale=1 / 4.0)
        log_target += jss.norm.logpdf(alpha, 0.0, 2.0)
        log_target += jss.norm.logpdf(sigma, 0.0, 1.0)

        # change of variables
        log_target += x[0] + x[1] + x[2]
        return log_target
