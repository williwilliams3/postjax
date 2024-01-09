import jax.numpy as jnp
import jax.scipy.stats as jss
from .utils import get_posterior


class gp_regr:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 3
        self.name = "gp_pois_regr-gp_regr"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()

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
