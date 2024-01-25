import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import scipy
import numpy as np


class simple_rosenbrock:
    # https://onlinelibrary.wiley.com/doi/full/10.1111/sjos.12532
    def __init__(self, D=2, a=1.0, b=100.0):
        self.a = a
        self.b = b
        self.D = D
        self.name = "Rosenbrock"
        self.xlim = [-2, 3]
        self.ylim = [-1, 10]

    def logp(self, theta):
        D = int(self.D)
        b = self.b
        a = self.a
        theta = jnp.array(theta)
        # First term
        logpdf = -((theta[0] - a) ** 2)
        # Terms dependent on previous term
        logpdf -= b * jnp.sum((theta[1:] - theta[:-1] ** 2) ** 2)
        return logpdf

    def inverse_jacobian(self, theta):
        D = self.D
        b = self.b
        first_entry = jnp.sqrt(2)
        rest_entries = jnp.sqrt(2 * b) * jnp.ones(D - 1)
        diag_term = jnp.concatenate([jnp.array([first_entry]), rest_entries])
        low_diag_term = -2 * jnp.sqrt(2 * b) * theta[:-1]
        result_matrix = jnp.diag(diag_term) + jnp.diag(low_diag_term, k=-1)
        return result_matrix

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)

    def densities(self):
        a = self.a
        b = self.b
        density1 = lambda x_lambda: scipy.stats.norm.pdf(x_lambda, a, np.sqrt(0.5))
        x = np.random.normal(loc=a, scale=np.sqrt(0.5), size=10000)
        a_temp = [np.random.normal(loc=xi**2, scale=np.sqrt(0.5 / b)) for xi in x]
        kde = scipy.stats.gaussian_kde(a_temp)
        density2 = lambda y: kde(y)
        return density1, density2

    def generate_samples_numpy(self, N=1000):
        X = [scipy.stats.norm.rvs(self.a, 1 / np.sqrt(2), (N, 1))]
        for j in range(1, self.D):
            index_dependency = -1
            X.append(
                scipy.stats.norm.rvs(
                    X[index_dependency] ** 2, 1 / np.sqrt(2 * self.b), (N, 1)
                )
            )
        return np.concatenate(X, axis=1)

    def generate_samples(self, rng_key, N=1000):
        D = self.D
        a = self.a
        b = self.b
        Z = jax.random.normal(rng_key, shape=(N, D), dtype=jnp.float32)
        samples = []
        theta_1 = 1 / jnp.sqrt(2) * Z[:, 0] + a
        samples.append(theta_1[:, None])
        for i in range(1, D):
            theta_i = 1 / jnp.sqrt(2 * b) * Z[:, i] + samples[-1].flatten() ** 2
            samples.append(theta_i[:, None])
        return jnp.concatenate(samples, axis=1)
