import jax.numpy as jnp
import jax.scipy.stats as jss
import scipy
import numpy as np


class hybrid_rosenbrock:
    # https://onlinelibrary.wiley.com/doi/full/10.1111/sjos.12532
    def __init__(self, a=1.0, b=100.0, D=2, n1=5):
        self.a = a
        self.b = b
        self.n1 = n1  # size of blocks
        self.D = D
        self.name = "Rosenbrock"
        self.xlim = [-2, 3]
        self.ylim = [-1, 10]

    def logp(self, theta):
        D = self.D
        n1 = self.n1
        b = self.b
        a = self.a
        # First term
        logpdf = -((theta[0] - a) ** 2)
        # Terms dependent on x[0]
        logpdf -= b * jnp.sum((theta[np.arange(1, D, n1)] - theta[0] ** 2) ** 2)
        # Terms dependent on x[i-1]
        middle_elements = np.arange(1, D)
        # remove terms dependent on x[0]
        middle_elements = middle_elements[
            ~np.isin(middle_elements, np.arange(1, D, n1))
        ]
        logpdf -= b * jnp.sum(
            (theta[middle_elements] - theta[middle_elements - 1] ** 2) ** 2
        )
        return logpdf

    def inverse_jacobian(self, theta):
        D = self.D
        b = self.b
        first_entry = jnp.sqrt(2)
        rest_entries = jnp.sqrt(2 * b) * jnp.ones(D - 1)
        diag_term = jnp.concatenate([jnp.array([first_entry]), rest_entries])
        low_diag_term = -jnp.sqrt(2 * b) * theta[:-1].at[0::5].set(0.0)
        result_matrix = jnp.diag(diag_term) + jnp.diag(low_diag_term, k=-1)
        result_matrix_new = result_matrix.at[1::5, 0].add(-jnp.sqrt(2 * b) * theta[0])
        return result_matrix_new

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)

    def densities(self):
        a = self.a
        xlim = self.xlim
        ylim = self.ylim
        density1 = lambda x_lambda: scipy.stats.norm.pdf(x_lambda, a, np.sqrt(0.5))

        def margial_integrator(y, dim=1):
            if dim == 1:
                I, _ = scipy.integrate.quad(
                    lambda x: np.exp(self.logp(np.array([x, y]))),
                    1.2 * xlim[0],
                    1.2 * xlim[1],
                )
            if dim == 0:
                I, _ = scipy.integrate.quad(
                    lambda x: np.exp(self.logp(np.array([y, x]))),
                    1.2 * ylim[0],
                    1.2 * ylim[1],
                )

        density2 = np.vectorize(margial_integrator)
        return density1, density2

    def generate_samples(self, N=1000):
        n1 = self.n1
        X = [scipy.stats.norm.rvs(self.a, 1 / np.sqrt(2), (N, 1))]
        for j in range(1, self.D):
            if (j - 1) % n1 == 0:
                index_dependency = 0
            else:
                index_dependency = -1
            X.append(
                scipy.stats.norm.rvs(
                    X[index_dependency] ** 2, 1 / np.sqrt(2 * self.b), (N, 1)
                )
            )
        return np.concatenate(X, axis=1)