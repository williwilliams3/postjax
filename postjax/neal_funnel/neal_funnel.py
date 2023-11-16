import jax.numpy as jnp
import jax.scipy.stats as jss
import scipy
import numpy as np


class neal_funnel:
    def __init__(self, D=2, mean=0.0, sigma=3.0):
        self.name = "NealFunnel"
        self.sigma = sigma
        self.mean = mean
        self.D = D
        self.xlim = [-10.0, 10.0]
        self.ylim = [-10.0, 10.0]

    def logp(self, theta):
        theta = jnp.array(theta)
        return jss.norm.logpdf(theta[self.D - 1], loc=0.0, scale=self.sigma) + jnp.sum(
            jss.norm.logpdf(
                theta[: self.D - 1], loc=0.0, scale=jnp.exp(0.5 * theta[self.D - 1])
            )
        )

    def dlogp(self, theta):
        sigma = self.sigma
        mean = self.mean
        D1 = self.D - 1
        theta_D = theta[-1]
        conditional_variance = jnp.exp(theta_D)
        u = jnp.append(
            -(theta[0:D1] - mean) / conditional_variance,
            -D1 / 2.0
            + 0.5 * jnp.linalg.norm(theta[0:D1] - mean) ** 2.0 / conditional_variance
            - theta_D / sigma**2,
        )
        return u

    def inverse_jacobian(self, theta):
        D = self.D
        upper_rows = jnp.c_[
            jnp.exp(-0.5 * theta[-1]) * jnp.eye(D - 1),
            -0.5 * jnp.exp(-0.5 * theta[-1]) * theta[0 : (D - 1)],
        ]
        lowest_row = jnp.append(jnp.zeros(D - 1), 1.0 / self.sigma)
        inverse_jacobian = jnp.r_[upper_rows, [lowest_row]]
        return inverse_jacobian

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian(theta)
        metric = inverse_jacobian.T @ inverse_jacobian
        return 0.5 * (metric + metric.T)

    def densities(self):
        density1 = lambda x: scipy.stats.norm.pdf(x, 0, self.sigma)
        xlim = self.xlim
        ylim = self.ylim

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
            return I

        density2 = np.vectorize(lambda t: margial_integrator(t, dim=0))
        return density2, density1

    def generate_samples(self, N=10000):
        D = self.D
        sigma = self.sigma
        mean = self.mean
        x = np.random.normal(loc=0, scale=sigma, size=N)
        a = [
            np.random.normal(loc=mean, scale=jnp.exp(0.5 * xi), size=D - 1) for xi in x
        ]
        return np.c_[np.array(a), x]
