import jax.numpy as jnp
import jax.scipy.stats as jss
import scipy
import numpy as np


class squiggle:
    def __init__(self, D=2, a=2.0, Sig=jnp.array([[5.0, 0.0], [0.0, 0.05]])):
        self.a = a
        self.D = D
        self.mu = np.zeros(D)
        self.Sig = Sig
        self.name = "Squiggle"
        self.alpha = 1.0
        self.xlim = [-9, 9]
        self.ylim = [-3, 3]
        if D > 2:
            self.Sig = jnp.diag(jnp.array([5] + (D - 1) * [0.05]))
        self.precision = jnp.linalg.inv(self.Sig)

    def logp(self, x):
        g = jnp.insert(x[1:] + jnp.sin(self.a * x[0]), 0, x[0])
        if self.D == 2:
            return jss.multivariate_normal.logpdf(g, self.mu, self.Sig)
        elif self.D > 2:
            # Uncorrelated gaussians
            return jss.norm.logpdf(
                g, self.mu, jnp.array([5] + (self.D - 1) * [0.05])
            ).sum()

    def inverse_jacobian(self, theta):
        D = self.D
        a = self.a
        inverse_jacobian = jnp.array([[1.0, 0.0], [a * jnp.cos(a * theta[0]), 1.0]])
        return inverse_jacobian

    def inverse_jacobian(self, theta):
        D = self.D
        a = self.a
        top_row = jnp.append(1.0, jnp.zeros(D - 1))
        bottom_rows = jnp.c_[
            a * jnp.cos(a * theta[0]) * jnp.ones(D - 1), jnp.eye(D - 1)
        ]
        inverse_jacobian = jnp.r_[[top_row], bottom_rows]
        return inverse_jacobian

    def fisher_metric_fn(self, theta):
        inverse_jacobian = self.inverse_jacobian(theta)
        metric = inverse_jacobian.T @ self.precision @ inverse_jacobian
        return 0.5 * (metric + metric.T)

    def densities(self):
        xlim = self.xlim
        ylim = self.ylim
        # only true for diagonal covariance matrix
        density1 = lambda x_lambda: scipy.stats.norm.pdf(
            x_lambda, self.mu[0], np.sqrt(self.Sig[0, 0])
        )

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

    def generate_samples(self, N=10000):
        a = self.a
        D = self.D
        y = np.random.normal(
            self.mu,
            scale=jnp.array([np.sqrt(5)] + (self.D - 1) * [np.sqrt(0.05)]),
            size=(N, D),
        )
        y[:, 1:] -= np.sin(a * y[:, 0])[:, None]
        return y
