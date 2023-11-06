import jax.numpy as jnp
import jax.scipy.stats as jss
import scipy
import numpy as np


class rosenbrock:
    # Adapted from https://github.com/scipy/scipy/blob/v0.14.0/scipy/optimize/optimize.py#L153
    def __init__(self, a=1.0, b=100.0, D=2):
        self.a = a
        self.b = b
        self.D = D
        self.name = "Rosenbrock"
        self.alpha = 1.0
        self.xlim = [-2, 3]
        self.ylim = [-1, 10]

    def logp(self, x):
        return -jnp.sum(
            self.b * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (self.a - x[:-1]) ** 2.0
        )

    def inverse_jacobian(self, theta):
        raise NotImplementedError()

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

    def generate_samples(self, N=10000):
        a = self.a
        b = self.b
        x = np.random.normal(loc=a, scale=np.sqrt(0.5), size=10000)
        a = [
            np.random.normal(loc=xi**2, scale=np.sqrt(0.5 / b), size=self.D - 1)
            for xi in x
        ]
        if self.D > 2:
            raise Exception(f"No closed form distribution for dim {self.D}>2")
        return np.c_[x, np.array(a)]
