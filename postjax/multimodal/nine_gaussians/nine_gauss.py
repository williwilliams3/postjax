import jax
import jax.numpy as jnp
import jax.scipy.stats as jss


class nine_gaussians:
    """
    Bimodal two-dimensional distribution
    https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/distributions/target.py
    """

    def __init__(self):
        self.D = 2
        self.name = "NineGaussians"
        self.xlim = [-4, 4]
        self.ylim = [-4, 4]
        self.sigma = 0.1

    def logp(self, x):
        means = jnp.array([(i, j) for i in range(-3, 4, 3) for j in range(-3, 4, 3)])
        sigma = self.sigma
        weights = jnp.ones(len(means)) / len(means)
        gaussian_densities = jnp.array(
            [jss.norm.logpdf(x, loc=mu, scale=sigma).sum() for mu in means]
        )
        return jax.scipy.special.logsumexp(jnp.log(weights) + gaussian_densities)

    def generate_samples(self, rng_key, N=10000):
        D = self.D
        means = jnp.array([(i, j) for i in range(-3, 4, 3) for j in range(-3, 4, 3)])
        sigma = self.sigma

        # Use jax random functions instead of numpy random functions
        Z = jax.random.normal(rng_key, shape=(N, D), dtype=jnp.float32)
        index = jax.random.randint(rng_key, shape=(N,), minval=0, maxval=9)
        X = means[index] + sigma * Z
        return X
