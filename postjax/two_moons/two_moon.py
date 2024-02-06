import jax.numpy as jnp
from jax.scipy.stats import norm


class twomoons:
    """
    Bimodal two-dimensional distribution
    https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/distributions/target.py
    """

    def __init__(self):
        self.D = 2
        self.max_log_prob = 0.0
        self.name = "TwoMoons"
        self.xlim = [-3, 3]
        self.ylim = [-3, 3]

    def logp(self, x):
        """
        ```
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = jnp.abs(x[0])
        log_prob = (
            -0.5 * ((jnp.linalg.norm(x) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + jnp.log(1 + jnp.exp(-4 * a / 0.09))
        )
        return log_prob
