import jax.numpy as jnp
import jax.scipy.stats as jss
from jax.nn import softplus, sigmoid
from jax.scipy.special import logsumexp
from .utils import get_posterior


class low_dim_gauss_mix:
    def __init__(self, pdb_path="../posteriordb/posterior_database"):
        self.D = 5
        self.name = "low_dim_gauss_mix-low_dim_gauss_mix"
        self.posterior = get_posterior(self.name, pdb_path)
        self.data = self.posterior.data.values()
        self.theta_ini = jnp.array(
            [-2.68687831, 2.86448131, 1.00374688, 1.02802978, 0.6113613]
        )  # First possition posteriordb
        self.theta_map = jnp.array(
            [
                -2.7343801728812065,
                1.724111539492249,
                0.5888580185638899,
                0.5624755456166634,
                0.48568336025057335,
            ]
        )

    def logp(self, x):
        # ['mu.1', 'mu.2', 'sigma.1', 'sigma.2', 'theta']
        if len(x) != 5:
            raise ("Input must be 5 dim vector")
        data = self.data
        y = jnp.array(data["y"])

        # vectorized likelihood
        return (
            logsumexp(
                jnp.vstack(
                    (
                        -softplus(-x[4])
                        + jss.norm.logpdf(y, x[0], scale=jnp.exp(x[2])),
                        -softplus(x[4])
                        + jss.norm.logpdf(y, x[0] + jnp.exp(x[1]), scale=jnp.exp(x[3])),
                    )
                ),
                axis=0,
            ).sum()
            + jss.norm.logpdf(x[0], 0, scale=2)
            + jss.norm.logpdf(x[0] + jnp.exp(x[1]), 0, scale=2)
            + x[1]
            + jss.norm.logpdf(jnp.exp(x[2]), 0, scale=2)
            + x[2]
            + jss.norm.logpdf(jnp.exp(x[3]), 0, scale=2)
            + x[3]
            + jss.beta.logpdf(sigmoid(x[4]), 5, 5)
            - softplus(-x[4])
            - softplus(x[4])
        )
