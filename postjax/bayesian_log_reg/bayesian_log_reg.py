
import numpy as np
import jax.numpy as jnp
from jax.nn import softplus

########################################################
# Model from RMHMC paper
class baylogreg:
    def __init__(self):
        self.name = "BayLogReg"
        self.alpha = 1.0
        self.data = self.load_data()
        self.D = self.data["D"]

    def load_data(self):
        data = np.loadtxt("postjax/data/australian.dat")
        A = data[:, :-1]
        A = (A - jnp.mean(A, axis=0)) / jnp.std(A, axis=0)
        A = jnp.hstack([jnp.ones((len(A), 1)), A])
        y = data[:, -1]
        N, D = A.shape
        return dict(N=N, D=D, X=A, y=y)

    def logp(self, x):
        # No prior

        data = self.data
        N = data["N"]
        X = data["X"]
        y = data["y"]

        assert len(x) == self.D
        mu = jnp.dot(X, x)
        return jnp.dot(y, mu) - jnp.sum(softplus(mu))