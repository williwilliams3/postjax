import os
import numpy as np
import jax.numpy as jnp
from jax.nn import sigmoid
import jax.scipy.stats as jss


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


########################################################
# Model from RMHMC paper
class baylogreg:
    def __init__(self, dataset_name="australian", normalize=True):
        self.name = "LogReg"
        self.normalize = normalize
        self.dataset_name = dataset_name
        self.data = self.load_data(f"data/{dataset_name}.npy", normalize)
        self.D = self.data["D"]
        self.alpha_var = 100.0  # prior variance

    def load_data(self, file_name, normalize):
        data = np.load(os.path.join(current_directory, file_name))
        X = data[:, :-1]
        if normalize:
            X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
        new_x = np.ones((X.shape[0], 1))
        X = jnp.asarray(np.concatenate([X, new_x], axis=1))
        y = data[:, -1]
        assert jnp.all((y == 0) | (y == 1))
        N, D = X.shape
        return dict(N=N, D=D, X=X, y=y)

    def logp(self, theta):
        # No prior
        sqrt_alpha = jnp.sqrt(self.alpha_var)
        data = self.data
        X = data["X"]
        y = data["y"]

        assert len(theta) == self.D
        return jnp.sum(jss.norm.logpdf(theta, 0.0, sqrt_alpha)) + jnp.sum(
            jss.bernoulli.logpmf(y, sigmoid(jnp.dot(X, theta)))
        )

    def fisher_metric_fn(self, theta):
        data = self.data
        X = data["X"]
        dim = self.D
        alpha = self.alpha_var
        preds = sigmoid(jnp.dot(X, theta))
        # Inspired by https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.svd.html
        metric = (X.T * preds * (1.0 - preds)) @ X + jnp.eye(dim) / alpha
        return 0.5 * (metric + metric.T)
