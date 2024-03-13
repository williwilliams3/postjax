import os
import numpy as np
import jax
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
        self.xlim = [-0.3, 0.3]
        self.ylim = [-0.5, 0.5]

    def load_data(self, file_name, normalize):
        data = np.load(os.path.join(current_directory, file_name))
        X = data[:, :-1]
        if normalize:
            X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
        new_x = np.ones((X.shape[0], 1))
        X = jnp.asarray(np.concatenate([X, new_x], axis=1))
        y = jnp.asarray(data[:, -1], dtype=jnp.int32)
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

    def empirical_fisher_metric_fn(self, theta):
        data = self.data
        X = data["X"]
        y = data["y"]
        n = len(y)
        dim = self.D
        alpha = self.alpha_var

        metric = jnp.eye(dim) / alpha

        def func(index):
            c_grad = jax.grad(
                lambda c_theta: jss.bernoulli.logpmf(
                    y[index], jax.nn.sigmoid(jnp.dot(X[index, :], c_theta))
                )
            )(theta)
            return jnp.outer(c_grad, c_grad), c_grad

        outers, grads = jax.vmap(func)(y)
        grad = jnp.sum(grads, axis=0)
        metric = metric + jnp.sum(outers, axis=0) - 1 / n * jnp.outer(grad, grad)
        return 0.5 * (metric + metric.T)
