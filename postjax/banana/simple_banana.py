import os
import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
from cmdstanpy import CmdStanModel

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


class banana:
    # Taken from https://github.com/ksnxr/riemannian_laplace/
    def __init__(
        self,
        D=2,
        sigma_theta=2.0,
        sigma_y=2.0,
    ):
        self.D = D
        self.name = "Banana"
        self.data = jnp.asarray(
            np.load(os.path.join(current_directory, "data/bananays.npy"))
        )
        self.sigma_theta = sigma_theta
        self.sigma_y = sigma_y
        self.xlim = [-5.5, 2.5]
        self.ylim = [-3.0, 3.0]

    def logp(self, theta):
        sigma_theta = self.sigma_theta
        sigma_y = self.sigma_y
        ys = self.data
        return (
            jss.norm.logpdf(theta[0], 0.0, sigma_theta)
            + jss.norm.logpdf(theta[1], 0.0, sigma_theta)
            + jnp.sum(jss.norm.logpdf(ys, theta[0] + jnp.square(theta[1]), sigma_y))
        )

    def fisher_metric_fn(self, theta):
        sigma_theta = self.sigma_theta
        sigma_y = self.sigma_y
        ys = self.data
        n = len(ys)
        quantity_prior = 1.0 / sigma_theta**2
        quantity_00 = n / sigma_y**2
        quantity_01 = 2.0 * n / sigma_y**2
        quantity_11 = 4.0 * n / sigma_y**2
        metric = jnp.array(
            [
                [quantity_prior + quantity_00, quantity_01 * theta[1]],
                [
                    quantity_01 * theta[1],
                    quantity_prior + quantity_11 * jnp.square(theta[1]),
                ],
            ]
        )
        return 0.5 * (metric + metric.T)

    def densities(self):
        raise NotImplementedError("This function is not implemented yet")

    def generate_samples_stan(
        self,
        seed=1,
        N=10_000,
        show_progress=False,
        samples_dir=f"data/ground_truth_samples/Banana/",
    ):
        data = dict(y=self.data.tolist())
        model = CmdStanModel(
            stan_file=os.path.join(current_directory, "stan_models/banana.stan")
        )
        fit = model.sample(
            data=data,
            thin=10,
            chains=10,
            iter_warmup=10_000,
            iter_sampling=N,
            adapt_delta=0.95,
            # fix seed for reproducibility
            seed=seed,
            show_progress=show_progress,
        )
        samples = fit.draws_pd()[["theta[1]", "theta[2]"]].to_numpy()
        if samples_dir is not None:
            np.save(samples_dir + "reference_samples.npy", samples)
        return samples