import stan
import numpy as np
from posteriordb import PosteriorDatabase


"""
This script converts the reference samples from posteriordb to the unconstrained space.
It relies on stan module to build the model and unconstrain the parameters.
"""


def constrained_to_unconstrained_params(posterior, samples: np.ndarray) -> np.ndarray:
    # Build the Stan model once
    model_stan = stan.build(posterior.model.stan_code(), data=posterior.data.values())

    # Precompute cumulative dimension indices
    dims = [1 if not dim else dim[0] for dim in model_stan.dims]
    dim_list_cs = np.cumsum([0] + dims)

    def make_unconstrained_param(row: np.ndarray) -> np.ndarray:
        constrained_param_dict = {
            param_name: (
                row[dim_start:dim_end][0]
                if dim_end - dim_start == 1
                else row[dim_start:dim_end].tolist()
            )
            for param_name, dim_start, dim_end in zip(
                model_stan.param_names, dim_list_cs[:-1], dim_list_cs[1:]
            )
        }
        return model_stan.unconstrain_pars(constrained_param_dict)

    # Apply make_unconstrained_param to each row
    output = [make_unconstrained_param(row) for row in samples]
    return np.array(output)


name = "eight_schools-eight_schools_centered"
# Specify path to posterior_dat abase
# https://github.com/stan-dev/posteriordb-python
my_pdb = PosteriorDatabase("posteriordb/posterior_database")
posterior = my_pdb.posterior(name)
draws = posterior.reference_draws()
# # Convert to numpy array
samples = np.array([list(d.values()) for d in draws]).transpose(0, 2, 1)
# # Reshape (1000, 10 , num_params ) to (10000, num_params)
samples = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2])
# Convert to unconstrained space
unconstrained_samples = constrained_to_unconstrained_params(posterior, samples)
