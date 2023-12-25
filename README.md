# postjax
Collection of posterior distributions in Jax.


Install requirements by running in terminal `pip install -r requirements.txt` .
The installation can be done terminal command  `pip install -e .` from the directory containing this README.

**Posteriordb models** requiere the directory path for loading the data, instructions can be foudn in [posteriordb](https://github.com/stan-dev/posteriordb-python/tree/main).


Posteriordb models included:

- [x] "arK-arK"
- [x] "arma-arma11"
- [x] "dogs-dogs"
- [x] "dogs-dogs_log"
- [x] "earnings-logearn_interaction"
- [x]  "eight_schools-eight_schools_centered"
- [x] "eight_schools-eight_schools_noncentered"
- [x]  "garch-garch11"
- [x] "gp_pois_regr-gp_regr"
- [x] "low_dim_gauss_mix-low_dim_gauss_mix"
- [x] "nes2000-nes"
- [x] "sblrc-blr"

### Example

Let us see one example in action, which relies on blackjax for sampling.

```python
import jax
import jax.numpy as jnp
import blackjax
from postjax.posteriordb_models import low_dim_gauss_mix()

# Load the model
M = low_dim_gauss_mix()

# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.ones(M.D)
nuts = blackjax.nuts(M.logp, step_size, inverse_mass_matrix)

# Initialize the state
initial_position = jnp.ones(M.D)
initial_state = nuts.init(initial_position)

# Iterate
num_samples = 100
rng_key = jax.random.PRNGKey(0)


@jax.jit
def one_step(state, rng_key):
    state, _ = nuts.step(rng_key, state)
    return state, state


# Inference loop
keys = jax.random.split(rng_key, num_samples)
_, states = jax.lax.scan(one_step, initial_state, keys)
```
