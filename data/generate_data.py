import numpy as np
import scipy.stats as sps


# banana
np.random.seed(1)
theta_1 = 0.5
theta_2_squared = 0.75
sigma_y = 2.0

ys = sps.norm.rvs(loc=theta_1 + theta_2_squared, scale=sigma_y, size=100)
np.save("bananays.npy", ys)
