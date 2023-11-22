import numpy as np


def butterfly(latitudes, latitude_sigma, n=1):
    theta = np.pi / 2 - (
        np.random.normal(latitudes, latitude_sigma, n) * np.random.choice([-1, 1], n)
    )
    phi = np.random.uniform(0, 2 * np.pi, n)
    return theta, phi
