import numpy as np


def butterfly(latitudes=0, latitude_sigma=0, n=1):
    theta = np.pi / 2 - (
        np.random.normal(latitudes, latitude_sigma, n) * np.random.choice([-1, 1], n)
    )
    phi = np.random.uniform(0, 2 * np.pi, n)
    return theta, phi


def uniform(n=1):
    # Latitude
    theta = np.pi / 2 - np.arcsin(np.random.uniform(-1, 1, n))
    # Longitude
    phi = np.random.uniform(0, 2 * np.pi, n)

    return theta, phi
