import jax.numpy as jnp
import numpy as np
from jax import random


def jax_butterfly(key, latitudes=0.0, latitude_sigma=0.0, n=1):
    new_key, subkey = random.split(key)
    theta = jnp.pi / 2 - (
        latitudes
        + random.normal(key, shape=(n,))
        * latitude_sigma
        * random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n,))
    )
    new_key, subkey = random.split(new_key)
    phi = random.uniform(subkey, minval=0.0, maxval=2.0 * jnp.pi, shape=(n,))
    return theta, phi


def jax_uniform(key, n=1):
    # Latitude
    theta = jnp.pi / 2 - jnp.arcsin(
        random.uniform(key, minval=-1.0, maxval=1.0, shape=(n,))
    )
    _, subkey = random.split(key)
    # Longitude
    phi = random.uniform(subkey, minval=0.0, maxval=2.0 * jnp.pi, shape=(n,))

    return theta, phi


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
