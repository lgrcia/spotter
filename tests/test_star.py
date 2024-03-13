import jax
import numpy as np

from spotter import Star, uniform


def test_show_empty_star():
    star = Star()


def test_flux():
    star = Star()
    np.random.rand(42)
    n = 5
    radii = np.random.uniform(0.01, 0.3, n)
    spot_map = star.spots(*uniform(n), radii)
    phase = np.linspace(0, 2 * np.pi, 300)
    jaxed = jax.vmap(star.flux, in_axes=(None, None, 0))(spot_map, [0.1, 0.2], phase)
