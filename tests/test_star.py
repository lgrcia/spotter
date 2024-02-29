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
    spot_map = star.circular_spots()(*uniform(n), radii)[-1]
    phase = np.linspace(0, 2 * np.pi, 300)
    jaxed = jax.vmap(star.flux, in_axes=(None, None, 0))(spot_map, [0.1, 0.2], phase)


def test_circular_spot_jit():
    star = Star(N=64)
    expected = star.circular_spots(jit=False)(0.1, 0.5, 0.1)
    calc = star.circular_spots(jit=True)(0.1, 0.5, 0.1)
    assert np.allclose(expected, calc)
