import numpy as np

from spotter import Star, uniform


def test_show_empty_star():
    star = Star()
    img = star.show()


def test_flux():
    star = Star()
    np.random.rand(42)
    n = 5
    radii = np.random.uniform(0.01, 0.3, n)
    star.add_spot(*uniform(n), radii, 0.1)
    phase = np.linspace(0, 2 * np.pi, 300)
    jaxed = star.jax_flux(phase)(star.map_spot)
    simple = star.flux(phase)
    np.testing.assert_allclose(simple, jaxed)
