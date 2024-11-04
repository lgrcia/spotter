import numpy as np
from spotter import Star, core


def test_star_from_sides():
    star = Star.from_sides(16)
    assert star.y.shape == (1, core.npix(16))
    assert star.u is None


def test_star_spectra():
    spectra = np.random.rand(4, core.npix(8))
    star = Star(spectra)
    assert star.y.shape == (4, core.npix(8))
    assert star.u is None


def test_star_with_u():
    star = Star.from_sides(16, u=(0.1, 0.3))
    assert star.u.shape == (1, 2)
