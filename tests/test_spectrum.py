from spotter import Star, doppler
import numpy as np
import pytest


def test_star_inputs():
    wv = np.linspace(0, 1, 10)
    star = Star.from_sides(10, wv=wv, period=1.0)
    with pytest.raises(ValueError):
        doppler.spectrum(star, [0])

    star = Star.from_sides(10)
    with pytest.raises(ValueError):
        doppler.spectrum(star, [0])


def test_spectrum():
    wv = np.linspace(0, 1, 10)
    star = Star.from_sides(10, wv=wv, period=1.0)
    spectra = np.random.rand(len(wv), star.size)
    star = star.set(y=spectra)
    assert doppler.spectrum(star, [0]).shape == (1, len(wv))


def test_no_period():
    wv = np.linspace(0, 1, 10)
    star = Star.from_sides(10, wv=wv)
    spectra = np.random.rand(len(wv), star.size)
    star = star.set(y=spectra)
    assert doppler.spectrum(star, [0]).shape == (1, len(wv))
