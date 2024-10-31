import healpy as hp

from spotter.core import integrated_spectrum


def spectrum(star, wv, spectra, phase):
    phi, theta = hp.pix2ang(star.sides, range(hp.nside2npix(star.sides)))
    return integrated_spectrum(
        star.sides, theta, phi, star.period, star.radius, wv, spectra, phase, star.y
    )
