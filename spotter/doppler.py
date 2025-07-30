"""
Doppler and spectral calculations for rotating stars.

This module provides functions to compute disk-integrated spectra, radial velocity
maps, and related quantities for stars with non-uniform surfaces.
"""

from functools import partial

import healpy as hp
import jax.numpy as jnp
from jax.typing import ArrayLike

from spotter import core, light_curves
from spotter.star import Star


def spectrum(star: Star, time: float, normalize: bool = True) -> ArrayLike:
    """
    Compute the integrated spectrum of a rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : float
        Time in days.
    normalize : bool, optional


    Returns
    -------
    spectrum : ndarray
        Integrated spectrum.
    """
    if star.wv is None:
        raise ValueError("Star.wv must be set.")

    if star.wv.shape[0] != star.y.shape[0]:
        raise ValueError(
            "The star spectrum (Star.y) must have the same number of wavelength as the "
            f"wavelengths provided (Star.wv).\nFound Star.wv.shape[0] = {star.wv.shape[0]} "
            f"and Star.y.shape[0] = {star.y.shape[0]}"
        )

    phi, theta = hp.pix2ang(star.sides, range(hp.nside2npix(star.sides)))

    def impl(star, time):
        return core.integrated_spectrum(
            star.sides,
            theta,
            phi,
            star.period,
            star.radius,
            star.wv,
            star.y,
            star.phase(time),
            star.inc,
            normalize=normalize,
        )

    return jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time)


def transit_spectrum(star: Star, time: float, x:float, y:float, z:float, r:float, normalize: bool = True) -> ArrayLike:
    """
    Compute the integrated spectrum of a rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : float
        Time in days.
    normalize : bool, optional


    Returns
    -------
    spectrum : ndarray
        Integrated spectrum.
    """
    if star.wv is None:
        raise ValueError("Star.wv must be set.")

    if star.wv.shape[0] != star.y.shape[0]:
        raise ValueError(
            "The star spectrum (Star.y) must have the same number of wavelength as the "
            f"wavelengths provided (Star.wv).\nFound Star.wv.shape[0] = {star.wv.shape[0]} "
            f"and Star.y.shape[0] = {star.y.shape[0]}"
        )

    phi, theta = hp.pix2ang(star.sides, range(hp.nside2npix(star.sides)))

    def impl(star, time, x, y, z, r):
        
        transit_design_matrix = light_curves.transit_design_matrix(star, x, y, z, r, time=time)
        
        w_shift = core.doppler_shift(theta, phi, star.period, star.radius, star.phase(time))
        dw = star.wv[1] - star.wv[0]
        shift = w_shift[:, None] * star.wv / dw
        spectra = jnp.atleast_2d(star.y)
        spectra_shifted = core.shifted_spectra(spectra.T, shift)

        if normalize:
            integrated_spec = jnp.sum(
            spectra_shifted * transit_design_matrix.T, 0
        ) / jnp.sum(transit_design_matrix.T * spectra.T)
        else:
            integrated_spec = jnp.einsum("ij,ij->j", transit_design_matrix.T, spectra_shifted)
        return integrated_spec
        
    return jnp.vectorize(impl, excluded=(0,), signature="(),(),(),(),()->(n)")(star, time, x, y, z, r)
    


def rv_design_matrix(star: Star, time: float) -> ArrayLike:
    """
    Compute the radial velocity design matrix for a Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : float
        Time in days.

    Returns
    -------
    matrix : ndarray
        Radial velocity design matrix.
    """
    phi, theta = hp.pix2ang(star.sides, range(hp.nside2npix(star.sides)))
    rv = core.radial_velocity(
        theta,
        phi,
        star.period,
        star.radius,
        star.phase(time),
        star.inc,
    )
    return light_curves.design_matrix(star, time) * rv


def radial_velocity(star: Star, time: float) -> ArrayLike:
    """
    Compute the disk-integrated radial velocity of a Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : float
        Time in days.

    Returns
    -------
    rv : ndarray
        Disk-integrated radial velocity.
    """

    def impl(star, time):
        return jnp.einsum(
            "ij,ij->i",
            rv_design_matrix(star, time),
            star.y,
        ) / light_curves.light_curve(star, time, normalize=False)

    return jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time).T
