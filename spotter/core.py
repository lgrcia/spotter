"""
Core utilities for spherical maps, geometry, and flux calculations.

This module provides functions for working with HEALPix maps, including
coordinate transformations, limb darkening, spot generation, and flux
integration for stellar surfaces.
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from spotter import utils


def distance(X, x):
    """
    Compute the great-circle distance between vectors X and x.

    Parameters
    ----------
    X : array_like, shape (..., 3)
        Array of 3D vectors.
    x : array_like, shape (3,)
        Single 3D vector.

    Returns
    -------
    d : array_like
        Great-circle distance(s) in radians.
    """
    return jnp.arctan2(
        jnp.linalg.norm(jnp.cross(X.T, x[:, None], axis=0), axis=0), X @ x
    )


def npix(N):
    """
    Return the number of HEALPix pixels for a given nside.

    Parameters
    ----------
    N : int
        HEALPix nside parameter.

    Returns
    -------
    n : int
        Number of pixels.
    """
    return hp.nside2npix(N)


def equator_coords(phi=None, inc=None, obl=None):
    """
    Compute the coordinates of the stellar equator for given orientation.

    Parameters
    ----------
    phi : float or None, optional
        Rotation phase in radians.
    inc : float or None, optional
        Inclination in radians.
    obl : float or None, optional
        Obliquity in radians.

    Returns
    -------
    coords : ndarray, shape (3,)
        Cartesian coordinates.
    """
    # inc = None -> inc = pi/2
    if inc is None:
        si = 1.0
        ci = 0.0
    else:
        si = jnp.sin(inc)
        ci = jnp.cos(inc)

    if phi is None:
        cp = 1.0
        sp = 0.0
    else:
        cp = jnp.cos(phi)
        sp = jnp.sin(phi)

    if obl is None:
        co = 1.0
        so = 0.0
    else:
        co = jnp.cos(obl)
        so = jnp.sin(obl)

    x = si * cp
    y = si * sp
    z = ci

    x, y, z = (
        x,
        co * y - so * z,
        so * y + co * z,
    )

    return jnp.array([x, y, z])


def mask_projected_limb(N_or_y, phase=None, inc=None, u=None, obl=None):
    """
    Compute mask, projected area, and limb darkening for visible pixels.

    Parameters
    ----------
    X : array_like, shape (..., 3)
        Cartesian coordinates of pixels.
    phase : float, optional
        Rotation phase in radians.
    inc : float, optional
        Inclination in radians.
    u : array_like or None, optional
        Limb darkening coefficients.
    obl : float, optional
        Obliquity in radians.

    Returns
    -------
    mask : ndarray
        Boolean mask for visible pixels.
    projected_area : ndarray
        Projected area for each pixel.
    limb_darkening : ndarray
        Limb darkening factor for each pixel.
    """
    N, _ = _N_or_Y_to_N_n(N_or_y)
    X = vec(N)
    d = distance(X, equator_coords(phase, inc, obl))
    mask = d < jnp.pi / 2
    z = jnp.cos(d)
    projected_area = z * hp.nside2pixarea(N)
    if u is not None:
        u = jnp.atleast_1d(u)
        terms = jnp.array([un * (1 - z) ** (n + 1) for n, un in enumerate(u)])
        limb_darkening = 1 - jnp.sum(terms, axis=z.ndim - 1)
    else:
        limb_darkening = jnp.ones_like(d)

    return mask, projected_area, limb_darkening


def _N_or_Y_to_N_n(N_or_y):
    """
    Convert nside or map to (nside, npix).

    Parameters
    ----------
    N_or_y : int or array_like
        HEALPix nside or map.

    Returns
    -------
    N : int
        nside.
    n : int
        Number of pixels.
    """
    if isinstance(N_or_y, int):
        n = hp.nside2npix(N_or_y)
        N = N_or_y
    else:
        N = hp.npix2nside(N_or_y.size)
        n = N_or_y.size
    return N, n


def vec(N_or_y):
    """
    Return xyz coordinates for all pixels of a HEALPix map.

    Parameters
    ----------
    N_or_y : int or array_like
        HEALPix nside or map.

    Returns
    -------
    xyz : ndarray, shape (npix, 3)
        Cartesian coordinates of pixels.
    """
    N, n = _N_or_Y_to_N_n(N_or_y)
    return np.array(hp.pix2vec(N, range(n))).T


def design_matrix(N_or_y, phase=None, inc=None, u=None, obl=None, normalize = True):
    """
    Compute the flux design matrix for a HEALPix map.

    Parameters
    ----------
    N_or_y : int or array_like
        HEALPix nside or map.
    phase : float, optional
        Rotation phase in radians.
    inc : float, optional
        Inclination in radians.
    u : array_like or None, optional
        Limb darkening coefficients.
    obl : float, optional
        Obliquity in radians.

    Returns
    -------
    matrix : ndarray
        Design matrix.
    """
    mask, projected_area, limb_darkening = mask_projected_limb(N_or_y, phase, inc, u, obl)
    geometry = mask * projected_area
    if normalize:
        return limb_darkening * geometry / (geometry * limb_darkening).sum()
    else:
        return limb_darkening * geometry


def flux(y, inc=None, u=None, phase=None, obl=None):
    """
    Compute the flux for a given map and orientation.

    Parameters
    ----------
    y : array_like
        HEALPix map.
    inc : float, optional
        Inclination in radians.
    u : array_like or None, optional
        Limb darkening coefficients.
    phase : float, optional
        Rotation phase in radians.
    obl : float, optional
        Obliquity in radians.

    Returns
    -------
    flux : float
        Integrated flux.
    """
    return design_matrix(y, inc=inc, u=u, phase=phase, obl=obl) @ y


def spherical_to_cartesian(theta, phi):
    """
    Convert spherical coordinates to cartesian.

    Parameters
    ----------
    theta : float or array_like
        Colatitude in radians.
    phi : float or array_like
        Longitude in radians.

    Returns
    -------
    xyz : ndarray, shape (3,) or (..., 3)
        Cartesian coordinates.
    """
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.array([x, y, z])


def spot(N, latitude, longitude, radius, sharpness=1000):
    """
    Generate a HEALPix map with a circular spot.

    Parameters
    ----------
    N : int
        HEALPix nside.
    latitude : float
        Latitude of the spot center in radians.
    longitude : float
        Longitude of the spot center in radians.
    radius : float
        Spot radius in radians.
    sharpness : float, optional
        Sharpness of the spot edge.

    Returns
    -------
    spot_map : ndarray
        HEALPix map with spot.
    """
    X = vec(N)
    d = distance(X, spherical_to_cartesian(jnp.pi / 2 - latitude, -longitude))
    return 1 - utils.sigmoid(d - radius, sharpness)


def soft_spot(N, latitude, longitude, radius):
    """
    Generate a HEALPix map with a soft-edged spot.

    Parameters
    ----------
    N : int
        HEALPix nside.
    latitude : float
        Latitude of the spot center in radians.
    longitude : float
        Longitude of the spot center in radians.
    radius : float
        Spot radius in radians.

    Returns
    -------
    spot_map : ndarray
        HEALPix map with soft-edged spot.
    """
    X = vec(N)
    d = distance(X, spherical_to_cartesian(jnp.pi / 2 - latitude, longitude))
    A = d / (2 * radius)
    C = 1 / 2
    profile = 0.5 * jnp.tanh(C - A) + 0.5 * jnp.tanh(C + A)
    return profile / jnp.max(profile)


def render(
    y, inc=None, u=None, phase=0.0, obl=0.0, xsize=800, period=None, radius=None
):
    """
    Render a HEALPix map as an orthographic projection.

    Parameters
    ----------
    y : array_like
        HEALPix map.
    inc : float, optional
        Inclination in radians.
    u : array_like or None, optional
        Limb darkening coefficients.
    phase : float, optional
        Rotation phase in radians.
    obl : float, optional
        Obliquity in radians.
    xsize : int, optional
        Output image size.

    Returns
    -------
    img : ndarray
        Rendered image.
    """
    import matplotlib.pyplot as plt

    N, n = _N_or_Y_to_N_n(y)

    phi, theta = hp.pix2ang(N, range(n))

    if period is not None and radius is not None:
        rv = radial_velocity(theta, phi, period, radius, phase, inc)
    else:
        rv = 1

    limb_darkening = mask_projected_limb(N, phase, inc, u)[2]
    rotated = hp.Rotator(
        rot=[phase, np.pi / 2 - inc or 0.0, obl or 0.0], deg=False
    ).rotate_map_pixel(y * limb_darkening * rv)

    projected_map = hp.orthview(
        rotated, half_sky=True, return_projected_map=True, xsize=xsize
    )
    plt.close()

    return projected_map


def amplitude(N_or_y, inc=None, u=None, undersampling: int = 3) -> callable:
    """
    Return a function to compute the amplitude of flux variations.

    Parameters
    ----------
    N_or_y : int or array_like
        HEALPix nside or map.
    inc : float, optional
        Inclination in radians.
    u : array_like or None, optional
        Limb darkening coefficients.
    undersampling : int, optional
        Undersampling factor for phase grid.

    Returns
    -------
    fun : callable
        Function that computes amplitude for a given map.
    """
    N, _ = _N_or_Y_to_N_n(N_or_y)
    resolution = hp.nside2resol(N)
    hp_resolution = resolution * undersampling
    phases = jnp.arange(0, 2 * jnp.pi, hp_resolution)

    mask, projected_area, limb_darkening = jax.jit(
        jax.vmap(jax.jit(mask_projected_limb), in_axes=(None, 0, None, None))
    )(N, phases, inc, u)

    geometry = mask * projected_area
    norm = (geometry * limb_darkening).sum(1)

    def fun(x):
        fluxes = (
            jnp.pi
            * jnp.einsum("ij,kj->ik", jnp.atleast_2d(x), limb_darkening * geometry)
            / norm
        )
        return jnp.ptp(fluxes, 1)

    return fun


def transit_chord(N, x, r, inc=None):
    """
    Compute mask for a transit chord across the stellar disk.

    Parameters
    ----------
    N : int
        HEALPix nside.
    x : float
        Chord center position.
    r : float
        Chord radius.
    inc : float, optional
        Inclination in radians.

    Returns
    -------
    mask : ndarray
        Boolean mask for pixels inside the chord.
    """
    if inc is None:
        c = 0.0
        s = 1.0
    else:
        c = jnp.sin(inc)
        s = jnp.cos(inc)
    _z, _y, _x = vec(N).T
    _x = _x * c - _z * s
    return jnp.abs(_x - x) < r


def radial_velocity(theta, phi, period, radius, phase, inc=None):
    """
    Compute the radial velocity at each pixel.

    Parameters
    ----------
    theta : array_like
        Colatitude in radians.
    phi : array_like
        Longitude in radians.
    period : float
        Rotation period in days.
    radius : float
        Stellar radius in solar radii.
    phase : float
        Rotation phase in radians.
    inc : float, optional
        Inclination in radians.

    Returns
    -------
    rv : ndarray
        Radial velocity at each pixel (m/s).
    """
    period_s = period * 24 * 60 * 60  # convert days to seconds
    omega = jnp.pi * 2 / period_s  # angular velocity
    radius_m = radius * 695700000.0  # convert solar radii to meters
    sin_phi = np.sin(phi)  # numpy here! as phi is static
    rv = radius_m * omega * jnp.sin(theta - phase) * sin_phi
    if inc is not None:
        rv = rv * jnp.sin(inc)
    return rv


def doppler_shift(theta, phi, period, radius, phase, inc=None):
    """
    Compute the Doppler shift at each pixel.

    Parameters
    ----------
    theta : array_like
        Colatitude in radians.
    phi : array_like
        Longitude in radians.
    period : float
        Rotation period in days.
    radius : float
        Stellar radius in solar radii.
    phase : float
        Rotation phase in radians.
    inc : float, optional
        Inclination in radians.

    Returns
    -------
    shift : ndarray
        Doppler shift at each pixel (fractional).
    """
    rv = radial_velocity(theta, phi, period, radius, phase, inc)
    c = 299792458.0
    shift = rv / c
    return shift


def shifted_spectra(spectra, shift):
    """
    Apply a Doppler shift to spectra.

    Parameters
    ----------
    spectra : ndarray, shape (n_pixels, n_wavelength)
        Input spectra.
    shift : ndarray, shape (n_pixels, n_wavelength)
        Doppler shift for each pixel and wavelength.

    Returns
    -------
    shifted : ndarray
        Shifted spectra.
    """
    _, n_wavelength = spectra.shape
    spectra_ft = jnp.fft.fft(spectra, axis=1)
    k = np.fft.fftfreq(n_wavelength).reshape(1, -1)
    phase_shift = jnp.exp(-2j * np.pi * k * shift)
    shifted = jnp.fft.ifft(spectra_ft * phase_shift, axis=1)
    return jnp.real(shifted)


def integrated_spectrum(
    design_matrix, theta, phi, period, radius, wv, spectra, phase, normalize=True
):
    """
    Compute the integrated spectrum of a rotating star.

    Parameters
    ----------
    design_matrix : array_like
        Flux design matrix for a HEALPix map.
    theta : array_like
        Colatitude in radians.
    phi : array_like
        Longitude in radians.
    period : float
        Rotation period in days.
    radius : float
        Stellar radius in solar radii.
    wv : array_like
        Wavelength grid.
    spectra : ndarray
        Spectra at each pixel.
    phase : float
        Rotation phase in radians.
    inc : float
        Inclination in radians.

    Returns
    -------
    integrated : ndarray
        Integrated spectrum.
    """
    spectra = jnp.atleast_2d(spectra)
    if period is None:
        spectra_shifted = spectra.T
    else:
        w_shift = doppler_shift(theta, phi, period, radius, phase)
        dw = wv[1] - wv[0]
        shift = w_shift[:, None] * wv / dw
        spectra_shifted = shifted_spectra(spectra.T, shift)
    if normalize:
        integrated_spec = jnp.sum(
            spectra_shifted * design_matrix.T, 0
        ) / jnp.sum(design_matrix.T * spectra.T)
    else:
        integrated_spec = jnp.einsum("ij,ij->j", design_matrix.T, spectra_shifted)

    return integrated_spec
