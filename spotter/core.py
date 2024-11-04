import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np


def distance(X, x):
    return jnp.arctan2(
        jnp.linalg.norm(jnp.cross(X.T, x[:, None], axis=0), axis=0), X @ x
    )


def npix(N):
    return hp.nside2npix(N)


def equator_coords(phi=None, inc=None):
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

    x = si * cp
    y = si * sp
    z = ci

    return jnp.array([x, y, z])


def mask_projected_limb(X, phase=None, inc=None, u=None):
    d = distance(X, equator_coords(phase, inc))
    mask = d < jnp.pi / 2
    z = jnp.cos(d)
    projected_area = z
    if u is not None:
        u = jnp.atleast_1d(u)
        terms = jnp.array([un * (1 - z) ** (n + 1) for n, un in enumerate(u)])
        limb_darkening = 1 - jnp.sum(terms, axis=z.ndim - 1)
    else:
        limb_darkening = jnp.ones_like(d)

    return mask, projected_area, limb_darkening


def _N_or_Y_to_N_n(N_or_y):
    if isinstance(N_or_y, int):
        n = hp.nside2npix(N_or_y)
        N = N_or_y
    else:
        N = hp.npix2nside(N_or_y.size)
        n = N_or_y.size
    return N, n


def vec(N_or_y):
    N, n = _N_or_Y_to_N_n(N_or_y)
    return np.array(hp.pix2vec(N, range(n))).T


def design_matrix(N_or_y, phase=None, inc=None, u=None):
    X = vec(N_or_y)
    mask, projected_area, limb_darkening = mask_projected_limb(X, phase, inc, u)
    geometry = mask * projected_area
    return jnp.pi * limb_darkening * geometry / (geometry * limb_darkening).sum()


def flux(y, inc=None, u=None, phase=None):
    return design_matrix(y, inc=inc, u=u, phase=phase) @ y


def spherical_to_cartesian(theta, phi):
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.array([x, y, z])


def spot(N, latitude, longitude, radius):
    X = vec(N)
    d = distance(X, spherical_to_cartesian(jnp.pi / 2 - latitude, longitude))
    return d < radius


def soft_spot(N, latitude, longitude, radius):
    X = vec(N)
    d = distance(X, spherical_to_cartesian(jnp.pi / 2 - latitude, longitude))
    A = d / (2 * radius)
    C = 1 / 2
    profile = 0.5 * jnp.tanh(C - A) + 0.5 * jnp.tanh(C + A)
    return profile / jnp.max(profile)


def render(y, inc=None, u=None, phase=0.0):
    import matplotlib.pyplot as plt

    X = vec(y)

    limb_darkening = mask_projected_limb(X, phase, inc, u)[2]
    rotated = hp.Rotator(
        rot=[phase, np.pi / 2 - inc or 0.0], deg=False
    ).rotate_map_pixel(y * limb_darkening)

    projected_map = hp.orthview(rotated, half_sky=True, return_projected_map=True)
    plt.close()

    return projected_map


def amplitude(N_or_y, inc=None, u=None, undersampling: int = 3) -> callable:

    N, _ = _N_or_Y_to_N_n(N_or_y)
    resolution = hp.nside2resol(N)
    X = vec(N)
    hp_resolution = resolution * undersampling
    phases = jnp.arange(0, 2 * jnp.pi, hp_resolution)

    mask, projected_area, limb_darkening = jax.jit(
        jax.vmap(jax.jit(mask_projected_limb), in_axes=(None, 0, None, None))
    )(X, phases, inc, u)

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
    if inc is None:
        c = 0.0
        s = 1.0
    else:
        c = jnp.sin(inc)
        s = jnp.cos(inc)
    _z, _y, _x = vec(N).T
    _x = _x * c - _z * s
    return jnp.abs(_x - x) < r


def doppler_shift(theta, phi, period, radius, phase):
    period_s = period * 24 * 60 * 60  # convert days to seconds
    omega = jnp.pi * 2 / period_s  # angular velocity
    radius_m = radius * 695700000.0  # convert solar radii to meters
    c = 299792458.0
    sin_phi = np.sin(phi)  # numpy here! as phi is static
    radial_velocity = radius_m * omega * jnp.sin(theta - phase) * sin_phi
    shift = radial_velocity / c
    return shift


def shifted_spectra(spectra, shift):
    _, n_wavelength = spectra.shape
    spectra_ft = jnp.fft.fft(spectra, axis=1)
    k = np.fft.fftfreq(n_wavelength).reshape(1, -1)
    phase_shift = jnp.exp(-2j * np.pi * k * shift)
    shifted = jnp.fft.ifft(spectra_ft * phase_shift, axis=1)
    return jnp.real(shifted)


def integrated_spectrum(N, theta, phi, period, radius, wv, spectra, phase, inc):
    spectra = jnp.atleast_2d(spectra)
    if period is None:
        spectra_shifted = spectra
    else:
        mask, projected, limb = mask_projected_limb(vec(N), phase, inc=inc)
        w_shift = doppler_shift(theta, phi, period, radius, phase)
        dw = wv[1] - wv[0]
        shift = w_shift[:, None] * wv / dw
        limb_geometry = projected * mask * limb
        spectra_shifted = shifted_spectra(spectra.T, shift)
    return jnp.sum(spectra_shifted * limb_geometry[:, None], 0) / jnp.sum(
        limb_geometry[:, None] * spectra.T
    )
