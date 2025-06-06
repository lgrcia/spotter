from collections import defaultdict

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np


def sigmoid(x, scale=1000):
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    x : float or array_like
        Input value(s).
    scale : float, optional
        Scaling factor (default 1000).

    Returns
    -------
    y : float or array_like
        Output value(s) in [0, 1].
    """
    return (jnp.tanh(x * scale / 2) + 1) / 2


def y1d_to_2d(ydeg: int, flm_1d: np.ndarray) -> np.ndarray:
    """
    Convert 1D starry Ylm to 2D s2fft format.

    Parameters
    ----------
    ydeg : int
        Maximum degree.
    flm_1d : ndarray
        1D array of coefficients.

    Returns
    -------
    flm_2d : ndarray
        2D array of coefficients.
    """
    new_flm = jnp.zeros((ydeg + 1, 2 * ydeg + 1), dtype=flm_1d.dtype)
    i = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            new_flm = new_flm.at[l, m + ydeg].set(flm_1d[i])
            i += 1

    return new_flm


def y2d_to_1d(ydeg: int, flm_2d: np.ndarray) -> np.ndarray:
    """
    Convert 2D s2fft Ylm to 1D starry format.

    Parameters
    ----------
    ydeg : int
        Maximum degree.
    flm_2d : ndarray
        2D array of coefficients.

    Returns
    -------
    flm_1d : ndarray
        1D array of coefficients.
    """
    new_flm = jnp.zeros((ydeg + 1) ** 2, dtype=flm_2d.dtype)
    i = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            new_flm = new_flm.at[i].set(flm_2d[l, m + ydeg])
            i += 1

    return new_flm


def C(l):
    """
    Complex to real spherical harmonics conversion matrix.

    Parameters
    ----------
    l : int
        Degree.

    Returns
    -------
    C : ndarray
        Conversion matrix.
    """
    # See https://doi.org/10.1016/s0166-1280(97)00185-1 (Blanco 1997, Eq. 19)
    A = np.eye(l, l)[:, ::-1]
    B = np.zeros(l)[:, None]
    C = np.diag((-1) ** np.arange(1, l + 1))

    ABC = np.hstack([A, B, C])
    jABC = np.hstack([1j * A, B, -1j * C])[::-1, :]
    one = np.zeros(2 * l + 1)
    one[l] = np.sqrt(2)

    return np.vstack([jABC, one, ABC]) / np.sqrt(2)


def ylm_to_hp(y, N):
    """
    Convert a ylm array (starry-like) to healpy coefficients and map.

    Parameters
    ----------
    y : ndarray
        Spherical harmonic coefficients.
    N : int
        HEALPix nside.

    Returns
    -------
    mh : ndarray
        HEALPix map.
    """
    from jaxoplanet.starry.core import rotation

    deg = int(np.floor(np.sqrt(len(y)))) - 1

    ry = rotation.dot_rotation_matrix(deg, None, None, 1.0, -np.pi / 2)(y)
    ry = rotation.dot_rotation_matrix(deg, 0.0, 1.0, None, -np.pi / 2)(ry)

    y2d = y1d_to_2d(deg, ry)
    c = C(deg)
    cy = y2d_to_1d(deg, (c.T.conj() @ y2d.T).T)

    _hy = defaultdict(lambda: 0 + 0j)
    i = 0

    for l in range(0, deg + 1):
        for m in range(-l, l + 1):
            j = hp.sphtfunc.Alm.getidx(deg, l, np.abs(m))
            _hy[j] = cy[i]
            i += 1

    hn = hp.sphtfunc.Alm.getsize(deg)
    hy = np.zeros(hn, dtype=np.complex128)
    for i in range(hn):
        hy[i] = _hy[i]

    mh = hp.alm2map(hy, nside=N)
    return mh
