from collections import defaultdict

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np


def sigmoid(x, scale=1000):
    return (jnp.tanh(x * scale / 2) + 1) / 2


def y1d_to_2d(ydeg: int, flm_1d: np.ndarray) -> np.ndarray:
    """1D starry Ylm to 2D s2fft"""
    new_flm = jnp.zeros((ydeg + 1, 2 * ydeg + 1), dtype=flm_1d.dtype)
    i = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            new_flm = new_flm.at[l, m + ydeg].set(flm_1d[i])
            i += 1

    return new_flm


def y2d_to_1d(ydeg: int, flm_2d: np.ndarray) -> np.ndarray:
    """2D starry Ylm to 1D s2fft"""
    new_flm = jnp.zeros((ydeg + 1) ** 2, dtype=flm_2d.dtype)
    i = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            new_flm = new_flm.at[i].set(flm_2d[l, m + ydeg])
            i += 1

    return new_flm


def C(l):
    """Complex to real conversion matrix"""
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
    """Converts a ylm array (starry like) to hp coefficients"""
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
