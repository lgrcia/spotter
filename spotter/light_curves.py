"""
Light curve and design matrix utilities for rotating and transited stars described by HEALPix maps.

This module provides functions to compute design matrices, light curves, and
transit light curves for stars with arbitrary surface maps and limb darkening.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from spotter import core, utils
from spotter.star import Star, transited_star


def design_matrix(star: Star, time: ArrayLike, normalize: bool = True) -> ArrayLike:
    """
    Compute the design matrix for a rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : ArrayLike
        Time array in days.

    Returns
    -------
    matrix : ndarray
        Design matrix.
    """
    def impl(star, time):
        if star.u is not None:
            if len(star.y) == 1:
                return jax.vmap(
                    lambda u: core.design_matrix(star.y[0], star.phase(time), star.inc, u, normalize = normalize)
                )(star.u)
            else:
                if len(star.u) == 1:
                    return jax.vmap(
                        lambda y: core.design_matrix(
                            y, star.phase(time), star.inc, star.u[0], normalize = normalize
                        )
                    )(star.y)
                else:
                    return jax.vmap(
                        lambda y, u: core.design_matrix(y, star.phase(time), star.inc, u, normalize = normalize)
                    )(star.y, star.u)
        else:
            return jax.vmap(
                lambda y: core.design_matrix(y, star.phase(time), star.inc, star.u, normalize = normalize)
            )(star.y)
    
    return jnp.vectorize(impl, excluded=(0,), signature="()->(m,n)")(star, time)


def light_curve(star: Star, time: ArrayLike, normalize=True) -> ArrayLike:
    """
    Compute the light curve of a rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : ArrayLike
        Time array in days.
    normalize : bool, optional
        Whether to normalize the light curve (default True).

    Returns
    -------
    lc : ndarray
        Light curve array.
    """

    def impl(star, time):
        return jnp.einsum("ij,ij->i", design_matrix(star, time), star.y)

    norm = 1 / jnp.mean(star.y) if normalize else 1.0

    return jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time).T * norm


def transit_design_matrix(star, x, y, z, r, time=None, normalize = True):
    """
    Compute the design matrix for a transited Star.

    Parameters
    ----------
    star : Star
        Star object.
    x : float
        x coordinate of the disk center.
    y : float
        y coordinate of the disk center.
    z : float
        z coordinate of the disk center.
    r : float
        Radius of the disk.
    time : float or None, optional
        Time in days.

    Returns
    -------
    matrix : ndarray
        Transit design matrix.
    """
    X = design_matrix(star, time, normalize)

    from jax.scipy.spatial.transform import Rotation

    _z, _y, _x = core.vec(star.sides).T
    v = jnp.stack((_x, _y, _z), axis=-1)

    phase = star.phase(time)
    # # ensures non-zero phase
    # phase = jnp.where(phase == 0.0, 1.0, phase)
    _rv = Rotation.from_rotvec([phase, 0.0, 0.0]).apply(v)
    rv = jnp.where(phase == 0.0, v, _rv)

    inc_angle = -jnp.pi / 2 + star.inc if star.inc is not None else 0.0
    _inc_angle = jnp.where(inc_angle == 0.0, 1.0, inc_angle)
    _rv = Rotation.from_rotvec([0.0, _inc_angle, 0.0]).apply(rv)
    rv = jnp.where(inc_angle == 0.0, rv, _rv)

    if star.obl is not None:
        obl_angle = jnp.where(star.obl == 0.0, 1.0, star.obl)
        _rv = Rotation.from_rotvec([0.0, 0.0, obl_angle]).apply(rv)
        rv = jnp.where(obl_angle == 0.0, rv, _rv)

    _x, _y, _ = rv.T

    distance = jnp.linalg.norm(
        jnp.array([_x, _y]) - jnp.array([y, -x])[:, None], axis=0
    )

    transited_y = utils.sigmoid(distance - r, 1000.0)
    return X * jnp.where(z >= 0, transited_y, jnp.ones_like(transited_y))


def transit_light_curve(
    star: Star,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    r: float = 0.0,
    time: float = 0.0,
    normalize=True,
):
    """
    Compute the light curve of a transited Star.

    Parameters
    ----------
    star : Star
        Star object.
    x : float, optional
        x coordinate of the disk center (default 0.0).
    y : float, optional
        y coordinate of the disk center (default 0.0).
    z : float, optional
        z coordinate of the disk center (default 0.0).
    r : float, optional
        Radius of the disk (default 0.0).
    time : float, optional
        Time in days (default 0.0).
    normalize : bool, optional
        Whether to normalize the light curve (default True).

    Returns
    -------
    lc : ndarray
        Transit light curve array.
    """

    def impl(star, time, x, y, z):
        return jnp.einsum(
            "ij,ij->i", transit_design_matrix(star, x, y, z, r, time), star.y
        )

    norm = 1 / jnp.mean(star.y) if normalize else 1.0

    return (
        jnp.vectorize(impl, excluded=(0,), signature="(),(),(),()->(n)")(
            star, time, x, y, z
        ).T
        * norm
    )

