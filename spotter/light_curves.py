from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from spotter import core, utils
from spotter.star import Star, transited_star


@partial(jnp.vectorize, excluded=(0,), signature="()->(m,n)")
def design_matrix(star: Star, time: ArrayLike) -> ArrayLike:
    """Design matrix for rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : ArrayLike
        Time array in days.

    Returns
    -------
    ArrayLike
        Design matrix.
    """
    if star.u is not None:
        if len(star.y) == 1:
            return jax.vmap(
                lambda u: core.design_matrix(star.y[0], star.phase(time), star.inc, u)
            )(star.u)
        else:
            if len(star.u) == 1:
                return jax.vmap(
                    lambda y: core.design_matrix(
                        y, star.phase(time), star.inc, star.u[0]
                    )
                )(star.y)
            else:
                return jax.vmap(
                    lambda y, u: core.design_matrix(y, star.phase(time), star.inc, u)
                )(star.y, star.u)
    else:
        return jax.vmap(
            lambda y: core.design_matrix(y, star.phase(time), star.inc, star.u)
        )(star.y)


def light_curve(star: Star, time: ArrayLike, normalize=True) -> ArrayLike:
    """Light curve of a rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : ArrayLike
        Time array in days.
    normalize: bool, optional
        Wether to normalize the light curve, by default True

    Returns
    -------
    ArrayLike
        Light curve array.
    """

    def impl(star, time):
        return jnp.einsum("ij,ij->i", design_matrix(star, time), star.y)

    norm = 1 / jnp.mean(star.y) if normalize else 1.0

    return jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time).T * norm


def transit_design_matrix(star, x, y, z, r, time=None):
    X = design_matrix(star, time)

    from jax.scipy.spatial.transform import Rotation

    _z, _y, _x = core.vec(star.sides).T
    v = jnp.stack((_x, _y, _z), axis=-1)

    phase = star.phase(time)
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
        jnp.array([_x, _y]) - jnp.array([x, -y])[:, None], axis=0
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
    """Light curve of a transited Star. The x-axis cross the star in the horizontal direction (→),
    and the y-axis cross the star in the vertical up direction (↑).
    Parameters
    ----------
    star : Star
        Star object.
    x : float, optional
        x coordinate of the center of the disk, by default 0.0.
    y : float, optional
        y coordinate of the center of the disk, by default 0.0.
    r : float, optional
        Radius of the disk, by default 0.0.
    time : float, optional
        Time array in days. by default 0.0.
    normalize: bool, optional
        Wether to normalize the light curve, by default True

    Returns
    -------
    ArrayLike
        Light curve array.
    """

    def impl(star, time):
        return jnp.einsum(
            "ij,ij->i", transit_design_matrix(star, x, y, z, r, time), star.y
        )

    norm = 1 / jnp.mean(star.y) if normalize else 1.0

    return jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time).T * norm
