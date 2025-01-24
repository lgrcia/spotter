from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from spotter import core
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


def light_curve(star: Star, time: ArrayLike) -> ArrayLike:
    """Light curve of a rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : ArrayLike
        Time array in days.

    Returns
    -------
    ArrayLike
        Light curve array.
    """

    def impl(star, time):
        return jnp.einsum("ij,ij->i", design_matrix(star, time), star.y)

    return (
        jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time).T / jnp.pi
    )


def transit_light_curve(
    star: Star,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    r: float = 0.0,
    time: float = 0.0,
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

    Returns
    -------
    ArrayLike
        Light curve array.
    """
    return light_curve(transited_star(star, y, x, z, r), star.phase(time))
