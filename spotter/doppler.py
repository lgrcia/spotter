from functools import partial

import healpy as hp
import jax.numpy as jnp
from jax.typing import ArrayLike

from spotter import core, light_curves
from spotter.star import Star


@partial(jnp.vectorize, excluded=(0,), signature="()->(n)")
def spectrum(star: Star, time: float) -> ArrayLike:
    """Integrated spectrum of a rotating Star.

    Parameters
    ----------
    star : Star
        Star object.
    time : float
        Time in days.

    Returns
    -------
    ArrayLike
        Integrated spectrum.
    """
    phi, theta = hp.pix2ang(star.sides, range(hp.nside2npix(star.sides)))
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
    )


def rv_design_matrix(star: Star, time: float) -> ArrayLike:
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
    def impl(star, time):
        return jnp.einsum(
            "ij,ij->i",
            rv_design_matrix(star, time),
            star.y,
        ) / light_curves.light_curve(star, time, normalize=False)

    return jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time).T
