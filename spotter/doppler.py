import healpy as hp

from spotter.core import integrated_spectrum
from spotter.star import Star
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial


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
    return integrated_spectrum(
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
