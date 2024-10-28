from functools import partial

import jax.numpy as jnp
from jax.typing import ArrayLike

from spotter import core
from spotter.star import Star, transited_star


@partial(jnp.vectorize, excluded=(0,), signature="()->(n)")
def design_matrix(star: Star, time: ArrayLike) -> ArrayLike:
    return core.design_matrix(star.y, star.phase(time), star.inc, star.u)


@partial(jnp.vectorize, excluded=(0,), signature="()->()")
def light_curve(star: Star, time: ArrayLike) -> ArrayLike:
    return design_matrix(star, time) @ star.y


def transit_light_curve(star, x=0.0, y=0.0, r=0.0, phase=0.0):
    return light_curve(transited_star(star, x, y, r), phase)
