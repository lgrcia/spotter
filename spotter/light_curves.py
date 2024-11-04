import jax

from functools import partial

import jax.numpy as jnp
from jax.typing import ArrayLike

from spotter import core
from spotter.star import Star, transited_star


@partial(jnp.vectorize, excluded=(0,), signature="()->(m,n)")
def design_matrix(star: Star, time: ArrayLike) -> ArrayLike:
    if star.u is not None:
        if len(star.y) == 1:
            return jax.vmap(
                lambda u: core.design_matrix(star.y[0], star.phase(time), star.inc, u)
            )(star.u)
        else:
            return jax.vmap(
                lambda y, u: core.design_matrix(y, star.phase(time), star.inc, u)
            )(star.y, star.u)
    else:
        return jax.vmap(
            lambda y: core.design_matrix(y, star.phase(time), star.inc, star.u)
        )(star.y)


def light_curve(star: Star, time: ArrayLike) -> ArrayLike:
    def impl(star, time):
        return jnp.einsum("ij,ij->i", design_matrix(star, time), star.y)

    return (
        jnp.vectorize(impl, excluded=(0,), signature="()->(n)")(star, time).T / jnp.pi
    )


def transit_light_curve(star, x=0.0, y=0.0, r=0.0, phase=0.0):
    return light_curve(transited_star(star, x, y, r), phase)
