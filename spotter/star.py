import equinox as eqx
import healpy as hp
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from spotter import core, viz


class Star(eqx.Module):
    """Docstring for Star."""

    y: ArrayLike
    u: ArrayLike | None = None
    period: float | None = None
    inc: float | None = None
    radius: float | None = None
    wv: float | None = None
    sides: int = eqx.field(static=True)

    def __init__(
        self,
        y: ArrayLike | None = None,
        u: ArrayLike | None = None,
        inc: float | None = None,
        period: float | None = None,
        radius: float | None = None,
        wv: float | None = None,
    ):
        self.y = jnp.atleast_2d(y)
        self.u = jnp.atleast_2d(u) if u is not None else None
        self.inc = inc
        self.period = period
        self.sides = core._N_or_Y_to_N_n(self.y[0])[0]
        self.radius = radius if radius is not None else 1.0
        self.wv = wv

    @property
    def x(self):
        return core.vec(self.sides)

    @property
    def size(self):
        return hp.nside2npix(self.sides)

    @property
    def resolution(self):
        return hp.nside2resol(self.sides)

    def __getitem__(self, key):
        return self.set(y=self.y[key])

    @classmethod
    def from_sides(cls, sides: int, **kwargs):
        y = np.ones(core._N_or_Y_to_N_n(sides)[1])
        return cls(y, **kwargs)

    def phase(self, time: ArrayLike) -> ArrayLike:
        return (
            2 * jnp.pi * time / self.period
            if self.period is not None
            else jnp.zeros_like(time)
        )

    def __mul__(self, other):
        if isinstance(other, Star):
            y = self.y * other.y
        else:
            y = self.y * other
        return self.__class__(y, self.u, self.inc, self.period)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Star):
            y = self.y + other.y
        else:
            y = self.y + other
        return self.__class__(y, self.u, self.inc, self.period)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Star):
            y = self.y - other.y
        else:
            y = self.y - other
        return self.__class__(y, self.u, self.inc, self.period)

    def __rsub__(self, other):
        return self.__sub__(other)

    def set(self, **kwargs):
        current = {
            "y": self.y,
            "u": self.u,
            "inc": self.inc,
            "period": self.period,
            "radius": self.radius,
            "wv": self.wv,
        }
        current.update(kwargs)
        return Star(**current)

    def set_y(self, y):
        return self.set(y=y)


def show(star: Star, phase: ArrayLike = 0.0, ax=None, **kwargs):
    viz.show(
        star.y[0],
        star.inc if star.inc is not None else np.pi / 2,
        star.u[0] if star.u is not None else None,
        phase,
        ax=ax,
        **kwargs,
    )


def video(star, duration=4, fps=10, **kwargs):
    viz.video(
        star.y[0],
        star.inc if star.inc is not None else np.pi / 2,
        star.u[0] if star.u is not None else None,
        duration=duration,
        fps=fps,
        **kwargs,
    )


def transited_star(star, x=0.0, y=0.0, r=0.0):
    if star.inc is None:
        c = 0.0
        s = 1.0
    else:
        c = jnp.sin(star.inc)
        s = jnp.cos(star.inc)
    _z, _y, _x = core.vec(star.sides).T
    _x = _x * c - _z * s
    return (
        1
        - (
            jnp.linalg.norm(jnp.array([_x, _y]) - jnp.array([x, -y])[:, None], axis=0)
            < r
        )
    ) * star
