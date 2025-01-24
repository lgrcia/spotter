import equinox as eqx
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from spotter import core, utils, viz


class Star(eqx.Module):
    """A Star object whose surface is describe by HEALPix map(s).

    The HEALPix maps can be a 2D array with a shape of (wavelengths, pixels), or a
    1D array with a shape of (pixels).

    When providing polynomial limb darkening coefficients, different options are possible:

    * u is 1D and y is 1D: Single set of limb darkening coefficients and a single map.
    * u is 1D and y is 2D: The same limb darkening coefficients are applied to all wavelengths maps.
    * u is 2D and y is 1D: The limb darkening coefficients are different for each wavelength but the map is the same.
    * ``u.shape[0]`` == ``y.shape[0]``: u and y are 2D arrays specifying the limb darkening coeffs and maps for each wavelength.

    Parameters
    ----------
    y : ArrayLike | None, optional
        HEALPix map of the star, with shape (pixels,) or (wavelengths, pixels). Must be provided.
    u : ArrayLike | None, optional
        Polynomial limb darkening coefficients with shape (order,) or (wavelengths, order). By default None.
        If provided, must either be coefficients applied to all wavelengths, or have the same length as y (
        i.e. defined for the same number of wavelengths).
    inc : float | None, optional
        Inclination of the star, in radians. 0 is pole-on, pi/2 is equator-on. By default None.
    period : float | None, optional
        Period of the star, in days. By default None
    radius : float | None, optional
        Radius of the star, in solar radii. By default None
    wv : float | None, optional
        Wavelength of the star maps, in meters. By default None. If  provided, must be compatible with either
        the shape of u and/or y.
    """

    y: ArrayLike
    """HEALPix map of the star, with shape (wavelengths, pixels)."""

    u: ArrayLike | None = None
    """Polynomial limb darkening coefficients with shape (wavelengths, order)."""

    period: float | None = None
    """Period of the star, in days."""

    inc: float | None = None
    """Inclination of the star, in radians. 0 is pole-on, pi/2 is equator-on."""

    obl: float | None = None
    """Obliquity of the star, in radians. 0 is no obliquity, pi/2 is maximum obliquity."""

    radius: float | None = None
    """Radius of the star, in solar radii."""

    wv: float | None = None
    """Wavelength of the star maps, in meters."""

    sides: int = eqx.field(static=True)
    """Number of HEALPix sides."""

    def __init__(
        self,
        y: ArrayLike | None = None,
        u: ArrayLike | None = None,
        inc: float | None = None,
        obl: float | None = None,
        period: float | None = None,
        radius: float | None = None,
        wv: float | None = None,
    ):
        self.y = jnp.atleast_2d(y)
        self.u = jnp.atleast_2d(u) if u is not None else None
        self.inc = inc
        self.obl = obl
        self.period = period
        self.sides = core._N_or_Y_to_N_n(self.y[0])[0]
        self.radius = radius if radius is not None else 1.0
        self.wv = wv

    @property
    def x(self):
        """Return the xyz coordinates of the star pixels."""
        return core.vec(self.sides)

    @property
    def size(self):
        """Return the number of pixels in the star map."""
        return hp.nside2npix(self.sides)

    @property
    def resolution(self):
        """Return the approximate size of a single map pixel in radians."""
        return hp.nside2resol(self.sides)

    def __getitem__(self, key):
        return self.set(y=self.y[key])

    @classmethod
    def from_sides(cls, sides: int, **kwargs):
        """Create a Star object with a given number of sides.

        Parameters
        ----------
        sides : int
            Number of sides of the HEALPix map.

        Returns
        -------
        Star
            Star object with the given number of sides.
        """
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
        return self.set(y=y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Star):
            y = self.y + other.y
        else:
            y = self.y + other
        return self.set(y=y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Star):
            y = self.y - other.y
        else:
            y = self.y - other
        return self.set(y=y)

    def __rsub__(self, other):
        return self.__sub__(other)

    def set(self, **kwargs):
        """Return a map with set attributes.

        Returns
        -------
        Star
            Star object with set attributes.
        """
        current = {
            "y": self.y,
            "u": self.u,
            "inc": self.inc,
            "obl": self.obl,
            "period": self.period,
            "radius": self.radius,
            "wv": self.wv,
        }
        current.update(kwargs)
        return Star(**current)


def show(star: Star, phase: ArrayLike = 0.0, ax=None, **kwargs):
    """Show the star map. If `star.y` is 2D, the first map is shown.

    Parameters
    ----------
    star : Star
        Star object to show.
    phase : ArrayLike, optional
        Phase of the star map to show, by default 0.0
    ax : matplotlib axis, optional
        Axis to plot the star map, by default None
    """
    viz.show(
        star.y[0],
        star.inc if star.inc is not None else np.pi / 2,
        star.obl if star.obl is not None else 0.0,
        star.u[0] if star.u is not None else None,
        phase,
        ax=ax,
        **kwargs,
    )


def video(star: Star, duration: int = 4, fps: int = 10, **kwargs):
    """Create a html video of the star map. Only suitable and displayed in jupyter notebooks.

    Parameters
    ----------
    star : Star
        Star object to show.
    duration : int, optional
        Duration of the video in seconds, by default
    fps : int, optional
        Frames per second of the video, by default 10
    """
    viz.video(
        star.y[0],
        star.inc if star.inc is not None else np.pi / 2,
        star.obl if star.obl is not None else 0.0,
        star.u[0] if star.u is not None else None,
        duration=duration,
        fps=fps,
        **kwargs,
    )


def transited_star(
    star: Star, x: float = 0.0, y: float = 0.0, z: float = 0.0, r: float = 0.0
):
    """Return a star transited by a circular opaque disk

    Parameters
    ----------
    star : Star
        Star object to be transited.
    x : float, optional
        x coordinate of the center of the disk, by default 0.0.
    y : float, optional
        y coordinate of the center of the disk, by default 0.0.
    r : float, optional
        Radius of the disk, by default 0.0.

    Returns
    -------
    Star
        Star object transited by the disk.
    """
    from jax.scipy.spatial.transform import Rotation

    _z, _y, _x = core.vec(star.sides).T
    v = jnp.stack((_x, _y, _z), axis=-1)

    inc_angle = -jnp.pi / 2 + star.inc if star.inc is not None else 0.0
    _inc_angle = jnp.where(inc_angle == 0.0, 1.0, inc_angle)
    _rv = Rotation.from_rotvec([0.0, _inc_angle, 0.0]).apply(v)
    rv = jnp.where(inc_angle == 0.0, v, _rv)

    if star.obl is not None:
        obl_angle = jnp.where(star.obl == 0.0, 1.0, star.obl)
        _rv = Rotation.from_rotvec([0.0, 0.0, obl_angle]).apply(rv)
        rv = jnp.where(obl_angle == 0.0, v, _rv)

    _x, _y, _ = rv.T

    distance = jnp.linalg.norm(
        jnp.array([_x, _y]) - jnp.array([x, -y])[:, None], axis=0
    )

    spotted_star = utils.sigmoid(distance - r, 1000.0) * star

    return star.set(y=jnp.where(z < 0, star.y, spotted_star.y))
