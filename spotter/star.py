import equinox as eqx
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from spotter import core, utils, viz


class Star(eqx.Module):
    """
    A Star object whose surface is described by HEALPix map(s).

    The HEALPix maps can be a 2D array with a shape of (wavelengths, pixels), or a
    1D array with a shape of (pixels).

    When providing polynomial limb darkening coefficients, different options are possible:

    * u is 1D and y is 1D: Single set of limb darkening coefficients and a single map.
    * u is 1D and y is 2D: The same limb darkening coefficients are applied to all wavelength maps.
    * u is 2D and y is 1D: The limb darkening coefficients are different for each wavelength but the map is the same.
    * ``u.shape[0]`` == ``y.shape[0]``: u and y are 2D arrays specifying the limb darkening coeffs and maps for each wavelength.

    Parameters
    ----------
    y : ArrayLike or None, optional
        HEALPix map of the star, with shape (pixels,) or (wavelengths, pixels). Must be provided.
    u : ArrayLike or None, optional
        Polynomial limb darkening coefficients with shape (order,) or (wavelengths, order). By default None.
        If provided, must either be coefficients applied to all wavelengths, or have the same length as y
        (i.e. defined for the same number of wavelengths).
    inc : float or None, optional
        Inclination of the star, in radians. 0 is pole-on, pi/2 is equator-on. By default None.
    obl : float or None, optional
        Obliquity of the star, in radians. 0 is no obliquity, pi/2 is maximum obliquity. By default None.
    period : float or None, optional
        Period of the star, in days. By default None.
    radius : float or None, optional
        Radius of the star, in solar radii. By default None.
    wv : float or None, optional
        Wavelength of the star maps, in meters. By default None. If provided, must be compatible with either
        the shape of u and/or y.

    Attributes
    ----------
    y : ArrayLike
        HEALPix map of the star, with shape (wavelengths, pixels).
    u : ArrayLike or None
        Polynomial limb darkening coefficients with shape (wavelengths, order).
    period : float or None
        Period of the star, in days.
    inc : float or None
        Inclination of the star, in radians. 0 is pole-on, pi/2 is equator-on.
    obl : float or None
        Obliquity of the star, in radians. 0 is no obliquity, pi/2 is maximum obliquity.
    radius : float or None
        Radius of the star, in solar radii.
    wv : float or None
        Wavelength of the star maps, in meters.
    sides : int
        Number of HEALPix sides.

    Examples
    --------

    .. plot::

        import numpy as np
        from spotter.star import Star, show

        star = Star.from_sides(30, inc=0.5, u=(0.4, 0.3), obl=0.5)
        show(star)

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
    def N(self):
        """Return the number of sides of the star map."""
        return self.sides

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
        """
        Return a new Star with selected wavelength(s).

        Parameters
        ----------
        key : int, slice, or array_like
            Index or indices to select.

        Returns
        -------
        Star
            New Star object with selected map(s).
        """
        return self.set(y=self.y[key])

    @classmethod
    def from_sides(cls, sides: int, **kwargs):
        """
        Create a Star object with a given number of sides.

        Parameters
        ----------
        sides : int
            Number of sides of the HEALPix map.
        **kwargs
            Additional keyword arguments for Star.

        Returns
        -------
        Star
            Star object with the given number of sides.
        """
        y = np.ones(core._N_or_Y_to_N_n(sides)[1])
        return cls(y, **kwargs)

    def phase(self, time: ArrayLike | None) -> ArrayLike:
        """
        Compute the rotation phase for a given time.

        Parameters
        ----------
        time : array_like or None
            Time(s) in days.

        Returns
        -------
        phase : float or array_like
            Rotation phase(s) in radians.
        """
        if time is None:
            return 0.0
        return (
            2 * jnp.pi * time / self.period
            if self.period is not None
            else jnp.zeros_like(time)
        )

    def __mul__(self, other):
        """
        Multiply the star map by another Star or scalar.

        Parameters
        ----------
        other : Star or scalar
            Object to multiply with.

        Returns
        -------
        Star
            Resulting Star object.
        """
        if isinstance(other, Star):
            y = self.y * other.y
        else:
            y = self.y * other
        return self.set(y=y)

    def __rmul__(self, other):
        """
        Multiply the star map by another Star or scalar (right-mult).

        Parameters
        ----------
        other : Star or scalar
            Object to multiply with.

        Returns
        -------
        Star
            Resulting Star object.
        """
        return self.__mul__(other)

    def __add__(self, other):
        """
        Add another Star or scalar to the star map.

        Parameters
        ----------
        other : Star or scalar
            Object to add.

        Returns
        -------
        Star
            Resulting Star object.
        """
        if isinstance(other, Star):
            y = self.y + other.y
        else:
            y = self.y + other
        return self.set(y=y)

    def __radd__(self, other):
        """
        Add another Star or scalar to the star map (right-add).

        Parameters
        ----------
        other : Star or scalar
            Object to add.

        Returns
        -------
        Star
            Resulting Star object.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract another Star or scalar from the star map.

        Parameters
        ----------
        other : Star or scalar
            Object to subtract.

        Returns
        -------
        Star
            Resulting Star object.
        """
        if isinstance(other, Star):
            y = self.y - other.y
        else:
            y = self.y - other
        return self.set(y=y)

    def __rsub__(self, other):
        """
        Subtract the star map from another Star or scalar (right-sub).

        Parameters
        ----------
        other : Star or scalar
            Object to subtract from.

        Returns
        -------
        Star
            Resulting Star object.
        """
        return self.__sub__(other)

    def set(self, **kwargs):
        """
        Return a Star object with updated attributes.

        Parameters
        ----------
        **kwargs
            Attributes to update.

        Returns
        -------
        Star
            Star object with updated attributes.
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

    def spot(self, lat: float, lon: float, radius: float, sharpness: float = 20):
        """
        Return a HEALPix map with a spot.

        Parameters
        ----------
        lat : float
            Latitude of the spot, in radians.
        lon : float
            Longitude of the spot, in radians.
        radius : float
            Radius of the spot, in radians.
        sharpness : float, optional
            Sharpness of the spot edge (default 20).

        Returns
        -------
        ArrayLike
            HEALPix map with a spot.
        """
        return core.spot(self.sides, lat, lon, radius, sharpness=sharpness)

    @property
    def coords(self):
        """
        Return the coordinates of the star pixels.

        Returns
        -------
        coords : ndarray
            Cartesian coordinates of pixels.
        """
        return core.vec(self.sides)


def show(star: Star, phase: ArrayLike = 0.0, ax=None, xsize=800, **kwargs):
    """
    Show the star map. If `star.y` is 2D, the first map is shown.

    Parameters
    ----------
    star : Star
        Star object to show.
    phase : ArrayLike, optional
        Phase of the star map to show (default 0.0).
    ax : matplotlib axis, optional
        Axis to plot the star map (default None).
    xsize : int, optional
        Output image size (default 800).
    **kwargs
        Additional keyword arguments for viz.show.
    """
    viz.show(
        star.y[0],
        star.inc if star.inc is not None else np.pi / 2,
        star.obl if star.obl is not None else 0.0,
        star.u[0] if star.u is not None else None,
        phase=phase,
        ax=ax,
        xsize=xsize,
        **kwargs,
    )


def video(star: Star, duration: int = 4, fps: int = 10, **kwargs):
    """
    Create an HTML video of the star map (for Jupyter notebooks).

    Parameters
    ----------
    star : Star
        Star object to show.
    duration : int, optional
        Duration of the video in seconds (default 4).
    fps : int, optional
        Frames per second (default 10).
    **kwargs
        Additional keyword arguments for viz.video.
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
    star: Star,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    r: float = 0.0,
    time: float = None,
):
    """
    Return a star transited by a circular opaque disk.

    Parameters
    ----------
    star : Star
        Star object to be transited.
    x : float, optional
        x coordinate of the disk center (default 0.0).
    y : float, optional
        y coordinate of the disk center (default 0.0).
    z : float, optional
        z coordinate of the disk center (default 0.0).
    r : float, optional
        Radius of the disk (default 0.0).
    time : float, optional
        Time in days (default None).

    Returns
    -------
    Star
        Star object transited by the disk.
    """
    from jax.scipy.spatial.transform import Rotation

    _z, _y, _x = core.vec(star.sides).T
    v = jnp.stack((_x, _y, _z), axis=-1)

    if time is not None:
        phase = star.phase(time)
        _rv = Rotation.from_rotvec([phase, 0.0, 0.0]).apply(v)
        rv = jnp.where(phase == 0.0, v, _rv)
    else:
        rv = v

    inc_angle = -jnp.pi / 2 + star.inc if star.inc is not None else 0.0
    _inc_angle = jnp.where(inc_angle == 0.0, 1.0, inc_angle)
    _rv = Rotation.from_rotvec([0.0, _inc_angle, 0.0]).apply(rv)
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
