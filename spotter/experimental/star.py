import equinox as eqx
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from spotter import core
from spotter.experimental import core as xcore
from spotter.experimental import viz
from spotter.utils import Array

_DEFAULT_CMAP = "magma"


def _wrap(*args):
    n = len(args)
    signature = ",".join(["()"] * n)
    signature = f"{signature}->{signature}"
    new_args = np.vectorize(lambda *args: args)(*args)
    if new_args[0].shape == ():
        return [np.array([n]) for n in new_args]
    else:
        return new_args


class Star(eqx.Module):
    """An object holding the geometry of the stellar surface map."""

    N: int = 64
    """HEALPix map nside"""
    n: int = eqx.field(static=True)
    """Number of pixels"""
    inclination: float = 0.0
    """The inclination of the star in radians"""
    u: ArrayLike = None
    """The polynomial limb darkening coefficients"""
    y: ArrayLike = eqx.field(static=True)
    """pixel values"""
    X: ArrayLike = eqx.field(static=True)
    """The unit vectors of the pixels"""

    def __init__(self, N: int = 64, inclination: float = 0.0, u: ArrayLike = None):
        """An object holding the geometry of the stellar surface map.

        Parameters
        ----------
        N : int, optional
            HEALPix map nside, by default 64
        """
        self.N = N
        self.n = hp.nside2npix(self.N)
        self.inclination = inclination
        self.u = u
        self.X = jnp.array(hp.pix2vec(self.N, range(self.n))).T

    def _spots(self, accumulate=False, jit=True):

        if jit:
            query = jax.jit(
                jnp.vectorize(
                    core.query_disk(self._phis, self._thetas),
                    signature="(),(),()->(n)",
                )
            )
        else:

            def query(lat, lon, r):
                lats, long, rs = _wrap(lat, lon, r)
                x = np.zeros((len(lats), self.n), dtype=np.int8)
                for i, (la, lo, r) in enumerate(zip(lats, long, rs)):
                    idxs = hp.query_disc(self.N, hp.ang2vec(la, lo), r)
                    x[i, idxs] = 1.0
                return x

        def fun(lat, lon, r):
            x = query(lat, lon, r)

            if accumulate is True and x.ndim == 2:
                x = jnp.cumsum(x, 0)
                x = jnp.asarray(x > 0, dtype=jnp.float64)

            return x

        return fun

    def spots(
        self,
        lat: Array,
        lon: Array,
        r: Array,
        summed: bool = True,
        cumulative: bool = False,
    ):
        """Generate an HEALPix map of spots.

        Parameters
        ----------
        lat : Array
            latitude(s) of the spots
        lon : Array
            longitude(s) of the spots
        r : Array
            radius(ii) of the spots
        summed : bool, optional
            wether one map per spot is returned or summed, by default True
        cumulative : bool, optional
            wether each map contain a given spot plus all the previous ones,
            by default False

        Returns
        -------
        Array
            HEALPix map of the spots
        """
        if cumulative:
            summed = False

        if summed:
            x = np.zeros(self.n, dtype=np.int8)
            for t, p, r in zip(*_wrap(lat, lon, r)):
                idxs = hp.query_disc(self.N, hp.ang2vec(t, p), r)
                x[idxs] = 1
        else:
            lats, lons, rs = _wrap(lat, lon, r)
            x = np.zeros((len(lats), self.n), dtype=np.int8)
            for i, (t, p, r) in enumerate(zip(lats, lons, rs)):
                idxs = hp.query_disc(self.N, hp.ang2vec(t, p), r)
                x[i, idxs] = 1

            if cumulative:
                x = np.cumsum(x, 0)
                x = (x > 0).astype(np.int8)

        return x

    def smooth_spots(self, lat, lon, r, c=12):
        return jax.jit(core.smooth_spot(self._phis, self._thetas))(lat, lon, r, c)

    def masked(self, x: Array = None, phase: float = 0.0) -> Array:
        """Returns a map where pixels outside the visible hemisphere
           of the star are set to zero.

        Parameters
        ----------
        x : Array
            pixels map
        phase : float, optional
            phase in radians, by default 0.0

        Returns
        -------
        Array
            masked map
        """
        raise DeprecationWarning()
        if x is None:
            x = np.ones(self.n)
        mask = core.hemisphere_mask(self._phis, phase)
        return x * mask

    def limbed(self, x: Array = None, u: Array = None, phase=0.0) -> Array:
        """Returns a map multiplied by the polynomial limb law.

        Parameters
        ----------
        x : Array
            pixels map
        u : Array
            polynomial limb law coefficients
        phase : float, optional
            phase in radians, by default 0.0

        Returns
        -------
        Array
            limbed map
        """
        raise DeprecationWarning()
        if x is None:
            x = np.ones(self.n)
        limb_darkening = core.polynomial_limb_darkening(
            self._phis, self._thetas, u, phase
        )
        return x * limb_darkening

    def masked_limbed(self, x: Array = None, u: Array = None, phase=0.0) -> Array:
        """Returns a map where pixels outside the visible hemisphere
           of the star are set to zero and multiplied by the polynomial limb law.

        Parameters
        ----------
        x : Array
            map
        u : Array
            polynomial limb law coefficients
        phase : float, optional
            phase in radians, by default 0.0

        Returns
        -------
        Array
            masked and limbed map
        """
        raise DeprecationWarning()
        if x is None:
            x = np.ones(self.n)

        mask = core.hemisphere_mask(self._phis, phase)
        limb_darkening = core.polynomial_limb_darkening(
            self._phis, self._thetas, u, phase
        )
        return x * limb_darkening * mask

    def area(self, phase: float = 0.0) -> ArrayLike:
        """Returns the projected area of each pixels in the map.

         Parameters
        ----------
        phase : float, optional
            phase in radians, by default 0.0
        """
        raise DeprecationWarning()
        return core.projected_area(self._phis, self._thetas, phase)

    def flux(self, x: Array, phase: float) -> float:
        """Returns the total flux of the map.

        Parameters
        ----------
        x : Array
            map
        u : Array
            polynomial limb law coefficients
        phase : Array
            phase in radians

        Returns
        -------
        float
            integrated flux at the given phase
        """
        mask, projected_area, limb_darkening = xcore.mask_projected_limb(
            self.X, phase, self.inclination, self.u
        )
        limbed = x * limb_darkening
        geometry = mask * projected_area
        return jnp.pi * (limbed * geometry).sum() / (geometry * limb_darkening).sum()

    @property
    def resolution(self):
        """Resolution of the map in radians."""
        return hp.nside2resol(self.N)

    def single_spot_coverage(self, r: float):
        """Return the coverage of a single spot of radius r.

        Parameters
        ----------
        r : float
            radius of the spot in radians

        Returns
        -------
        float
            coverage of the spot
        """
        return ((2 * np.pi * (1 - np.cos(r))) / self.resolution**2) / self.n

    def amplitude(self, undersampling: int = 3) -> callable:
        """Returns a function to compute the amplitude of rotational light
           curve of a given map.

        Parameters
        ----------
        u : Array
            polynomial limb law coefficients
        resolution : int, optional
            undersampling of the light curve according to the
            resolution element of the map, by default 3

        Returns
        -------
        callable
            signature:
            - if single map: (map: Array) -> amplitude: float
            - if multiple maps: (maps: Array[Array]) -> amplitudes: Array
        """
        hp_resolution = self.resolution * undersampling
        phases = jnp.arange(0, 2 * jnp.pi, hp_resolution)

        mask, projected_area, limb_darkening = jax.jit(
            jax.vmap(jax.jit(xcore.mask_projected_limb), in_axes=(None, 0, None, None))
        )(self.X, phases, self.inclination, self.u)

        geometry = mask * projected_area
        norm = (geometry * limb_darkening).sum(1)

        def fun(x):
            fluxes = (
                np.pi
                * jnp.einsum("ij,kj->ik", jnp.atleast_2d(x), limb_darkening * geometry)
                / norm
            )
            return jnp.ptp(fluxes, 1)

        return fun

    def render(self, x: Array, phase=0.0):
        """Render the map disk at a given rotation phase.

        Parameters
        ----------
        x : Array
            map
        u : Array
            polynomial limb law coefficients
        phase : Array
            phase in radians, by default 0.0

        Returns
        -------
        Array[Array]
            Image of the map disk
        """
        import matplotlib.pyplot as plt

        limb_darkening = xcore.mask_projected_limb(
            self.X, phase, self.inclination, self.u
        )[2]
        rotated = hp.Rotator(rot=[phase, self.inclination], deg=False).rotate_map_pixel(
            x * limb_darkening
        )

        projected_map = hp.orthview(rotated, half_sky=True, return_projected_map=True)
        plt.close()

        return projected_map

    def show(self, x: Array = None, phase: float = 0.0, ax=None, **kwargs):
        """Show the map disk.

        Parameters
        ----------
        x : Array
            map
        u : Array
            polynomial limb law coefficients
        phase : Array
            phase in radians, by default 0.0
        ax : matplotlib.pyplot.Axe, optional
            by default None
        """
        import matplotlib.pyplot as plt

        if x is None:
            x = np.ones(self.n)

        kwargs.setdefault("cmap", _DEFAULT_CMAP)
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("vmin", 0.0)
        kwargs.setdefault("vmax", 1.0)
        ax = ax or plt.gca()

        img = self.render(x, phase)
        ax.axis(False)
        ax.imshow(img, extent=(-1, 1, -1, 1), **kwargs)
        viz.graticule(self.inclination, ax=ax)

    def transit_chord(self, r: float, b: float = 0.0):
        """
        Returns the map of a transit chord.

        Parameters
        ----------
        b : float
            Impact parameter of the transit chord.
        r : float
            Planet radius.
        """
        x = np.zeros(self.n, dtype=np.int8)
        theta1 = np.arccos(b + r)
        theta2 = np.arccos(b - r)
        x[hp.query_strip(self.N, theta1, theta2)] = 1.0
        return x

    def video(self, x, u=None, duration=4, fps=10, **kwargs):
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from IPython import display

        kwargs.setdefault("vmin", 0.0)
        kwargs.setdefault("vmax", 1.0)
        kwargs.setdefault("cmap", _DEFAULT_CMAP)
        kwargs.setdefault("origin", "lower")

        fig, ax = plt.subplots(figsize=(3, 3))
        im = plt.imshow(self.render(x), extent=(-1, 1, -1, 1), **kwargs)
        plt.axis("off")
        plt.tight_layout()
        ax.set_frame_on(False)
        fig.patch.set_alpha(0.0)
        frames = duration * fps

        def update(frame):
            a = im.get_array()
            phase = np.pi * 2 * frame / frames
            a = self.render(x, phase)
            for art in list(ax.lines):
                art.remove()
            viz.graticule(self.inclination, ax=ax, theta=phase, white_contour=False)
            im.set_array(a)
            return [im]

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=frames, interval=1000 / fps
        )
        video = ani.to_jshtml(embed_frames=True)
        html = display.HTML(video)
        plt.close()
        return display.display(html)
