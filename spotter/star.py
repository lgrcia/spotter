import equinox as eqx
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from spotter import core


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
    N: int = 64
    n: int = eqx.field(static=True)
    phis: ArrayLike = eqx.field(static=True)  # lat
    thetas: ArrayLike = eqx.field(static=True)  # lon

    def __init__(self, N: int = 64):
        self.N = N
        self.n = hp.nside2npix(self.N)
        self.thetas, self.phis = jnp.array(hp.pix2ang(self.N, jnp.arange(self.n)))

    def _spots(self, accumulate=False, jit=True):

        if jit:
            query = jax.jit(
                jnp.vectorize(
                    core.query_disk(self.phis, self.thetas),
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

    def spots(self, lat, lon, r, summed=True, cumulative=False):
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

    def masked(self, x, phase=0.0) -> ArrayLike:
        mask = core.hemisphere_mask(self.phis, phase)
        return x * mask

    def limbed(self, x, u, phase=0.0) -> ArrayLike:
        limb_darkening = core.polynomial_limb_darkening(
            self.phis, self.thetas, u, phase
        )
        return x * limb_darkening

    def masked_limbed(self, x, u, phase=0.0) -> ArrayLike:
        mask = core.hemisphere_mask(self.phis, phase)
        limb_darkening = core.polynomial_limb_darkening(
            self.phis, self.thetas, u, phase
        )
        return x * limb_darkening * mask

    def area(self, phase=0.0) -> ArrayLike:
        return core.projected_area(self.phis, self.thetas, phase)

    def flux(self, x, u, phase) -> float:
        mask = core.hemisphere_mask(self.phis, phase)
        limb_darkening = core.polynomial_limb_darkening(
            self.phis, self.thetas, u, phase
        )
        projected_area = core.projected_area(self.phis, self.thetas, phase)
        limbed = x * limb_darkening
        geometry = mask * projected_area
        return jnp.pi * (limbed * geometry).sum() / (geometry * limb_darkening).sum()

    def amplitude(self, u, resolution=3) -> ArrayLike:
        hp_resolution = hp.nside2resol(self.N) * resolution
        phases = jnp.arange(0, 2 * jnp.pi, hp_resolution)

        mask = jax.vmap(core.hemisphere_mask, in_axes=(None, 0))(self.phis, phases)
        projected_area = jax.vmap(core.projected_area, in_axes=(None, None, 0))(
            self.phis, self.thetas, phases
        )
        limb_darkening = jax.vmap(
            core.polynomial_limb_darkening, in_axes=(None, None, None, 0)
        )(self.phis, self.thetas, u, phases)

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

    def render(self, x, u, phase=0.0):
        import matplotlib.pyplot as plt

        limb_darkening = core.polynomial_limb_darkening(self.phis, self.thetas, u, 0.0)
        rotated = hp.Rotator(rot=[phase, 0], deg=False).rotate_map_pixel(x)
        limbed = rotated * limb_darkening

        projected_map = hp.orthview(limbed, half_sky=True, return_projected_map=True)
        plt.close()

        return projected_map

    def show(self, x, u=None, phase=0.0, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if u is None:
            u = ()

        kwargs.setdefault("cmap", "magma")
        kwargs.setdefault("origin", "lower")
        ax = ax or plt.gca()

        img = self.render(x, u, phase)
        ax.axis(False)
        ax.imshow(img, **kwargs)

    def transit_chord(self, r, b=0.0):
        """
        Return the map of a transit chord.

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
