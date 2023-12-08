import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


def _wrap(*args):
    n = len(args)
    signature = ",".join(["()"] * n)
    signature = f"{signature}->{signature}"
    new_args = np.vectorize(lambda *args: args)(*args)
    if new_args[0].shape == ():
        return [np.array([n]) for n in new_args]
    else:
        return new_args


def hemisphere_mask_function(thetas):
    def mask(phase):
        a = (phase + np.pi / 2) % (2 * np.pi)
        b = (phase - np.pi / 2) % (2 * np.pi)
        mask_1 = jnp.logical_and((thetas < a), (thetas > b))
        mask_2 = jnp.logical_or((thetas > b), (thetas < a))
        cond1 = a > phase % (2 * jnp.pi)
        cond2 = b < phase % (2 * jnp.pi)
        cond = cond1 * cond2
        return jnp.where(cond, mask_1, mask_2)

    return mask


def polynomial_limb_darkening(thetas, phis):
    def ld(u, phase):
        z = jnp.sin(phis) * jnp.cos(thetas - phase)
        terms = jnp.array([u * (1 - z) ** (n + 1) for n, u in enumerate(u)])
        return 1 - jnp.sum(terms, 0)

    return ld


# For the record, this is not faster than the C healpy wrapper
# i.e. 100x slower than using hp.query_disc. But if we ever need
# full jax compatibility, this is the way to go.
def query_idxs_function(thetas, phis):
    @jax.jit
    def query_idxs(theta, phi, radius):
        # https://en.wikipedia.org/wiki/Great-circle_distance
        # Vincenty formula
        p1 = phis - jnp.pi / 2
        p2 = theta - jnp.pi / 2

        t1 = thetas
        t2 = phi
        dl = jnp.abs((t1 - t2))

        sp1 = jnp.sin(p1)
        sp2 = jnp.sin(p2)
        cp1 = jnp.cos(p1)
        cp2 = jnp.cos(p2)
        cdl = jnp.cos(dl)
        sdl = jnp.sin(dl)

        a = (cp2 * sdl) ** 2 + (cp1 * sp2 - sp1 * cp2 * cdl) ** 2
        b = sp1 * sp2 + cp1 * cp2 * cdl
        d = jnp.arctan2(jnp.sqrt(a), b)

        return jnp.array(d <= radius)

    return query_idxs


class Star:
    def __init__(self, u=None, N=64, b=None, r=None):
        self.N = N
        self.u = u
        self.n = hp.nside2npix(self.N)
        self._phis, self._thetas = hp.pix2ang(self.N, np.arange(self.n))
        self._sin_phi = np.sin(self._phis)

        # these two maps are subject to different limb laws
        self.map_spot = None
        self.map_faculae = None
        self.clear_surface()

        self._cached_masks = None
        self._cached_ld = None

        self.hemisphere_mask = jax.vmap(hemisphere_mask_function(self._thetas))
        self.poly_lim_dark = jax.vmap(
            polynomial_limb_darkening(self._thetas, self._phis), in_axes=(None, 0)
        )

        # Define transit chord if impact parameter (b) and planet radius (r) provided
        self._transit_chord_map = np.zeros(self.n)
        self.b = b
        self.r = r
        if b is not None and r is not None:
            self.define_transit_chord(b, r)

    def _z(self, phase=0):
        return self._sin_phi * np.cos(self._thetas - phase)

    def clear_surface(self):
        self.map_spot = np.zeros(self.n)
        self.map_faculae = np.zeros(self.n)

    @property
    def resolution(self):
        return hp.nside2resol(self.N)

    def add_spot(self, theta, phi, radius, contrast):
        for t, p, r, c in zip(*_wrap(theta, phi, radius, contrast)):
            idxs = hp.query_disc(self.N, hp.ang2vec(t, p), r)
            self.map_spot[idxs] = c

    def add_faculae(self, theta, phi, radius_in, radius_out, contrast):
        for t, p, ri, ro, c in zip(*_wrap(theta, phi, radius_in, radius_out, contrast)):
            inner_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ri)
            outer_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ro)
            idxs = np.setdiff1d(outer_idxs, inner_idxs)
            self.map_faculae[idxs] = c

    def add_spot_faculae(
        self, theta, phi, radius_in, radius_out, contrast_spot, contrast_faculae
    ):
        for t, p, ri, ro, cs, cf in zip(
            *_wrap(theta, phi, radius_in, radius_out, contrast_spot, contrast_faculae)
        ):
            inner_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ri)
            outer_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ro)
            facuale_idxs = np.setdiff1d(outer_idxs, inner_idxs)
            self.map_faculae[facuale_idxs] = cf
            self.map_spot[inner_idxs] = cs

    def define_transit_chord(self, b, r):
        self.b = b
        self.r = r
        theta1 = np.arccos(b + r)
        theta2 = np.arccos(b - r)
        idx = hp.query_strip(self.N, theta1, theta2)
        self._transit_chord_map[idx] = 1

    def jax_flux(self, phases):
        mask = self.hemisphere_mask(phases)
        limb_darkening = self.poly_lim_dark(self.u, phases)

        @jax.jit
        def flux(spot_map):
            m = (1 - spot_map) * mask * limb_darkening
            return m.sum(1) / (mask * limb_darkening).sum(1)

        return flux

    def jax_amplitude(self, resolution=3):
        hp_resolution = hp.nside2resol(self.N) * resolution
        phases = np.arange(0, 2 * np.pi, hp_resolution)
        flux = self.jax_flux(phases)

        @jax.jit
        def amplitude(spot_map):
            f = flux(spot_map)
            return jnp.max(f) - jnp.min(f)

        return amplitude

    def flux(self, phases):
        mask = np.vectorize(
            hemisphere_mask_function(self._thetas), signature="()->(n)"
        )(phases)
        limb_darkening = np.vectorize(
            polynomial_limb_darkening(self._thetas, self._phis),
            signature="()->(n)",
            excluded={0},
        )(self.u, phases)
        m = (1 - self.map_spot) * mask * limb_darkening
        # faculae contribution, with same ld for now (TODO)
        m += self.map_faculae * mask * limb_darkening

        return m.sum(1) / (mask * limb_darkening).sum(1)

    def m(self, phase=0):
        mask = self._get_mask(phase)
        limb_darkening = self._ld(phase)
        # spot contribution
        m = (1 - self.map_spot) * mask * limb_darkening
        # faculae contribution, with same ld for now (TODO)
        m += self.map_faculae * mask * limb_darkening
        return m

    def show(self, phase=0, grid=False, return_img=False, transit_chord=True, **kwargs):
        kwargs.setdefault("cmap", "magma")
        # both spot and faculae with same ld for now (TODO)
        rotated_m = hp.Rotator(rot=[phase, 0], deg=False).rotate_map_pixel(
            (1 - self.map_spot) + self.map_faculae
        )
        projected_map = hp.orthview(
            rotated_m * self.poly_lim_dark(self.u, np.array([phase]))[0],
            half_sky=True,
            return_projected_map=True,
        )
        plt.close()
        if return_img:
            return projected_map
        else:
            plt.axis(False)
            plt.imshow(projected_map, **kwargs)
            if transit_chord:
                plt.plot()

    def covering_fraction(
        self,
        phase: float = None,
        vmin: float = 0.01,
        vmax: float = 1.0,
        transit_chord=False,
    ):
        """Return the covering fraction of active regions

        Either computed for the whole star (`phase=None`) or for the stellar
        disk given a phase

        Parameters
        ----------
        phase : float, optional
            stellar rotation phase, by default None
        vmin : float, optional
            minimum contrast value for spots, by default 0.01
        vmax : float, optional
            minimum contrast value for faculae, by default 1.0
        transit_chord : bool, optional
            calculate the covering fraction within the transit chord

        Returns
        -------
        float
            full star or disk covering fraction
        """
        if not transit_chord:
            if phase is None:
                return np.sum(self.map_spot >= vmin) / self.n
            else:
                mask = self._get_mask(phase)
                return np.sum(self.map_spot[mask] >= vmin) / mask.sum()
        elif transit_chord:
            in_chord = self._transit_chord_map
            is_spotted = self.map_spot >= vmin
            if phase is None:
                return np.logical_and(in_chord, is_spotted).sum() / in_chord.sum()
            else:
                mask = self._get_mask(phase)
                return (
                    np.logical_and(in_chord, is_spotted)[mask].sum()
                    / in_chord[mask].sum()
                )
