import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def _wrap(*args):
    n = len(args)
    signature = ",".join(["()"] * n)
    signature = f"{signature}->{signature}"
    new_args = np.vectorize(lambda *args: args)(*args)
    if new_args[0].shape == ():
        return [np.array([n]) for n in new_args]
    else:
        return new_args


class Star:
    def __init__(self, u=None, N=64):
        self.N = N
        self.u = u
        self.n = hp.nside2npix(self.N)
        self._phis, self._thetas = hp.pix2ang(self.N, np.arange(self.n))
        self._sin_phi = np.sin(self._phis)

        self._spot_map = np.zeros(self.n)
        self._faculae_map = np.zeros(self.n)

    def _z(self, phase=0):
        return self._sin_phi * np.cos(self._thetas - phase)

    def _ld(self, phase=0):
        if self.u is None:
            return 1
        else:
            z = self._z(phase)
            return 1 - np.sum([u * (1 - z) ** (n + 1) for n, u in enumerate(self.u)], 0)

    def _get_mask(self, phase=0):
        a = (phase + np.pi / 2) % (2 * np.pi)
        b = (phase - np.pi / 2) % (2 * np.pi)
        if a > phase % (2 * np.pi) and b < phase % (2 * np.pi):
            mask = (self._thetas < a) & (self._thetas > b)
        else:
            mask = (self._thetas > b) | (self._thetas < a)

        return mask

    def add_spot(self, theta, phi, radius, contrast):
        for t, p, r, c in zip(*_wrap(theta, phi, radius, contrast)):
            idxs = hp.query_disc(self.N, hp.ang2vec(t, p), r)
            self._spot_map[idxs] = c

    def add_faculae(self, theta, phi, radius_in, radius_out, contrast):
        for t, p, ri, ro, c in zip(*_wrap(theta, phi, radius_in, radius_out, contrast)):
            inner_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ri)
            outer_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ro)
            idxs = np.setdiff1d(outer_idxs, inner_idxs)
            self._faculae_map[idxs] = c

    def add_spot_faculae(
        self, theta, phi, radius_in, radius_out, contrast_spot, contrast_faculae
    ):
        for t, p, ri, ro, cs, cf in zip(
            *_wrap(theta, phi, radius_in, radius_out, contrast_spot, contrast_faculae)
        ):
            inner_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ri)
            outer_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ro)
            facuale_idxs = np.setdiff1d(outer_idxs, inner_idxs)
            self._faculae_map[facuale_idxs] = cf
            self._spot_map[inner_idxs] = cs

    def flux(self, phase=0):
        def _flux(phase):
            mask = self._get_mask(phase)
            limb_darkening = self._ld(phase)
            # spot contribution
            m = (1 - self._spot_map) * mask * limb_darkening
            # facuale contribution will have a different limb darkening
            # and another normalization will be needed
            return m.sum() / (mask * limb_darkening).sum()

        return np.vectorize(_flux)(phase)

    def m(self, phase=0):
        mask = self._get_mask(phase)
        limb_darkening = self._ld(phase)
        # spot contribution
        m = 1 - self._spot_map * mask * limb_darkening
        return m

    def show(self, phase=0, grid=False, return_img=False, **kwargs):
        kwargs.setdefault("cmap", "magma")
        # only spot contribution for now
        rotated_m = hp.Rotator(rot=[phase, 0], deg=False).rotate_map_pixel(
            1 - self._spot_map
        )
        projected_map = hp.orthview(
            rotated_m * self._ld(0), half_sky=True, return_projected_map=True
        )
        plt.close()
        if return_img:
            return projected_map
        else:
            plt.axis(False)
            plt.imshow(projected_map, **kwargs)

    def covering_fraction(
        self, phase: float = None, vmin: float = 0.01, vmax: float = 1.0
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

        Returns
        -------
        float
            full star or disk covering fraction
        """
        if phase is None:
            return np.sum(self._spot_map >= vmin) / self.n
        else:
            mask = self._get_mask(phase)
            return np.sum(self._spot_map[mask] >= vmin) / mask.sum()
