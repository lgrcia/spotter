import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


class Star:
    def __init__(self, u=None, N=64):
        self.N = N
        self.u = u
        self.n = hp.nside2npix(self.N)
        self._m = np.ones(self.n)
        self._phis, self._thetas = hp.pix2ang(self.N, np.arange(self.n))
        self._sin_phi = np.sin(self._phis)

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

        return mask * self._ld(phase)

    def add_spot(self, theta, phi, radius, contrast):
        @np.vectorize(signature="(),(),(),()->(),(),(),()")
        def foo(a, b, c, d):
            return a, b, c, d

        it = foo(theta, phi, radius, contrast)

        if it[0].shape == ():
            it = [[i] for i in it]

        for t, p, r, c in zip(*it):
            idxs = hp.query_disc(self.N, hp.ang2vec(t, p), r)
            self._m[idxs] = 1 - c

    def flux(self, phase=0):
        def _flux(phase):
            mask = self._get_mask(phase)
            return (self._m * mask).sum() / mask.sum()

        return np.vectorize(_flux)(phase)

    def m(self, phase=0):
        return self._m * self._get_mask(phase)

    def show(self, phase=0, grid=False, return_img=False, **kwargs):
        kwargs.setdefault("cmap", "magma")
        rotated_m = hp.Rotator(rot=[phase, 0], deg=False).rotate_map_pixel(self._m)
        projected_map = hp.orthview(
            rotated_m * self._ld(0), half_sky=True, return_projected_map=True
        )
        plt.close()
        if return_img:
            return projected_map
        else:
            plt.axis(False)
            plt.imshow(projected_map, **kwargs)
