import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from distributions import butterfly


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
    def __init__(self, u=None, N=64, b=None, r=None):
        self.N = N
        self.u = u
        self.n = hp.nside2npix(self.N)
        self._phis, self._thetas = hp.pix2ang(self.N, np.arange(self.n))
        self._sin_phi = np.sin(self._phis)

        self._spot_map = np.zeros(self.n)
        self._faculae_map = np.zeros(self.n)
        # Define transit chord if impact parameter (b) and planet radius (r) provided
        self._transit_chord_map = np.zeros(self.n)
        self.b = b
        self.r = r
        if b is not None and r is not None:
            self.define_transit_chord(b, r)

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

    def reset_coverage(self):
        self._spot_map = np.zeros(self.n)
        self._faculae_map = np.zeros(self.n)

    def define_transit_chord(self, b, r):
        self.b = b
        self.r = r
        theta1 = np.arccos(b + r)
        theta2 = np.arccos(b - r)
        idx = hp.query_strip(self.N, theta1, theta2)
        self._transit_chord_map[idx] = 1

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

    def show(self, phase=0, grid=False, return_img=False, transit_chord=True, **kwargs):
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
                return np.sum(self._spot_map >= vmin) / self.n
            else:
                mask = self._get_mask(phase)
                return np.sum(self._spot_map[mask] >= vmin) / mask.sum()
        elif transit_chord:
            in_chord = self._transit_chord_map
            is_spotted = self._spot_map >= vmin
            if phase is None:
                return np.logical_and(in_chord, is_spotted).sum() / in_chord.sum()
            else:
                mask = self._get_mask(phase)
                return (
                    np.logical_and(in_chord, is_spotted)[mask].sum()
                    / in_chord[mask].sum()
                )


def estimate_spot_coverage(
    star,
    ref_amp,
    contrast,
    spatial_dist="butterfly",
    latitudes=0.25,
    latitudes_sigma=0.08,
    radius_dist="uniform",
    radius_min=0.01,
    radius_max=0.1,
    spot_step_size=1,
    n_phases=100,
    n_iter=1,
    transit_chord=True,
):
    """
    Estimates the spot coverage on a stellar surface for a given light curve amplitude.

    Parameters
    ----------
    star : `Star`
        The stellar object on which spots are added.
    ref_amp : float
        The reference amplitude of the light curve to be achieved.
    contrast : float
        The contrast of the spots.
    spatial_dist : str, optional
        The spatial distribution of spots, only "butterfly" works for now.
    latitudes : float, optional
        Parameter for the butterfly distribution, controlling the latitudes of spots.
    latitudes_sigma : float, optional
        Parameter for the butterfly distribution, controlling the latitudinal spread of
        spots.
    radius_dist : str, optional
        The distribution of spot radii, either "uniform" or another custom distribution.
    radius_min : float, optional
        Minimum spot radius for uniform distribution.
    radius_max : float, optional
        Maximum spot radius for uniform distribution.
    spot_step_size : int, optional
        The step size for adjusting the number of spots in each iteration.
    n_phases : int, optional
        The number of phases for the light curve computation.
    n_iter : int, optional
        The number of iterations to estimate spot coverage.
    transit_chord : bool, optional
        If True, includes the covering fraction during a transit chord in the results.

    Returns
    -------
    If transit_chord is True:
        Tuple
            Tuple containing covering fraction during transit chords, covering fraction
            over the entire stellar disk, the number of spots, and corresponding light
            curve amplitudes.
    If transit_chord is False:
        Tuple
            Tuple containing covering fraction over the entire stellar disk, the number
            of spots, and corresponding light curve amplitudes.
    """
    phase = np.linspace(0, 2 * np.pi, n_phases)
    f_chords = []
    f_disks = []
    n_spots = []
    amps = []
    for i in range(n_iter):
        n_spot = 1
        amp = 0
        while amp < ref_amp:
            star.reset_coverage()

            # spots properties
            if spatial_dist == "butterfly":
                theta, phi = butterfly(latitudes, latitudes_sigma, n_spot)
            if radius_dist == "uniform":
                radii = np.random.uniform(radius_min, radius_max, n_spot)

            # add spots
            star.add_spot(theta, phi, radii, contrast)

            # compute light curve amplitude
            amp = star.flux(phase).ptp()
            if amp < ref_amp:
                n_spot += spot_step_size
            else:
                # Once we cross the threshold, we need to randomly do 1 of 2 things
                # 1. Accept this spot distribution (which overshoots the ref_amp)
                # 2. Remove the last added spot (thus undershooting the ref_amp)
                if np.random.choice([0, 1]):
                    star.reset_coverage()
                    n_spot -= spot_step_size

                    # spots properties
                    if spatial_dist == "butterfly":
                        theta, phi = butterfly(latitudes, latitudes_sigma, n_spot)
                    if radius_dist == "uniform":
                        radii = np.random.uniform(radius_min, radius_max, n_spot)

                    # add spots
                    star.add_spot(theta, phi, radii, contrast)

                    # compute light curve amplitude
                    amp = star.flux(phase).ptp()

                if transit_chord:
                    f_chords.append(star.covering_fraction(transit_chord=True))
                f_disks.append(star.covering_fraction())
                n_spots.append(n_spot)
                amps.append(amp)
                break

    if transit_chord:
        return f_chords, f_disks, n_spots, amps

    else:
        return f_disks, n_spots, amps
