from functools import partial

import numpy as np

from spotter.distributions import butterfly
from spotter.star import Star

default_latitudes = partial(butterfly, latitudes=0.25, latitudes_sigma=0.08)
default_distribution = partial(np.random.uniform, 0.01, 0.1)


def estimate_spot_coverage(
    star: Star,
    amplitude: float,
    contrast: float,
    draw_theta_phi: callable = default_latitudes,
    draw_radii: callable = default_distribution,
    resolution: float = 10.0,
    spot_step_size: int = 1,
    n_iter: int = 1,
    transit_chord: bool = True,
):
    """
    Estimates the spot coverage on a stellar surface for a given light curve amplitude.

    Parameters
    ----------
    star : `Star`
        The stellar object on which spots are added.
    amplitude: float
        The reference amplitude of the light curve to be achieved.
    contrast : float
        The contrast of the spots.
    draw_theta_phi : callable, optional
        A single `n_spot` argument function that draws the spot latitudes and longitudes. By default,
        a butterfly distribution with `latitudes=0.25` and `latitudes_sigma=0.08`.
    draw_radii : callable, optional
        A single `n_spot` argument function that draws the spot radii. By default, a
        uniform distribution between 0.01 and 0.1.
    resolution : float, optional
        The resolution of the light curve to compute the amplitude. See `Star.jax_amplitude`.
        By default 10.0.
    spot_step_size : int, optional
        The step size for adjusting the number of spots in each iteration.
    n_iter : int, optional
        The number of iterations to estimate spot coverage.
    transit_chord : bool, optional
        If True, includes the covering fraction during a transit chord in the results.

    Returns
    -------
    Tuple
        f_disks, n_spots, amps, f_chords

        where:
        - f_disks: list of covering fractions over the entire stellar disk
        - n_spots: list of number of spots
        - amps: list of corresponding light curve amplitudes
        - f_chords: list of covering fractions over the transit chord (None if `transit_chord=False`)
    """
    f_chords = [] if transit_chord else None
    f_disks = []
    n_spots = []
    amps = []

    _amplitude = star.jax_amplitude(resolution=resolution)

    def amplitude_of(n_spot):
        star.clear_surface()

        # spots properties
        theta, phi = draw_theta_phi(n=n_spot)
        radii = draw_radii(n=n_spot)
        # add spots
        star.add_spot(theta, phi, radii, contrast)
        # compute light curve amplitude
        amp = _amplitude(star.map_spot)

        return amp

    for _ in range(n_iter):
        n_spot = 1
        amp = 0
        while amp < amplitude:
            amp = amplitude_of(n_spot)

            if amp < amplitude:
                n_spot += spot_step_size
            else:
                # Once we cross the threshold, we need to randomly do 1 of 2 things
                # 1. Accept this spot distribution (which overshoots the ref_amp)
                # 2. Remove the last added spot (thus undershooting the ref_amp)
                if np.random.choice([0, 1]):
                    n_spot -= spot_step_size
                    amp = amplitude_of(n_spot)

                if transit_chord:
                    f_chords.append(star.covering_fraction(transit_chord=True))
                f_disks.append(star.covering_fraction())
                n_spots.append(n_spot)
                amps.append(amp)
                break

    return f_disks, n_spots, amps, f_chords
