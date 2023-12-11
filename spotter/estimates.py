import numpy as np

from spotter.distributions import butterfly


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
