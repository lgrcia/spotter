from typing import Any

import healpy as hp
import numpy as np

from spotter import core

Array = Any


def show_map(
    x,
    u=None,
    phase: float = 0,
    return_img: bool = False,
    chord: float = None,
    ax=None,
    **kwargs,
):
    """
    Show the stellar disk at a given rotation phase.

    Parameters
    ----------
    phase : float, optional
        The rotation phase of the stellar disk. Defaults to 0.
    grid : bool, optional
        Whether to display a grid on the plot. Defaults to False.
    return_img : bool, optional
        Whether to return the projected map as an image. Defaults to False.
    chord : float, optional
        An additional contrast applied on the map to visualize the
        position of the transit chord. Defaults to `None`.

    Returns
    -------
    numpy.ndarray or None
        If `return_img` is True, returns the projected map as a numpy array.
        Otherwise, returns None.

    Examples
    --------
    To show the stellar disk

    >>> from spotter import Star
    >>> star = Star(u=[0.1, 0.2], N=2**7, b=-0.7, r=0.06)
    >>> star.show()

    .. plot::
        :context:

        import matplotlib.pyplot as plt
        from spotter import Star
        star = Star(u=[0.1, 0.2], N=2**7, b=-0.7, r=0.06)
        star.show()
        plt.show()

    To visualize the transit chord

    >>> star.show(chord=0.1)

    .. plot::
        :context:

        star.show(chord=0.1)
        plt.show()

    """
    import matplotlib.pyplot as plt

    if u is None:
        u = ()

    kwargs.setdefault("cmap", "magma")
    kwargs.setdefault("origin", "lower")
    ax = ax or plt.gca()

    limb_darkening = core.polynomial_limb_darkening(self.phis, self.thetas, u, phase)
    limbed = x * limb_darkening * mask
    rotated = hp.Rotator(rot=[phase, 0], deg=False).rotate_map_pixel(limbed)

    projected_map = hp.orthview(
        rotated * self.polynomial_limb_darkening(self.u, np.array([0]))[0],
        half_sky=True,
        return_projected_map=True,
    )
    plt.close()
    if return_img:
        return projected_map
    else:
        ax.axis(False)
        ax.imshow(projected_map, **kwargs)
