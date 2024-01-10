from collections import defaultdict

import healpy as hp
import numpy as np
import pytest

from spotter import Star
from spotter.utils import ylm2healpix


@pytest.mark.parametrize("deg", (3, 10))
@pytest.mark.parametrize("u", ([], [0.1, 0.4]))
def test_starry(deg, u):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # starry map with random coefficients
    np.random.seed(deg + len(u))
    y = np.random.randn((deg + 1) ** 2)
    y[0] = 1.0
    y[1:] *= 1e-2
    inc = np.pi / 2
    ms = starry.Map(ydeg=deg, udeg=len(u), inc=np.rad2deg(inc))
    if len(u) > 0:
        ms[1:] = u
    ms[:, :] = y

    # rotation to map healpix
    _m = starry._core.core.OpsYlm(deg, 0, 0, 1)
    ry = _m.dotR([y], 0.0, 1.0, 0.0, np.pi / 2)[0]
    ry = _m.dotR([ry], 1.0, 0.0, 0.0, -np.pi / 2)[0]
    ry = _m.dotR([ry], 0.0, 0.0, 1.0, np.pi)[0]

    # starry to healpix to spotter
    N = 2**7
    y2 = ylm2healpix(ry)
    mh = hp.alm2map(y2, nside=N)

    # mh with same ptp as ims
    ims = ms.render(projection="moll")
    mh = mh - np.nanmin(mh)
    mh = mh / np.nanmax(mh)
    mh = mh * (np.nanmax(ims) - np.nanmin(ims))
    mh = mh + np.nanmin(ims)

    star = Star(N=N, u=u)
    star.map_spot = 1 - mh

    # comparison
    phases = np.linspace(0, 2 * np.pi, 100)
    expected = np.array(ms.flux(theta=np.rad2deg(phases)))
    calc = star.flux(phases)

    np.testing.assert_allclose(calc, expected, atol=1e-4)
