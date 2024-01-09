import pytest
import numpy as np
from spotter import Star
from collections import defaultdict
import healpy as hp


@pytest.mark.parametrize("deg", (3, 10))
@pytest.mark.parametrize("u", ([], [0.1, 0.4]))
def test_starry(deg, u):
    pytest.importorskip("starry")

    starry.config.lazy = False
    starry.config.quiet = True

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

    def starry2healpy(y):
        # Edmonds to Condon-Shortley phase convention
        lmax = int(np.floor(np.sqrt(len(y))))

        _hy = defaultdict(lambda: 0 + 0j)

        i = 0

        for l in range(0, lmax):
            for m in range(-l, l + 1):
                j = hp.sphtfunc.Alm.getidx(lmax, l, np.abs(m))
                if m < 0:
                    _hy[j] += 1j * y[i] / (np.sqrt(2) * (-1) ** m)
                elif m == 0:
                    _hy[j] += y[i]
                else:
                    _hy[j] += y[i] / (np.sqrt(2) * (-1) ** m)
                i += 1

        hn = hp.sphtfunc.Alm.getsize(lmax)
        hy = np.zeros(hn, dtype=np.complex128)
        for i in range(hn):
            hy[i] = _hy[i]

        return hy

    # starry to healpix to spotter
    N = 2**7
    y2 = starry2healpy(ry)
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
    expected = ms.flux(theta=np.rad2deg(phases))
    calc = np.pi * star.flux(phases)

    np.testing.assert_allclose(calc, expected, atol=1e-4)
