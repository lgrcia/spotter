import healpy as hp
import jax
import numpy as np
import pytest

from spotter import Star
from spotter.light_curves import light_curve
from spotter.utils import ylm2healpix


@pytest.mark.parametrize("deg", (3, 7))
@pytest.mark.parametrize("u", ([], [0.1, 0.4]))
def test_starry(deg, u):
    pytest.importorskip("jaxoplanet")
    from jaxoplanet.starry import Surface, Ylm
    from jaxoplanet.starry.core import rotation
    from jaxoplanet.starry.light_curves import surface_light_curve

    y = np.array([1, *(1e-2 * np.random.randn((deg + 1) ** 2 - 1))])

    # rotation to map HEALPix
    ry = rotation.dot_rotation_matrix(deg, None, 1.0, None, np.pi / 2)(y)
    ry = rotation.dot_rotation_matrix(deg, 1.0, None, None, -np.pi / 2)(ry)
    ry = rotation.dot_rotation_matrix(deg, None, None, 1.0, np.pi)(ry)

    # starry to HEALPix to spotter
    N = 2**7
    y2 = ylm2healpix(ry)
    mh = hp.alm2map(y2, nside=N)

    surface = Surface(y=Ylm.from_dense(y), u=u)
    star = Star(mh, period=2 * np.pi, inc=np.pi / 2, u=u)

    phases = np.linspace(0, 2 * np.pi, 100)
    expected = jax.vmap(lambda phi: surface_light_curve(surface, theta=phi))(phases)
    calc = light_curve(star, phases)[0]

    np.testing.assert_allclose(calc / calc.max(), expected / expected.max(), atol=1e-4)
