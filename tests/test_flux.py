import healpy as hp
import jax
import numpy as np
import pytest

from spotter import Star
from spotter.light_curves import light_curve
from spotter.utils import ylm_to_hp


@pytest.mark.parametrize("deg", (3, 7))
@pytest.mark.parametrize("u", ([], [0.1, 0.4]))
def test_starry(deg, u):
    pytest.importorskip("jaxoplanet")
    from jaxoplanet.starry import Surface, Ylm
    from jaxoplanet.starry.light_curves import surface_light_curve

    y = np.array([1, *(1e-2 * np.random.randn((deg + 1) ** 2 - 1))])

    N = 20
    mh = ylm_to_hp(y, N)

    surface = Surface(y=Ylm.from_dense(y), u=u)
    star = Star(mh, period=2 * np.pi, inc=np.pi / 2, u=u)

    phases = np.linspace(0, 2 * np.pi, 100)
    expected = jax.vmap(lambda phi: surface_light_curve(surface, theta=phi))(phases)
    calc = light_curve(star, phases)[0]

    np.testing.assert_allclose(calc, expected, atol=1e-4)
