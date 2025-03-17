import healpy as hp
import jax
import numpy as np
import pytest

from spotter import Star
from spotter.light_curves import light_curve, transit_light_curve
from spotter.utils import ylm_to_hp
from spotter import core


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


def test_dipole_occultation():
    star = Star.from_sides(2**4, period=1.0)
    star = star.set(y=1.0 - core.spot(star.sides, 0.0, -np.pi / 2, np.pi / 2))

    np.allclose(
        transit_light_curve(star, z=10, y=-0.5, r=0.3),
        transit_light_curve(star, z=10, y=0.5, r=0.3),
    )

    np.testing.assert_allclose(
        transit_light_curve(star, z=10, x=-0.5, r=0.25), 1.0, 1e-4
    )
    assert transit_light_curve(star, z=10, x=0.5, r=0.25) < 1.0


def test_spot_occulation():
    pytest.importorskip("jaxoplanet")
    from jaxoplanet.starry import Surface, Ylm
    from jaxoplanet.starry.light_curves import surface_light_curve

    y = np.array([1, *(1e-1 * np.random.randn((5 + 1) ** 2 - 1))])
    u = (0,)

    N = 60
    mh = ylm_to_hp(y, N)

    surface = Surface(y=Ylm.from_dense(y), u=u)
    star = Star(mh, period=2 * np.pi, inc=np.pi / 2, u=u)

    x = np.linspace(-2, 2, 300)
    expected = jax.vmap(lambda x: surface_light_curve(surface, x=x, z=10, r=0.5))(x)
    calc = transit_light_curve(star, z=10, x=x, r=0.5)[0]

    np.testing.assert_allclose(calc, expected, atol=1e-3)
