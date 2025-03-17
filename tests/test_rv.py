import healpy as hp
import jax
import numpy as np
import pytest

from spotter import Star
from spotter.doppler import radial_velocity
from spotter.utils import ylm_to_hp


@pytest.mark.parametrize("deg", (3, 7))
@pytest.mark.parametrize("u", ([], [0.1, 0.4]))
def test_jaxoplanet(deg, u):
    pytest.importorskip("jaxoplanet")
    from jaxoplanet.starry import Surface, Ylm
    from jaxoplanet.starry.doppler import surface_radial_velocity

    y = np.array([1, *(1e-2 * np.random.randn((deg + 1) ** 2 - 1))])

    N = 20
    mh = ylm_to_hp(y, N)

    surface = Surface(y=Ylm.from_dense(y), u=u, period=1.0)
    star = Star(mh, period=1.0, inc=np.pi / 2, u=u)

    phases = np.linspace(0, 1, 100)
    expected = jax.vmap(
        lambda phase: surface_radial_velocity(surface, theta=phase * 2 * np.pi)
    )(phases)

    # import astropy.units as u
    # (1*u.m/u.s).to("R_sun/day").value
    conversion = 0.00012419146183699872
    calc = -radial_velocity(star, phases)[0] * conversion
    np.testing.assert_allclose(calc, expected, atol=1e-4)
