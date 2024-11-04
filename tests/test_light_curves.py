import pytest
import numpy as np
from spotter import Star, light_curves, show, core
from jax._src.public_test_util import tolerance
import jax.numpy as jnp

N = core.npix(2)


def test_dark_hemisphere():
    star = Star.from_sides(2**4, period=1.0)
    star = star.set(y=1.0 - core.spot(star.sides, 0.0, 0.0, np.pi / 2))
    time = np.array([0.0, 0.5])
    result = light_curves.light_curve(star, time)
    expected = np.array([0.0, 1.0])
    assert jnp.allclose(expected, result)


@pytest.mark.parametrize(
    "y_u_shape",
    (
        (np.ones(N), None, (1, 3)),
        (np.ones((2, N)), None, (2, 3)),
        (np.ones((2, N)), ((0.1,)), (2, 3)),
        (np.ones(N), ((0.1,), (0.2,)), (2, 3)),
        (np.ones((2, N)), ((0.1, 0.4), (0.2, 0.3)), (2, 3)),
        (np.ones(N), ((0.1, 0.4), (0.2, 0.3)), (2, 3)),
    ),
)
def test_light_curve(y_u_shape):
    y, u, expected_shape = y_u_shape
    star = Star(y, u=u, period=1.0)
    f = light_curves.light_curve(star, np.array([0.0, 0.5, 0.3]))
    u = () if star.u is None else star.u
    assert f.shape == expected_shape
