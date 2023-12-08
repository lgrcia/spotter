import healpy as hp
import numpy as np
import pytest

from spotter import Star


@pytest.mark.parametrize("N", [2**n for n in range(1, 10)])
@pytest.mark.parametrize(
    "center", [(0.5, 0.0), (np.pi / 2, 1.0), (1.0, np.pi), (1.0, 1.0)]
)
@pytest.mark.parametrize("radius", [0.1, 0.5, 1.0, 2.0])
def test_query_idxs(N, center, radius):
    from spotter.star import query_idxs_function

    expected = hp.query_disc(N, hp.ang2vec(*center), radius)

    star = Star(N=N)
    computed = np.flatnonzero(
        query_idxs_function(star._thetas, star._phis)(*center, radius)
    )

    np.testing.assert_array_equal(computed, expected)
