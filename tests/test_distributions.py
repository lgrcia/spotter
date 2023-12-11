import numpy as np

from spotter import distributions


def test_butterfly():
    distributions.butterfly()

    calc = np.array(distributions.butterfly(n=20))
    assert calc.shape == (2, 20)


def test_uniform():
    distributions.uniform()

    calc = np.array(distributions.uniform(n=20))
    assert calc.shape == (2, 20)
