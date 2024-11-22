from collections import defaultdict

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np


def ylm2healpix(y):
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


def sigmoid(x, scale=1000):
    return (jnp.tanh(x * scale / 2) + 1) / 2
