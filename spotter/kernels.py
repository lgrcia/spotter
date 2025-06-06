import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.special import roots_legendre
from tinygp import kernels

from spotter import Star, core


class GreatCircleDistance(kernels.stationary.Distance):
    """Compute the great-circle distance between two 3-vectors."""

    def distance(self, X1, X2):
        if jnp.shape(X1) != (3,) or jnp.shape(X2) != (3,):
            raise ValueError(
                "The great-circle distance is only defined for unit 3-vector"
            )
        return jnp.arctan2(jnp.linalg.norm(jnp.cross(X1, X2)), (X1.T @ X2))


class ActiveLatitude(kernels.Kernel):
    """A kernel describing the correlation between pixels values on a sphere whose
    intensity is modulated by the latitude of the pixel.
    """

    kernel: kernels.Kernel
    latitude: jax.Array
    sigma: jax.Array
    symetric: bool = True

    def amplitude(self, X):
        assert X.shape == (3,)
        cos_phi = X[2] / jnp.linalg.norm(X)
        a1 = norm.pdf(cos_phi, jnp.cos(jnp.pi / 2 - self.latitude), self.sigma)
        if self.symetric:
            a2 = norm.pdf(cos_phi, -jnp.cos(jnp.pi / 2 - self.latitude), self.sigma)
            return (a1 + a2) * 0.5
        else:
            return a1

    def evaluate(self, X1, X2):
        amp = self.amplitude(X1) * self.amplitude(X2)
        return amp * self.kernel.evaluate(X1, X2)


def norm_cov_series(mu, cov, N=10):
    """
    Series approximation to the normalized covariance.

    Stolen from starry_process

    """
    # Terms
    K = cov.shape[0]
    j = jnp.ones((K, 1))
    m = jnp.mean(cov)
    q = (cov @ j) / (K * m)
    z = m / mu**2

    # Coefficients
    fac = 1.0
    alpha = 0.0
    beta = 0.0
    for n in range(0, N + 1):
        alpha += fac
        beta += 2 * n * fac
        fac *= z * (2 * n + 3)

    # We're done
    return (alpha / mu**2) * cov + z * (
        (alpha + beta) * (j - q) @ (j - q).T - alpha * q @ q.T
    )


def probability_inc(inc):
    return jnp.sin(inc)


class FluxKernel(kernels.Kernel):
    surface_kernel: kernels.Kernel
    star: Star
    u: jax.Array = None
    inc: jax.Array = None
    order: int = 30
    normalize: bool = True
    _Ky: jax.Array = None

    def __post_init__(self):
        self._Ky = self.surface_kernel(self.star.x, self.star.x)

    def X(self, time, i):
        return core.design_matrix(
            self.star.N, inc=i, u=self.u, phase=self.star.phase(time)
        )

    def eval(self, X2, X1, inc):
        return self.X(X1, inc) @ self._Ky @ self.X(X2, inc).T

    def evaluate(self, X1, X2):
        # marginalize over inclination
        if self.inc is None:
            eval_over_incs = jax.vmap(self.eval, in_axes=(None, None, 0))
            roots, weights = roots_legendre(self.order)
            a = 0
            b = jnp.pi / 2
            t = (b - a) / 2 * roots + (a + b) / 2
            EI = ((b - a) / 2 * eval_over_incs(X1, X2, t) * probability_inc(t)).dot(
                weights
            )
            return EI
        else:
            return self.eval(X1, X2, self.inc)

    def __call__(self, X1, X2=None):
        if self.normalize:
            # in order to ensure normalization
            return norm_cov_series(1.0, super().__call__(X1, X2))
        else:
            return super().__call__(X1, X2)
