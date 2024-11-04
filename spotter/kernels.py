import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from tinygp import kernels


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
