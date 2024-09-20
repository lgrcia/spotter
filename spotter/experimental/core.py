import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np


def distance(X, x):
    return jnp.arctan2(
        jnp.linalg.norm(jnp.cross(X.T, x[:, None], axis=0), axis=0), X @ x
    )


def inclination_convention(theta):
    return theta


def equator_coords(phi, i):
    x = jnp.cos(inclination_convention(i)) * jnp.cos(phi)
    y = jnp.cos(inclination_convention(i)) * jnp.sin(phi)
    z = jnp.sin(inclination_convention(i)) * jnp.ones_like(phi)

    return jnp.array([x, y, z])


def mask_projected_limb(X, phase, inclination=None, u=None):
    if inclination is None:
        inclination = jnp.pi / 2
    d = distance(X, equator_coords(phase, inclination))
    mask = d < jnp.pi / 2
    z = jnp.cos(d)
    projected_area = z
    if u is not None:
        u = jnp.atleast_1d(u)
        terms = jnp.array([un * (1 - z) ** (n + 1) for n, un in enumerate(u)])
        limb_darkening = 1 - jnp.sum(terms, axis=z.ndim - 1)
    else:
        limb_darkening = jnp.ones_like(d)

    return mask, projected_area, limb_darkening


def N_or_Y_to_N_n(N_or_y):
    if isinstance(N_or_y, int):
        n = hp.nside2npix(N_or_y)
        N = N_or_y
    else:
        N = hp.npix2nside(N_or_y.size)
        n = N_or_y.size
    return N, n


def vec(N_or_y):
    N, n = N_or_Y_to_N_n(N_or_y)
    return jnp.array(hp.pix2vec(N, range(n))).T


def design_matrix(N_or_y, inclination, u, phase):
    X = vec(N_or_y)
    mask, projected_area, limb_darkening = mask_projected_limb(X, phase, inclination, u)
    geometry = mask * projected_area
    return jnp.pi * limb_darkening * geometry / (geometry * limb_darkening).sum()


def flux(y, inclination, u, phase):
    return design_matrix(y, inclination, u, phase) @ y


def spherical_to_cartesian(theta, phi):
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.array([x, y, z])


def spots(N, latitude, longitude, radius):
    X = vec(N)
    d = distance(X, spherical_to_cartesian(jnp.pi / 2 - latitude, longitude))
    return d < radius


def render(y, inclination, u, phase):
    import matplotlib.pyplot as plt

    X = vec(y)

    limb_darkening = mask_projected_limb(X, phase, inclination, u)[2]
    rotated = hp.Rotator(
        rot=[phase, inclination_convention(inclination)], deg=False
    ).rotate_map_pixel(y * limb_darkening)

    projected_map = hp.orthview(rotated, half_sky=True, return_projected_map=True)
    plt.close()

    return projected_map


def amplitude(N_or_y, inclination, u, undersampling: int = 3) -> callable:

    N, _ = N_or_Y_to_N_n(N_or_y)
    resolution = hp.nside2resol(N)
    X = vec(N)
    hp_resolution = resolution * undersampling
    phases = jnp.arange(0, 2 * jnp.pi, hp_resolution)

    mask, projected_area, limb_darkening = jax.jit(
        jax.vmap(jax.jit(mask_projected_limb), in_axes=(None, 0, None, None))
    )(X, phases, inclination, u)

    geometry = mask * projected_area
    norm = (geometry * limb_darkening).sum(1)

    def fun(x):
        fluxes = (
            jnp.pi
            * jnp.einsum("ij,kj->ik", jnp.atleast_2d(x), limb_darkening * geometry)
            / norm
        )
        return jnp.ptp(fluxes, 1)

    return fun


def transit_chord(N, r: float, b: float = 0.0):
    """
    Returns the map of a transit chord.

    Parameters
    ----------
    b : float
        Impact parameter of the transit chord.
    r : float
        Planet radius.
    """
    N, n = N_or_Y_to_N_n(N)
    x = np.zeros(n, dtype=np.int8)
    theta1 = np.arccos(b + r)
    theta2 = np.arccos(b - r)
    x[hp.query_strip(N, theta1, theta2)] = 1.0
    return x
