import healpy as hp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def hemisphere_mask(theta, phase):
    theta = jnp.atleast_1d(theta)
    a = (phase + jnp.pi / 2) % (2 * jnp.pi)
    b = (phase - jnp.pi / 2) % (2 * jnp.pi)
    mask_1 = jnp.logical_and((theta < a), (theta > b))
    mask_2 = jnp.logical_or((theta > b), (theta < a))
    cond1 = a > phase % (2 * jnp.pi)
    cond2 = b < phase % (2 * jnp.pi)
    cond = cond1 * cond2
    return jnp.where(cond, mask_1, mask_2)


def polynomial_limb_darkening(theta, phi, u=None, phase=0.0):
    if u is None:
        return 1.0
    else:
        theta = jnp.atleast_1d(theta)
        phi = jnp.atleast_1d(phi)
        u = jnp.atleast_1d(u)
        z = jnp.sin(phi) * jnp.cos(theta - phase)
        terms = jnp.array([un * (1 - z) ** (n + 1) for n, un in enumerate(u)])
        return 1 - jnp.sum(terms, axis=theta.ndim - 1)


def projected_area(theta, phi, phase):
    return jnp.cos(theta - phase) * jnp.sin(phi)


def covering_fraction(x):
    return jnp.mean(x > 0)


def distance(thetas, phis):

    p1 = phis - jnp.pi / 2
    t1 = thetas
    sp1 = jnp.sin(p1)
    cp1 = jnp.cos(p1)

    def fun(theta0, phi0):
        # https://en.wikipedia.org/wiki/Great-circle_distance
        # Vincenty formula
        p2 = theta0 - jnp.pi / 2
        t2 = phi0
        dl = jnp.abs((t1 - t2))

        sp2 = jnp.sin(p2)
        cp2 = jnp.cos(p2)
        cdl = jnp.cos(dl)
        sdl = jnp.sin(dl)

        a = (cp2 * sdl) ** 2 + (cp1 * sp2 - sp1 * cp2 * cdl) ** 2
        b = sp1 * sp2 + cp1 * cp2 * cdl
        return jnp.arctan2(jnp.sqrt(a), b)

    return fun


def query_disk(thetas, phis):

    distance_fn = distance(thetas, phis)

    def fun(theta, phi, radius):
        d = distance_fn(theta, phi)
        return jnp.array(d <= radius, dtype=jnp.int8)

    return fun


def smooth_spot(thetas, phis):

    distance_fn = distance(thetas, phis)

    def fun(theta, phi, r, c):
        A = c * distance_fn(theta, phi) / (2 * r)
        C = c / 2
        return 0.5 * jnp.tanh(C - A) + 0.5 * jnp.tanh(C + A)

    return fun
