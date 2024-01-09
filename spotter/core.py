import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def hemisphere_mask(thetas):
    def mask(phase):
        a = (phase + jnp.pi / 2) % (2 * jnp.pi)
        b = (phase - jnp.pi / 2) % (2 * jnp.pi)
        mask_1 = jnp.logical_and((thetas < a), (thetas > b))
        mask_2 = jnp.logical_or((thetas > b), (thetas < a))
        cond1 = a > phase % (2 * jnp.pi)
        cond2 = b < phase % (2 * jnp.pi)
        cond = cond1 * cond2
        return jnp.where(cond, mask_1, mask_2)

    return mask


def polynomial_limb_darkening(thetas, phis):
    def ld(u, phase):
        z = jnp.sin(phis) * jnp.cos(thetas - phase)
        terms = jnp.array([u * (1 - z) ** (n + 1) for n, u in enumerate(u)])
        return 1 - jnp.sum(terms, 0)

    return ld


def projected_area(thetas, phis):
    def area(phase):
        return jnp.cos(thetas - phase) * jnp.sin(phis)

    return area
