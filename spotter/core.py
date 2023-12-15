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


def doppler_shift_function(thetas, period, radius):
    period_s = period * 24 * 60 * 60
    omega = jnp.pi * 2 / period_s
    radius_m = radius * 695700000.0
    c = 299792458.0

    def doppler_shift(phase):
        radial_velocity = radius_m * omega * jnp.sin(thetas - phase)
        shift = radial_velocity / c
        return shift

    return doppler_shift


def shifted_spectra(spectra):
    n = jnp.shape(spectra)[1]
    spectra_fft = jnp.fft.fft(spectra)
    spectra_fft_shift = jnp.fft.fftshift(spectra_fft)
    u = jnp.arange(-n / 2, n / 2)

    def function(shift):
        spectra_fft_shift_ = spectra_fft_shift * jnp.exp(
            -1j * 2 * jnp.pi * shift * u[None, :] / n
        )
        spectra_fft_ = jnp.fft.ifftshift(spectra_fft_shift_, axes=1)
        return jnp.real(jnp.fft.ifft(spectra_fft_))

    return function


def integrated_spectrum(thetas, phis, period, radius, wv, spectra):
    mask_function = hemisphere_mask(thetas)
    shift_function = doppler_shift_function(thetas, period, radius)
    shifted_spectra_function = shifted_spectra(spectra)
    sinphi = jnp.sin(phis)
    dw = wv[1] - wv[0]

    def function(phase):
        mask = mask_function(phase)
        shift = shift_function(phase).T
        s = shift[:, None] * wv / dw
        spectra_shifted = shifted_spectra_function(s)
        angle = sinphi * jnp.cos(thetas - phase)
        weighted = spectra_shifted * angle[:, None]
        return 2 * jnp.sum(weighted * mask[:, None], 0) / jnp.sum(mask)

    return function


# For the record, this is not faster than the C healpy wrapper
# i.e. 100x slower than using hp.query_disc. But if we ever need
# full jax compatibility, this is the way to go.
def query_idxs_function(thetas, phis):
    @jax.jit
    def query_idxs(theta, phi, radius):
        # https://en.wikipedia.org/wiki/Great-circle_distance
        # Vincenty formula
        p1 = phis - jnp.pi / 2
        p2 = theta - jnp.pi / 2

        t1 = thetas
        t2 = phi
        dl = jnp.abs((t1 - t2))

        sp1 = jnp.sin(p1)
        sp2 = jnp.sin(p2)
        cp1 = jnp.cos(p1)
        cp2 = jnp.cos(p2)
        cdl = jnp.cos(dl)
        sdl = jnp.sin(dl)

        a = (cp2 * sdl) ** 2 + (cp1 * sp2 - sp1 * cp2 * cdl) ** 2
        b = sp1 * sp2 + cp1 * cp2 * cdl
        d = jnp.arctan2(jnp.sqrt(a), b)

        return jnp.array(d <= radius)

    return query_idxs
