from dataclasses import dataclass

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from spotter import core

jax.config.update("jax_enable_x64", True)


def _wrap(*args):
    n = len(args)
    signature = ",".join(["()"] * n)
    signature = f"{signature}->{signature}"
    new_args = np.vectorize(lambda *args: args)(*args)
    if new_args[0].shape == ():
        return [np.array([n]) for n in new_args]
    else:
        return new_args


@dataclass
class Star:
    """
    A star object
    """

    u: list = None
    """List of limb darkening coefficients. Defaults to None."""
    N: int = 64
    """Star's HEALPix map nside parameter. Defaults to 64."""
    b: float = None
    """Impact parameter of the transit chord. Defaults to None."""
    r: float = None
    """Planet radius. Defaults to None."""
    map_spot: np.ndarray = None
    """The star's spot map contrast as a fraction of the quiescent photosphere. Defaults to None."""
    map_faculae: np.ndarray = None
    """The star's faculae map contrast as a fraction of the quiescent photosphere. Defaults to None."""
    spectrum: callable = None
    wv: np.ndarray = None
    veq: float = 0.0

    def __post_init__(self):
        if self.u is None:
            self.u = [0.0]

        # Maps
        # ----
        self.n = hp.nside2npix(self.N)
        self._phis, self._thetas = hp.pix2ang(self.N, np.arange(self.n))
        self._sin_phi = np.sin(self._phis)
        self.clear_surface()

        # Spectral
        # --------
        if self.wv is not None:
            dw = np.unique(np.diff(self.wv))
            # if len(dw) > 1:
            #     raise ValueError("wavelength must be evenly spaced")
            # else:
            self.dw = dw[0]
        else:
            self.dw = None

        if self.spectrum is None:
            self.spectrum = lambda x: np.ones_like(x)
            if self.wv is None:
                self.map_spectrum = np.ones(self.n)
                self.dw = None
            else:
                self.map_spectrum = np.ones((self.n, len(self.wv)))
                dw = np.unique(np.diff(self.wv))
                self.dw = dw[0]
        else:
            self.map_spectrum = (
                np.ones(self.n)[:, None] * self.spectrum(self.wv)[None, :]
            )

        if self.dw is not None:
            # TODO : this must be trivial now that we work with veq
            self.max_shift = (
                int(
                    np.ceil(
                        (
                            np.max(
                                core.doppler_shift_function(self._thetas, self.veq)(0.0)
                            )
                            * self.wv[-1]
                        )
                        / self.dw
                    )
                )
                * 2
            )

            self.extended_wv = np.linspace(
                self.wv[0] - self.max_shift * self.dw,
                self.wv[-1] + self.max_shift * self.dw,
                self.wv.size + 2 * self.max_shift,
            )
            self.extended_map_spectrum = (
                np.ones(self.n)[:, None] * self.spectrum(self.extended_wv)[None, :]
            )

        # JAX functions
        # -------------
        self.hemisphere_mask = jax.vmap(core.hemisphere_mask(self._thetas))
        self.polynomial_limb_darkening = jax.vmap(
            core.polynomial_limb_darkening(self._thetas, self._phis), in_axes=(None, 0)
        )
        self.projected_area = jax.vmap(core.projected_area(self._thetas, self._phis))

        # Transit chord
        # -------------
        self._map_chord = np.zeros(self.n)
        assert (self.b is None and self.r is None) or (
            self.b is not None and self.r is not None
        ), "Either both b and r must be provided or neither."
        if self.b is not None and self.r is not None:
            self.define_transit_chord(self.b, self.r)

    def clear_surface(self):
        """
        Clear the surface of the star by setting the spot and faculae maps to zero.
        """
        self.map_spot = np.zeros(self.n)
        self.map_faculae = np.zeros(self.n)

    @property
    def shape(self):
        """
        Shape of the star's HEALPix map.

        Returns
        -------
        tuple
            The shape of the star's HEALPix map.
        """
        return (self.n,)

    @property
    def has_chord(self):
        """
        Check if the star has a transit chord defined.

        Returns
        -------
        bool
            True if the star has a transit chord defined, False otherwise.
        """
        return self.r is not None

    @property
    def resolution(self):
        """
        Resolution of the star's HEALPix map.

        Returns
        -------
        float
            The resolution of the star's HEALPix map.
        """
        return hp.nside2resol(self.N)

    def add_spot(self, theta, phi, radius, contrast, spectrum=None):
        """
        Add spot(s) to the star's surface.

        Parameters
        ----------
        theta : float or list
            The polar angle(s) of the spot(s).
        phi : float or list
            The azimuthal angle(s) of the spot(s).
        radius : float or list
            The radius(es) of the spot(s).
        contrast : float or list or None
            The contrast(s) of the spot(s) as the fraction
            of the difference of intensity with the quiescent photosphere,
            i.e. :math:`I_{spot} = I_{quiescent} * (1 - contrast)`.
            Either contrast or spectrum must be provided.
        spectrum : float or list or None
            The spectrum(s) of the spot(s) normalized to 1. If specified,
            the spectrum is multiplied by :math:`I_{spot}` (see ``contrast``)

        Examples
        --------
        .. plot::
            :context:
            :nofigs:

            import matplotlib.pyplot as plt
            from spotter import Star
            star = Star(u=[0.1, 0.2], N=2**7)

        >>> from spotter import Star
        >>> star = Star(u=[0.1, 0.2], N=2**7)

        To add spot(s)

        >>> star.add_spot([1.5, 1.], [0.2, 0.5], [0.1, 0.3], 0.1)
        >>> star.show()

        .. plot::
            :context:

            star.clear_surface()
            star.add_spot([1.5, 1.], [0.2, 0.5], [0.1, 0.3], 0.1)
            star.show()
            plt.tight_layout()

        """
        if spectrum is not None:
            s = spectrum(self.extended_wv)
        for t, p, r, c in zip(*_wrap(theta, phi, radius, contrast)):
            idxs = hp.query_disc(self.N, hp.ang2vec(t, p), r)
            self.map_spot[idxs] = c

            if spectrum is not None:
                self.extended_map_spectrum[idxs] = s * c

    def add_faculae(self, theta, phi, radius_in, contrast, radius_out=None):
        """
        Add facula(e) to the star's surface.

        Parameters
        ----------
        theta : float or list
            The polar angle(s) of the faculae.
        phi : float or list
            The azimuthal angle(s) of the faculae.
        radius_in : float or list
            The inner radius(es) of the faculae.
        contrast : float or list
            The contrast(s) of the faculae as a fraction
            of the intensity of the quiescent photosphere,
            i.e. :math:`I_{faculae} = I_{quiescent} * contrast`.
        radius_out : float or list
            The outer radius(es) of the faculae. Defaults to None.

        Examples
        --------
        .. plot::
            :context:
            :include-source:

            from spotter import Star
            import numpy as np
            from spotter.distributions import butterfly

            # adding faculae
            np.random.seed(15)
            star = Star(u=[0.1, 0.2], N=2**7)
            lat, lon = butterfly(0.25, 0.08, 100)
            star.add_faculae(lat, lon, 0.02, 0.05)
            star.show()
            plt.tight_layout()

        And to create annulus faculae

        .. plot::
            :context: close-figs
            :include-source:

            star.clear_surface()
            star.add_faculae(lat, lon, 0.1, 0.1, radius_out=0.11)
            star.show()
            plt.tight_layout()

        """
        if radius_out is None:
            for t, p, r, c in zip(*_wrap(theta, phi, radius_in, contrast)):
                idxs = hp.query_disc(self.N, hp.ang2vec(t, p), r)
                self.map_faculae[idxs] = c
        else:
            for t, p, ri, ro, c in zip(
                *_wrap(theta, phi, radius_in, radius_out, contrast)
            ):
                inner_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ri)
                outer_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ro)
                idxs = np.setdiff1d(outer_idxs, inner_idxs)
                self.map_faculae[idxs] = c

    def add_spot_faculae(
        self, theta, phi, radius_in, radius_out, contrast_spot, contrast_faculae
    ):
        """
        Add both spot(s) and facula(e) to the star's surface.

        Parameters
        ----------
        theta : float or list
            The polar angle(s) of the spot(s) and faculae.
        phi : float or list
            The azimuthal angle(s) of the spot(s) and faculae.
        radius_in : float or list
            The inner radius(es) of the faculae.
        radius_out : float or list
            The outer radius(es) of the faculae.
        contrast_spot : float or list
            The contrast(s) of the spot(s).
        contrast_faculae : float or list
            The contrast(s) of the faculae.

        Examples
        --------
        If we create a stellar map

        .. plot::
            :context:
            :include-source:

            from spotter import Star
            import numpy as np
            from spotter.distributions import butterfly

            # adding spot and faculae
            np.random.seed(15)
            star = Star(u=[0.1, 0.2], N=2**7)
            lat, lon = butterfly(0.25, 0.08, 200)
            radii = np.random.uniform(0.05, 0.1, len(lat))
            star.add_spot_faculae(lat, lon, radii, radii + 0.02, 0.05, 0.03)
            star.show()
            plt.tight_layout()


        """
        for t, p, ri, ro, cs, cf in zip(
            *_wrap(theta, phi, radius_in, radius_out, contrast_spot, contrast_faculae)
        ):
            inner_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ri)
            outer_idxs = hp.query_disc(self.N, hp.ang2vec(t, p), ro)
            facuale_idxs = np.setdiff1d(outer_idxs, inner_idxs)
            self.map_faculae[facuale_idxs] = cf
            self.map_spot[inner_idxs] = cs

    def define_transit_chord(self, b, r):
        """
        Define the transit chord on the star's surface.

        Parameters
        ----------
        b : float
            Impact parameter of the transit chord.
        r : float
            Planet radius.
        """
        self.b = b
        self.r = r
        theta1 = np.arccos(b + r)
        theta2 = np.arccos(b - r)
        idx = hp.query_strip(self.N, theta1, theta2)
        self._map_chord[idx] = 1

    def jax_flux(self, phases):
        """
        Return a [JAX](https://jax.readthedocs.io/en/latest/) function to compute the star's flux.

        Parameters
        ----------
        phases : numpy.ndarray
            Array of phases at which to calculate the flux.

        Returns
        -------
        function
            A JAX function that calculates the flux of the star at the given phases.

        Examples
        --------

        If we create a stellar map with random spots

        .. plot::
            :context:
            :include-source:

            from spotter import Star
            import numpy as np
            import matplotlib.pyplot as plt
            from spotter.distributions import butterfly

            # adding spots
            np.random.seed(15)
            star = Star(u=[0.1, 0.2], N=2**6)
            lat, lon = butterfly(0.25, 0.08, 200)
            star.add_spot(lat, lon, 0.05, 0.1)
            star.show()

        we can compute the light curve of the star at a given phase with

        .. plot::
            :include-source:
            :context: close-figs

            phases = np.linspace(0, 4 * np.pi, 1000)
            flux = star.jax_flux(phases)
            y = flux(star.map_spot)
            plt.plot(phases, y)
            plt.tight_layout()

        Note the gain from using a pre-computed jax flux compared to the base ``flux`` method

        .. code-block:: python

            from time import time
            import jax

            t0 = time()
            y = star.flux(phases)
            time_base = time() - t0

            t0 = time()
            y = jax.block_until_ready(flux(star.map_spot))
            time_jax = time() - t0

            print(f"base: {time_base:.3f} s")
            print(f"jax: {time_jax:.3f} s")

        .. code-block:: none

            base: 1.115 s
            jax: 0.031 s


        """
        mask = self.hemisphere_mask(phases)
        limb_darkening = self.polynomial_limb_darkening(self.u, phases)
        projected_area = self.projected_area(phases)

        @jax.jit
        def flux(spot_map):
            _spot = (1 - spot_map) * limb_darkening
            _geometry = mask * projected_area
            return (
                np.pi * (_spot * _geometry).sum(1) / (_geometry * limb_darkening).sum(1)
            )

        return flux

    def jax_amplitude(self, resolution=3):
        """
        Return a [JAX](https://jax.readthedocs.io) function to compute the star's peak to peak amplitude.

        Parameters
        ----------
        resolution : int, optional
            The resolution parameter for the flux calculation. Defaults to 3.

        Returns
        -------
        function
            A JAX function that calculates the amplitude of the star's peak to peak amplitude.

        Examples
        --------

        If we create a stellar map with random spots

        .. plot::
            :context:
            :include-source:

            from spotter import Star
            import numpy as np
            import matplotlib.pyplot as plt
            from spotter.distributions import butterfly

            # adding spots
            np.random.seed(15)
            star = Star(u=[0.1, 0.2], N=2**6)
            lat, lon = butterfly(0.25, 0.08, 200)
            star.add_spot(lat, lon, 0.05, 0.1)
            star.show()

        We can compute the amplitude of the star at a given phase with

        .. plot::
            :include-source:
            :context: close-figs

            amplitude = star.jax_amplitude(resolution=3)
            a = amplitude(star.map_spot)
            print(f"Amplitude: {a:.3e}")

        .. code-block:: none

            Amplitude: 1.279e-03

        Note the gain from using a pre-computed jax amplitude compared to the base ``amplitude`` method

        .. code-block:: python

            from time import time
            import jax

            phase = np.arange(0, 2 * np.pi, star.resolution)
            t0 = time()
            a = star.flux(phases).ptp()  # assuming this method exists
            time_base = time() - t0

            t0 = time()
            a = jax.block_until_ready(amplitude(star.map_spot))
            time_jax = time() - t0

            print(f"base: {time_base:.3f} s")
            print(f"jax: {time_jax:.3f} s")

        .. code-block:: none

            base: 1.210 s
            jax: 0.004 s
        """
        hp_resolution = hp.nside2resol(self.N) * resolution
        phases = np.arange(0, 2 * np.pi, hp_resolution)
        flux = self.jax_flux(phases)

        @jax.jit
        def amplitude(spot_map):
            f = flux(spot_map)
            return jnp.ptp(f)

        return amplitude

    def jax_spectrum(self):
        _integrated_spectrum = core.integrated_spectrum(
            self._thetas,
            self._phis,
            self.period,
            self.radius,
            self.extended_wv,
            self.extended_map_spectrum,
        )
        max_shift = self.max_shift

        def function(phase):
            spectrum = _integrated_spectrum(phase)
            return spectrum[max_shift:-max_shift]

        return function

    def flux(self, phases):
        """
        Calculate the flux of the star at given phases.

        Parameters
        ----------
        phases : numpy.ndarray
            Array of phases at which to calculate the flux.

        Returns
        -------
        numpy.ndarray
            The flux of the star at the given phases.
        """
        mask = np.vectorize(core.hemisphere_mask(self._thetas), signature="()->(n)")(
            phases
        )
        projected_area = np.vectorize(
            core.projected_area(self._thetas, self._phis), signature="()->(n)"
        )(phases)
        limb_darkening = (
            np.vectorize(
                core.polynomial_limb_darkening(self._thetas, self._phis),
                signature="()->(n)",
                excluded={0},
            )(self.u, phases)
            if len(self.u) > 0
            else 1
        )
        _spot = (1 - self.map_spot) * limb_darkening
        _geometry = mask * projected_area
        # faculae contribution, with same ld for now (TODO)
        _faculae = 0  # self.map_faculae * limb_darkening

        return (
            np.pi
            * ((_spot + _faculae) * _geometry).sum(1)
            / (_geometry * limb_darkening).sum(1)
        )

    def map(self, phase=None, limb_darkening=False):
        """
        Return the pixel elements values of the map.

        Parameters
        ----------
        phase : float, optional
            The rotation phase of the star. Defaults to 0.

        Returns
        -------
        numpy.ndarray
            Pixel elements values of the map.
        """
        if phase is None:
            mask = 1
        else:
            mask = self.hemisphere_mask(np.array([phase]))[0].__array__()

        if limb_darkening and phase is not None:
            spot_limb_darkening = self.polynomial_limb_darkening(
                self.u, np.array([phase])
            )[0].__array__()
        else:
            spot_limb_darkening = 1

        faculae_limb_brightening = 1
        m = (1 - self.map_spot) * mask * spot_limb_darkening
        spots = self.map_spot == 0.0
        if np.any(spots):
            m[spots] = m[spots] + (self.map_faculae * faculae_limb_brightening)[spots]
        return m

    def show(
        self,
        phase: float = 0,
        grid: bool = False,
        return_img: bool = False,
        chord: float = None,
        ax=None,
        **kwargs,
    ):
        """
        Show the stellar disk at a given rotation phase.

        Parameters
        ----------
        phase : float, optional
            The rotation phase of the stellar disk. Defaults to 0.
        grid : bool, optional
            Whether to display a grid on the plot. Defaults to False.
        return_img : bool, optional
            Whether to return the projected map as an image. Defaults to False.
        chord : float, optional
            An additional contrast applied on the map to visualize the
            position of the transit chord. Defaults to `None`.

        Returns
        -------
        numpy.ndarray or None
            If `return_img` is True, returns the projected map as a numpy array.
            Otherwise, returns None.

        Examples
        --------
        To show the stellar disk

        >>> from spotter import Star
        >>> star = Star(u=[0.1, 0.2], N=2**7, b=-0.7, r=0.06)
        >>> star.show()

        .. plot::
            :context:

            import matplotlib.pyplot as plt
            from spotter import Star
            star = Star(u=[0.1, 0.2], N=2**7, b=-0.7, r=0.06)
            star.show()
            plt.show()

        To visualize the transit chord

        >>> star.show(chord=0.1)

        .. plot::
            :context:

            star.show(chord=0.1)
            plt.show()

        """
        kwargs.setdefault("cmap", "magma")
        kwargs.setdefault("origin", "lower")
        ax = ax or plt.gca()

        # both spot and faculae with same ld for now (TODO)
        if (
            self.map_spot.max() == 0.0
            and self.map_faculae.max() == 0.0
            and self.u == [0.0]
        ):
            rotated_m = self.map()
        else:
            rotated_m = hp.Rotator(rot=[phase, 0], deg=False).rotate_map_pixel(
                self.map()
            )
        if self.has_chord and (chord is not None):
            assert isinstance(chord, float), "chord must be a float (or None)"
            mask = self._map_chord > 0
            rotated_m[mask] = rotated_m[mask] * (1 - chord)

        projected_map = hp.orthview(
            rotated_m * self.polynomial_limb_darkening(self.u, np.array([0]))[0],
            half_sky=True,
            return_projected_map=True,
        )
        plt.close()
        if return_img:
            return projected_map
        else:
            ax.axis(False)
            ax.imshow(projected_map, **kwargs)

    def covering_fraction(
        self, phase: float = None, vmin: float = 0.01, chord=False, disk=False
    ):
        """Return the covering fraction of active regions

        Either computed for the whole star (`phase=None`) or for the stellar
        disk given a phase

        Parameters
        ----------
        phase : float, optional
            stellar rotation phase, by default None
        vmin : float, optional
            minimum contrast value for spots, by default 0.01
        vmax : float, optional
            minimum contrast value for faculae, by default 1.0
        transit_chord : bool, optional
            calculate the covering fraction within the transit chord

        Returns
        -------
        float
            full star or disk covering fraction

        Examples
        --------
        >>> star = Star(u=[0.1, 0.2], N=2**7, b=-0.7, r=0.06)
        >>> star.covering_fraction()
        0.0
        """
        if not chord:
            if phase is None:
                return np.sum(self.map_spot >= vmin) / self.n
            else:
                mask = self._get_mask(phase)
                return np.sum(self.map_spot[mask] >= vmin) / mask.sum()

        elif chord:
            in_chord = self._map_chord
            is_spotted = self.map_spot >= vmin
            if phase is None:
                return np.logical_and(in_chord, is_spotted).sum() / in_chord.sum()
            else:
                mask = self._get_mask(phase)
                return (
                    np.logical_and(in_chord, is_spotted)[mask].sum()
                    / in_chord[mask].sum()
                )
