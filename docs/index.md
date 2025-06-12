# spotter

*Approximate forward models of fluxes and spectra time-series of non-uniform stars.*

---

*spotter* serves as a flexible, low-level backend for scientific applications that require modeling stellar surfaces using spherical pixelation. While it is not a comprehensive simulation suite like [StarSim](https://ui.adsabs.harvard.edu/abs/2016A&A...586A.131H/abstract) or [SOAP](https://ui.adsabs.harvard.edu/abs/2014ApJ...796..132D/abstract), *spotter* provides core building blocks for custom workflows, including:
- Modeling [surface features](notebooks/introduction.ipynb) (e.g., stellar spots or faculae) with arbitrary shapes
- Forward modeling of stellar surface [flux](notebooks/multiband.ipynb), [spectra](notebooks/spectral.ipynb), [radial velocities](notebooks/rv.ipynb), and CCFs
- Simulation of [observables during transits](notebooks/spot_crossing.ipynb) by stellar or planetary companions
- Gaussian process frameworks for representing [stellar surfaces](notebooks/surface_gp.ipynb) and [modeling their flux time series](notebooks/flux_gp.ipynb)

*spotter* uses the [HEALPix](https://healpix.sourceforge.io/) subdivision scheme and is powered by the high-performance numerical package [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), enabling its use on multiple CPUs and accelerators like GPUs and TPUs.


```{toctree}
:maxdepth: 1
:caption: Get started

installation
notebooks/introduction
notebooks/precision.ipynb
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

notebooks/surface_gp
notebooks/flux_gp
notebooks/spectral
notebooks/multiband
notebooks/spot_crossing
notebooks/rv


```

```{toctree}
:maxdepth: 1
:caption: Reference

notebooks/principle.ipynb

```