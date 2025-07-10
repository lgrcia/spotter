# spotter

<p align="center">
    <img src="https://spotter.readthedocs.io/en/latest/_static/spotter.png" width="270">
</p>

<p align="center">
Approximate forward models of fluxes and spectra time-series of non-uniform stars
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/spotter">
      <img src="https://img.shields.io/badge/github-lgrcia/spotter-white.svg?style=flat" alt="github"/></a>
    <a href="LICENCE">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
  </p>
</p>

*spotter* serves as a flexible, low-level backend for scientific applications that require modeling stellar surfaces using spherical pixelation. While it is not a comprehensive simulation suite like [StarSim](https://ui.adsabs.harvard.edu/abs/2016A&A...586A.131H/abstract) or [SOAP](https://ui.adsabs.harvard.edu/abs/2014ApJ...796..132D/abstract), *spotter* provides core building blocks for custom workflows, including:
- Modeling [surface features](https://spotter.readthedocs.io/en/latest/notebooks/introduction/) (e.g., stellar spots or faculae) with arbitrary shapes
- Forward modeling of stellar surface [flux](https://spotter.readthedocs.io/en/latest/notebooks/multiband/), [spectra](https://spotter.readthedocs.io/en/latest/notebooks/spectral/), [radial velocities](https://spotter.readthedocs.io/en/latest/notebooks/rv/), and CCFs
- Simulation of [observables during transits](https://spotter.readthedocs.io/en/latest/notebooks/spot_crossing/) by stellar or planetary companions
- Gaussian process frameworks for representing [stellar surfaces](https://spotter.readthedocs.io/en/latest/notebooks/surface_gp/) and [modeling their flux time series](https://spotter.readthedocs.io/en/latest/notebooks/flux_gp/)

*spotter* uses the [HEALPix](https://healpix.sourceforge.io/) subdivision scheme and is powered by the high-performance numerical package [JAX](https://jax.readthedocs.io/en/latest/https://spotter.readthedocs.io/en/latest/notebooks/quickstart.html), enabling its use on multiple CPUs and accelerators like GPUs and TPUs.

Documentation can be found at [spotter.readthedocs.io](https://spotter.readthedocs.io)

## Installation

To install *spotter* from pypi
    
```bash
pip install spotter
```

To get the latest version under development, clone the repository and install the package using pip:

```bash
git clone https://github.com/lgrcia/spotter
pip install -e spotter
```