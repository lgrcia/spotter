# spotter

<p align="center">
    <img src="docs/source/_static/spotter.png" width="270">
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

*spotter* uses the [HEALPix](https://healpix.sourceforge.io/) subdivision scheme and is powered by the high-performance numerical package [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), enabling its use on GPUs.


## Features

- Small-scale surface features (e.g. beyond limitations of [starry]()) <span style="color:grey">
- Modeling of any active regions with their limb laws (e.g. limb-brightened faculae)
- GPU compatible <span style="color:grey">
- Possibility to input any stellar spectra model

## Installation

For now only locally with

```
pip install -e spotter
```

with *spotter* cloned using
```
git clone https://github.com/lgrcia/spotter
```
