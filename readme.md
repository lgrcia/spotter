> Work in progress, beta released

# spotter

<p align="center" style="margin-bottom:-50px">
    <img src="docs/_static/spotter.jpg" width="380">
</p>

<p align="center">
  Forward models of non-uniform stellar photospheres and their spectra
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/spotter">
      <img src="https://img.shields.io/badge/github-lgrcia/spotter-e3a8a1.svg?style=flat" alt="github"/></a>
    <a href="LICENCE">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
  </p>
</p>

*spotter* is a Python package to produce forward models of non-uniform stellar photospheres and their spectra. It uses the [HEALPix](https://healpix.sourceforge.io/) subdivision scheme and is powered by the high-performance numerical package [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), enabling its use on GPUs.

**Note**

In its beta version, *spotter* is mainly developed to estimate transmission spectra stellar contamination from stellar rotational light curves. Use at your own risk as the code is completely untested and its API subject to change.

## Features
- Small-scale surface features (e.g., beyond limitations of [starry]()) <span style="color:grey">- *in beta*</span>
- Modeling of active regions with unique angular dependence on brightness (e.g., limb-brightened faculae)
- GPU compatible <span style="color:grey">- *in beta*</span>
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
