# spotter

<p align="center" style="margin-bottom:-50px">
    <img src="docs/_static/spotter.jpg" width="400">
</p>

<p align="center">
  Time-series spectra forward models of non-uniform stars
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/spotter">
      <img src="https://img.shields.io/badge/github-lgrcia/spotter-indianred.svg?style=flat" alt="github"/></a>
    <a href="LICENCE">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
  </p>
</p>

*spotter* is a Python package to produce time-series forward models of stars. It uses the [HEALPix](https://healpix.sourceforge.io/) subdivision scheme and is powered by the high-performance machine learning package [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), enabling its use on GPUs.

In its beta version, *spotter* is mainly developed to estimate transmission spectra stellar contamination from stellar rotational light curves.

## Installation

For now only locally with

```
pip install -e spotter
```

with *spotter* cloned using
```
git clone https://github.com/lgrcia/spotter
```
- 