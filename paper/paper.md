---
title: '*spotter*: Hardware-Accelerated Forward Models of Pixelated Stars'
tags:
  - Python
  - astronomy
  - stars
  - exoplanets
  - time-series
authors:
  - name: Lionel Garcia
    orcid: 0000-0002-4296-2246
    affiliation: 1
  - name: Benjamin V. Rackham
    orcid: 0000-0002-3627-1676
    affiliation: 2
affiliations:
 - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY, USA
   index: 1
 - name: Department of Earth, Atmospheric and Planetary Science, Massachusetts Institute of Technology, MA, USA
   index: 2
date: 01 January 2025
bibliography: paper.bib
---

# Summary

The study of exoplanets predominantly relies on measurements of host stars' observables, such as flux and spectral time-series. However, stellar activity, including spots and faculae, has been increasingly recognized as a significant source of noise in signal analysis. For instance, photospheric active regions can mimic or hide exoplanetary signals [@rackham:2023]. These active regions also complicate exoplanet detection, whether through transit observations [@garcia:2024] or radial velocity measurements [@collier:2021]. Addressing these challenges requires modeling the stellar surface and its temporal evolution. Although various methods exist for representing stellar surfaces, such as @luger:2019, accurate inference of their properties necessitates forward models that are computationally efficient and compatible with widely used inference tools. Furthermore, the growing volume and resolution of datasets, combined with advancements in instrumentation, emphasize the need for scalable and high-performance models.

# Statement of Need

*spotter* is a Python package designed for fast forward modeling of non-uniform stellar surfaces. Built with JAX [@jax:2018], it leverages hardware acceleration to enable efficient simulations and inferences through a Pythonic interface. *spotter* can model flux and spectral time-series from pixelated stellar surfaces and define surface Gaussian processes conditioned on parameters such as spot size and contrast, inspired by @luger:2021. Its minimal and flexible design makes it a versatile tool for stellar and exoplanet science applications, including ensemble analyses of stellar light curves, akin to approaches similar to @luger:2021 and @morris:2020.

# Principle

If $\mathbf{y}$ represents a vector of pixel values corresponding to the surface intensity of a star, *spotter* models the observable $f$ (e.g., flux) as

$$f = \mathbf{X} \mathbf{y}$$

where, at any given time, $\mathbf{X}$ is constructed by accounting for:

- The pixels belonging to the visible hemisphere of the stellar surface,
- The projected area of each pixel, and
- The limb darkening intensity of each pixel.

[*spotter*'s documentation](https://spotter.readthedocs.io/en/latest/) demonstrates how this linear model is applied to simulate both flux and spectral time-series for rotating stellar surfaces. To facilitate this computation, the HEALPix subdivision scheme [@healpix:2005] is employed to decompose the stellar surface into pixels.

By adopting this formalism, *spotter* implements its model using simple linear algebra operations in JAX. It supports execution on hardware accelerators such as GPUs and TPUs, enabling fast and scalable simulations and inferences. This capability makes *spotter* well-suited for processing large datasets and high-resolution observations.

# Performance

## Speed

\autoref{perf} shows the performance of *spotter* in evaluating the flux forward model of a rotation surface with an increasing number of pixels, as well as for an increasing number of data points. This figure shows that running *spotter* on GPU provides two orders of magnitude speedups compared to running it on CPU.

![Flux forward model evaluation time on a single CPU and GPU. *Left*: evaluation time of a 1000-points rotation light curve versus the number of pixels used to represent the surface. *Right*: evaluation time of a 3072-pixels surface depending on the number of points in the time series. \label{perf}](figures/flux.png)

\autoref{perf2} shows the performance of *spotter* in evaluating the spectrum forward model of a star depending on the number of wavelength bins. In this case, using a GPU leads to an order of magnitude speedup compared to a CPU.

![Spectrum forward model evaluation time on a single CPU and GPU. *Left*: evaluation time of the spectrum of a surface with an increasing number of pixels (for 500 wavelength bins). *Right*: evaluation time of the spectrum of a 3072-pixels surface versus the number of wavelength bins. \label{perf2}](figures/spectrum.png)

On CPU, our benchmark is done on the single core of an Apple M2 max chip, while on GPU we use the single core of an NVIDIA Tesla V100 processor.

## Precision

We validate the precision of *spotter*'s flux forward model against models evaluated with *starry* [@luger:2019]. We make these precision benchmarks part of the unit tests of *spotter*.

# Acknowledgements

*spotter* makes use of the following dependencies: *numpy* [@numpy], *healpy* [@healpy], *jax* [@jax:2018], *equinox* [@equinox] and [*tinygp*](https://tinygp.readthedocs.io).


# References
