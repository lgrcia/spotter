# spotter

Approximate forward models of fluxes and spectra time-series of non-uniform stars.

---

```{warning}
Use at your own risk as the code is completely untested and its API subject to change.
```

*spotter* uses the [HEALPix](https://healpix.sourceforge.io/) subdivision scheme and is powered by the high-performance numerical package [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), enabling its use on GPUs.


## Features

- Adjustable surface resolution <span style="color:grey">
- Small-scale surface features modeling (e.g. beyond limitations of [starry]()) <span style="color:grey">
- Modeling of any active regions with their limb laws (e.g. limb-brightened faculae)
- GPU compatible <span style="color:grey">
- Possibility to input any stellar spectra model

```{toctree}
:maxdepth: 1
:caption: Get started

notebooks/introduction
```

```{toctree}
:maxdepth: 1
:caption: Reference

notebooks/rotation.ipynb
api
```