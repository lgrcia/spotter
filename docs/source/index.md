# spotter

```{image} _static/spotter.jpg
:width: 400px
:align: center
```

*spotter* is a Python package to produce forward models of non-uniform stars spectra. It uses the [HEALPix](https://healpix.sourceforge.io/) subdivision scheme and is powered by the high-performance numerical package [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), enabling its use on GPUs.

**Note**

In its beta version, *spotter* is mainly developed to estimate transmission spectra stellar contamination from stellar rotational light curves.

## Features

- Adjustable surface resolution <span style="color:grey">- *in beta*</span>
- Small-scale surface features modeling (e.g. beyond limitations of [starry]()) <span style="color:grey">- *in beta*</span>
- Modeling of any active regions with their limb laws (e.g. limb-brightened faculae)
- GPU compatible <span style="color:grey">- *in beta*</span>
- Possibility to input any stellar spectra model


```{toctree}
:maxdepth: 1
:caption: Get started

api
```

```{toctree}
:maxdepth: 1
:caption: Examples

notebooks/simple_example
notebooks/experiments
notebooks/amplitude_constraints.ipynb
```

```{toctree}
:maxdepth: 1
:caption: Notes

notebooks/rotation.ipynb
```