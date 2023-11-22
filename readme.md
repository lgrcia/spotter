# spotter

<p align="center">
  Stellar contamination estimates from rotational light curves
  <br>
  <p align="center">
    <a href="https://github.com/lgrcia/spotter">
      <img src="https://img.shields.io/badge/github-lgrcia/spotter-indianred.svg?style=flat" alt="github"/>
    </a>
    <a href="LICENCE">
      <img src="https://img.shields.io/badge/license-MIT-lightgray.svg?style=flat" alt="license"/>
    </a>
  </p>
</p>

*spotter* is a Python package to estimate transmission spectra stellar contamination from stellar rotational light curves.

## Installation

For now only locally with

```
pip install -e spotter
```

with *spotter* cloned using
```
git clone https://github.com/lgrcia/spotter
```

## Example

```python
import numpy as np
from spotter import Star, butterfly
import matplotlib.pyplot as plt

star = Star(u=[0.1, 0.2], N=128)

# spots properties
n = 200
np.random.seed(42)
theta, phi = butterfly(0.25, 0.08, n)
radii = np.random.uniform(0.01, 0.1, n)
contrast = 0.1

# add spots
star.add_spot(theta, phi, radii, contrast)

# compute light curve
phase = np.linspace(0, 4 * np.pi, 200)
flux = star.flux(phase)
```

### Plotting

```python
plt.figure(figsize=(9, 3))
plt.subplot(1, 5, (1, 2))
star.plot()

plt.subplot(1, 5, (3, 5))
plt.plot(phase, flux, c="k")
plt.xlabel("phase")
plt.ylabel("diff. flux")

```

<p align="center" style="margin-bottom:-50px">
    <img src="docs/_static/example_star.png" width="100%">
</p>

