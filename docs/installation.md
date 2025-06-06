# Installation

## Installation

To install *spotter* from pypi
    
```bash
pip install spotter
```

As *spotter* is still under development, we recommend installing the latest version from the GitHub repository. To do so, clone the repository and install the package using pip:

```bash
git clone https://github.com/lgrcia/spotter
pip install -e spotter
```

## Testing

To run the test suite, first install the development dependencies:

```bash
pip install "spotter[test]"
```

Then run the tests using `pytest`:

```bash
pytest tests
```