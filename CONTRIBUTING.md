# Contributor Guide

Thank you for your interest in improving this project. This project is
open-source and welcomes contributions in the form of bug reports, feature
requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code](https://github.com/lgrcia/spotter)
- [Documentation](https://spotter.readthedocs.io)
- [Issue Tracker](https://github.com/lgrcia/spotter/issues)

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/lgrcia/spotter/issues).

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to
reproduce the issue. In particular, please include a [Minimal, Reproducible
Example](https://stackoverflow.com/help/minimal-reproducible-example).

## How to request a feature

Feel free to request features on the [Issue
Tracker](https://github.com/lgrcia/spotter/issues).

## How to test the project

We recommend running tests in a clean virtual environment to ensure all dependencies are properly isolated. You can use [uv](https://github.com/astral-sh/uv) for fast and reproducible Python environments.

First, create and activate a new virtual environment using uv:

```bash
uv venv .venv
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
uv pip install -e "spotter[dev]"
```

Then run the tests using `pytest`:

```bash
pytest
```

This will ensure you are testing against the correct dependencies and a clean environment.

## How to submit changes

Open a [Pull Request](https://github.com/lgrcia/spotter/pulls).
