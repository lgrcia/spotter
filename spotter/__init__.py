"""
This package provides tools for modeling, visualizing, and analyzing
non-uniform stellar surfaces using HEALPix and JAX.

"""

from spotter.star import Star, show, video
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spotter")
except PackageNotFoundError:
    __version__ = "unknown"
