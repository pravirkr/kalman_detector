"""Detecting fast radio bursts with spectral structure."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kalman_detector")
except PackageNotFoundError:
    __version__ = "0+unknown"
