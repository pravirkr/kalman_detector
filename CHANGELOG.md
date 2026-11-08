# Changelog

## Unreleased

- Build the package from `src/kalman_detector` with the uv build backend.
- Require Python 3.12 or newer, on Linux and macOS.
- Set lower bounds on the runtime dependencies and drop `typing_extensions`.
- Mark the package as typed with `py.typed`, and read `__version__` from the installed distribution.
- Run tests with uv on Python 3.12, 3.13, and 3.14, and publish releases with `uv build`.

## 0.5.0 (2024-05-26)

- First tagged release, published to PyPI.
- Kalman detector for smoothly varying signals in Gaussian noise, including the binary-state implementation.
- Efficiency curves for simulated signals.
- Example script and data for FRB 20171003 and FRB 20171004.
