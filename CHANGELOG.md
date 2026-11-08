# Changelog

## Unreleased

- Build the package from `src/kalman_detector` with the uv build backend.
- Require Python 3.12 or newer, on Linux and macOS.
- Set lower bounds on the runtime dependencies and drop `typing_extensions`.
- Mark the package as typed with `py.typed`, and read `__version__` from the installed distribution.
- Run tests with uv on Python 3.12, 3.13, and 3.14, and publish releases with `uv build`.
- Compute Monte Carlo efficiency with NumPy and drop the `uncertainties` dependency.
- Document the Python 3.12 install and the uv development setup.
- Add the diffuse initial prior (`diffuse=True` or `v0=np.inf`) to `kalman_filter`, scoring the mean-free contrast likelihood ratio, which is invariant to constant offsets.
- Add `prepare_gain_cache` and `kalman_filter_cached`, which precompute the Kalman gains once per noise profile and score single spectra or batches.
- Calibrate `KalmanDetector` analytically by default (`prepare_fits(method="saddlepoint")`), using tridiagonal eigenvalues and a saddlepoint tail instead of Monte Carlo. `KalmanDetector` now uses the diffuse prior by default; `method="mc"` and `diffuse=False` keep the legacy behaviour.
- Correct `SnrResult` for the choice of the best smoothness (`n_eff`) and combine p-values with Fisher's method; the old value is kept as `snr_kalman_legacy`.
- Add `binary_tree_bank` and `kalman_binary_compress_diffuse` to `svm`: the diffuse contrast score by a canonical binary-tree reduction (16-FLOP merge), for any channel count, per-channel noise and channel masks. It matches `kalman_filter(..., diffuse=True)` to about 1e-12.
- Take the default `v0` of `kalman_filter` and `prepare_gain_cache` from the unmasked channels only.

## 0.5.0 (2024-05-26)

- First tagged release, published to PyPI.
- Kalman detector for smoothly varying signals in Gaussian noise, including the binary-state implementation.
- Efficiency curves for simulated signals.
- Example script and data for FRB 20171003 and FRB 20171004.
