from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import jit, njit, prange

from kalman_detector.svm import kalman_binary_compress


@jit(nopython=True)
def kalman_filter(
    spec: np.ndarray,
    spec_std: np.ndarray,
    sig_eta: float,
    e0: float = 0,
    v0: float | None = None,
    chan_mask: np.ndarray | None = None,
    diffuse: bool = False,  # noqa: FBT001, FBT002
) -> float:
    """Kalman score estimator for input 1d spectrum data.

    Parameters
    ----------
    spec : np.ndarray
        1D array of spectrum values.
    spec_std : np.ndarray
        1D array of spectrum (noise) standard deviations.
    sig_eta : float
        State transition std or Process noise. Sets the smoothness scale of
        model change.
    e0 : float
        Initial guess of model expectation value in first channel, by default 0.
        Ignored for the diffuse prior.
    v0 : float, optional
        Initial guess of model variance in first channel, by default None
        (median(spec_std)**2). ``v0=np.inf`` selects the diffuse prior.
    chan_mask : np.ndarray, optional
        mask of channels to ignore, by default None
    diffuse : bool, optional
        Use the diffuse initial prior (1/v0 = 0), by default False.

    Returns
    -------
    float
        score, which is the Log likelihood ratio of the NP hypothesis test.

    Notes
    -----
    Number of changes is sqrt(nchan)*sig_eta/mean(spec_std).
    Frequency scale is 1/sig_eta**2.
    For details, see Eq. 10--12 in Kumar, Zackay & Law (2024).

    With the diffuse prior, the filter is initialised at the first unmasked
    channel and the score is the proper log likelihood ratio of the mean-free
    contrast subspace:
    ``K = S_quad - B**2 / 2 + c_det``, where ``B`` is the inverse-variance
    weighted broadband statistic. This score is exactly invariant to constant
    offsets of the spectrum and to frequency reversal, so no
    ``normalize_spectrum`` is needed.
    """
    v_init = np.median(spec_std) ** 2 if v0 is None else v0
    nchans = len(spec)
    if chan_mask is None:
        chan_mask = np.zeros(nchans, dtype=np.bool_)

    v_transit = sig_eta**2
    if diffuse or np.isinf(v_init):
        f0 = -1
        for ichan in range(nchans):
            if not chan_mask[ichan]:
                f0 = ichan
                break
        if f0 < 0:
            return 0.0
        r0 = spec_std[f0] ** 2
        e_cur = spec[f0]
        v_cur = r0 + v_transit
        quad = spec[f0] ** 2 / (2 * r0)
        lin = spec[f0] / r0
        inv_r_sum = 1 / r0
        c_det = 0.5 * np.log(r0)
        for ichan in range(f0 + 1, nchans):
            if chan_mask[ichan]:
                v_cur += v_transit
                continue
            x = spec[ichan]
            v_spec = spec_std[ichan] ** 2
            innov = x - e_cur
            quad += 0.5 * (x * x / v_spec - innov * innov / (v_cur + v_spec))
            c_det -= 0.5 * np.log1p(v_cur / v_spec)
            lin += x / v_spec
            inv_r_sum += 1 / v_spec
            gain = v_cur / (v_cur + v_spec)
            e_cur += gain * innov
            v_cur = v_transit + gain * v_spec
        c_det += 0.5 * np.log(inv_r_sum)
        return quad - 0.5 * lin * lin / inv_r_sum + c_det

    e_cur = e0
    v_cur = v_init
    log_l_h1 = 0
    log_l_h0 = 0

    for ichan in range(nchans):
        if chan_mask[ichan]:
            v_cur += v_transit
        else:
            v_spec = spec_std[ichan] ** 2
            log_l_h1 += -((spec[ichan] - e_cur) ** 2) / (
                v_cur + v_spec
            ) / 2 - 0.5 * np.log(2 * np.pi * (v_cur + v_spec))
            e_cur = e_cur + (v_cur / (v_cur + v_spec)) * (spec[ichan] - e_cur)
            v_cur = v_transit + (v_cur / (v_cur + v_spec)) * v_spec

            log_l_h0 += -(
                spec[ichan] * spec[ichan] / (spec_std[ichan] * spec_std[ichan]) / 2
            ) - 0.5 * np.log(2 * np.pi * spec_std[ichan] * spec_std[ichan])

    return log_l_h1 - log_l_h0


class GainCache(NamedTuple):
    """Data-independent quantities of the Kalman recurrence for one noise profile.

    Attributes
    ----------
    obs_idx : np.ndarray
        Indices of the unmasked channels, in frequency order.
    gains : np.ndarray
        Kalman gain ``V / (V + r)`` at each unmasked channel (1 at the diffuse
        initial channel).
    inv_innov : np.ndarray
        Inverse innovation variance ``1 / (V + r)`` at each unmasked channel
        (0 at the diffuse initial channel).
    inv_r : np.ndarray
        Inverse noise variance ``1 / r`` at each unmasked channel.
    const : float
        Log-determinant constant of the score.
    e0 : float
        Initial state mean (0 for the diffuse prior).
    inv_r_sum : float
        ``sum(1 / r)`` over unmasked channels.
    diffuse : bool
        Whether the cache represents the diffuse prior.
    """

    obs_idx: np.ndarray
    gains: np.ndarray
    inv_innov: np.ndarray
    inv_r: np.ndarray
    const: float
    e0: float
    inv_r_sum: float
    diffuse: bool


def prepare_gain_cache(
    spec_std: np.ndarray,
    sig_eta: float,
    chan_mask: np.ndarray | None = None,
    e0: float = 0,
    v0: float | None = None,
    *,
    diffuse: bool = False,
) -> GainCache:
    """Precompute the Riccati variances, gains and log-determinant constant.

    The Kalman variance recursion does not depend on the measured spectrum, so
    it can be computed once per (noise profile, mask, sig_eta) and reused for
    every spectrum scored with :func:`kalman_filter_cached`.

    Parameters
    ----------
    spec_std : np.ndarray
        1D array of spectrum (noise) standard deviations.
    sig_eta : float
        State transition std or Process noise.
    chan_mask : np.ndarray, optional
        mask of channels to ignore, by default None
    e0 : float, optional
        Initial guess of model expectation value, by default 0 (non-diffuse only).
    v0 : float, optional
        Initial guess of model variance, by default None (median(spec_std)**2).
        ``v0=np.inf`` selects the diffuse prior.
    diffuse : bool, optional
        Use the diffuse initial prior (1/v0 = 0), by default False.

    Returns
    -------
    GainCache
        Cached recurrence coefficients.
    """
    spec_std = np.asarray(spec_std, dtype=np.float64)
    nchans = len(spec_std)
    if chan_mask is None:
        chan_mask = np.zeros(nchans, dtype=bool)
    obs_idx = np.flatnonzero(~np.asarray(chan_mask, dtype=bool))
    if len(obs_idx) == 0:
        msg = "All channels are masked."
        raise ValueError(msg)
    v_init = np.median(spec_std) ** 2 if v0 is None else v0
    diffuse = diffuse or bool(np.isinf(v_init))
    v_transit = sig_eta**2
    r_obs = spec_std[obs_idx] ** 2
    gains, inv_innov, log_term = _riccati_recursion(
        obs_idx,
        r_obs,
        float(v_init),
        float(v_transit),
        diffuse,
    )
    inv_r = 1 / r_obs
    inv_r_sum = float(np.sum(inv_r))
    const = -0.5 * float(np.sum(log_term))
    if diffuse:
        const += 0.5 * np.log(r_obs[0]) + 0.5 * np.log(inv_r_sum)
    return GainCache(
        obs_idx=obs_idx,
        gains=gains,
        inv_innov=inv_innov,
        inv_r=inv_r,
        const=const,
        e0=0.0 if diffuse else float(e0),
        inv_r_sum=inv_r_sum,
        diffuse=diffuse,
    )


@njit(cache=True, fastmath=False)
def _riccati_recursion(
    obs_idx: np.ndarray,
    r_obs: np.ndarray,
    v_init: float,
    v_transit: float,
    diffuse: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data-independent Kalman variance recursion over the unmasked channels.

    Returns the gains, inverse innovation variances and ``log(1 + V / r)``
    terms. Transition variance accumulates across masked-channel gaps.
    """
    nobs = len(obs_idx)
    gains = np.empty(nobs)
    inv_innov = np.empty(nobs)
    log_term = np.empty(nobs)
    if diffuse:
        gains[0] = 1.0
        inv_innov[0] = 0.0
        log_term[0] = 0.0
        v_cur = r_obs[0]
        start = 1
    else:
        v_cur = v_init + v_transit * obs_idx[0]
        start = 0
    for ii in range(start, nobs):
        if ii > 0:
            v_cur += v_transit * (obs_idx[ii] - obs_idx[ii - 1])
        v_spec = r_obs[ii]
        inv_innov[ii] = 1 / (v_cur + v_spec)
        gains[ii] = v_cur * inv_innov[ii]
        log_term[ii] = np.log1p(v_cur / v_spec)
        v_cur = gains[ii] * v_spec
    return gains, inv_innov, log_term


@njit(cache=True, fastmath=False)
def _kalman_cached_kernel(
    spec: np.ndarray,
    obs_idx: np.ndarray,
    gains: np.ndarray,
    inv_innov: np.ndarray,
    inv_r: np.ndarray,
    e0: float,
    inv_r_sum: float,
    diffuse: bool,  # noqa: FBT001
) -> float:
    e_cur = e0
    quad = 0.0
    lin = 0.0
    for ii in range(len(obs_idx)):
        x = spec[obs_idx[ii]]
        innov = x - e_cur
        quad += x * x * inv_r[ii] - innov * innov * inv_innov[ii]
        lin += x * inv_r[ii]
        e_cur += gains[ii] * innov
    quad *= 0.5
    if diffuse:
        quad -= 0.5 * lin * lin / inv_r_sum
    return quad


@njit(cache=True, parallel=True, fastmath=False)
def _kalman_cached_batch_kernel(
    specs: np.ndarray,
    obs_idx: np.ndarray,
    gains: np.ndarray,
    inv_innov: np.ndarray,
    inv_r: np.ndarray,
    e0: float,
    inv_r_sum: float,
    diffuse: bool,  # noqa: FBT001
) -> np.ndarray:
    nspec = specs.shape[0]
    out = np.empty(nspec)
    for ispec in prange(nspec):
        out[ispec] = _kalman_cached_kernel(
            specs[ispec],
            obs_idx,
            gains,
            inv_innov,
            inv_r,
            e0,
            inv_r_sum,
            diffuse,
        )
    return out


def kalman_filter_cached(spec: np.ndarray, cache: GainCache) -> float | np.ndarray:
    """Kalman score using a precomputed :class:`GainCache`.

    Equivalent to :func:`kalman_filter` with the parameters used to build
    ``cache``, but only the data-dependent mean update and quadratic terms are
    evaluated.

    Parameters
    ----------
    spec : np.ndarray
        1D spectrum (nchans,) or 2D batch of spectra (nspec, nchans).
    cache : GainCache
        Output of :func:`prepare_gain_cache`.

    Returns
    -------
    float | np.ndarray
        Score (1D input) or array of scores (2D input).
    """
    spec = np.asarray(spec, dtype=np.float64)
    args = (
        cache.obs_idx,
        cache.gains,
        cache.inv_innov,
        cache.inv_r,
        cache.e0,
        cache.inv_r_sum,
        cache.diffuse,
    )
    if spec.ndim == 1:
        return _kalman_cached_kernel(spec, *args) + cache.const
    if spec.ndim == 2:
        return _kalman_cached_batch_kernel(np.ascontiguousarray(spec), *args) + (
            cache.const
        )
    msg = f"spec must be 1D or 2D, got {spec.ndim}D."
    raise ValueError(msg)


def kalman_filter_binary(
    spec: np.ndarray,
    spec_std: np.ndarray,
    sig_eta: float,
    e0: float = 0,
    v0: float | None = None,
) -> float:
    """Kalman score estimator for input 1d spectrum data in binary search tree.

    Parameters
    ----------
    spec : np.ndarray
        1D array of spectrum values.
    spec_std : np.ndarray
        1D array of spectrum (noise) standard deviations.
    sig_eta : float
        State transition std or Process noise.
    e0 : float
        Initial guess of the expected value of the first hidden state A0, by default 0.
    v0 : float, optional
        Initial guess of the variance of the first hidden state A0, by default None.

    Returns
    -------
    float
        score, Log likelihood ratio of the NP hypothesis test.
    """
    if v0 is None:
        v0 = np.median(spec_std) ** 2

    state = kalman_binary_compress(spec, spec_std, sig_eta, e0, v0)
    log_l_h1 = state.log_s + np.log(np.sqrt((2 * np.pi) ** 2 / np.linalg.det(state.m)))
    log_l_h0 = np.sum(
        0.5 * np.log(1 / spec_std**2 / 2 / np.pi) - spec**2 / (2 * spec_std**2),
    )
    return log_l_h1 - log_l_h0
