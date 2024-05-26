from __future__ import annotations

import numpy as np
from numba import jit

from kalman_detector.svm import kalman_binary_compress


@jit(nopython=True)
def kalman_filter(
    spec: np.ndarray,
    spec_std: np.ndarray,
    sig_eta: float,
    e0: float = 0,
    v0: float | None = None,
    chan_mask: np.ndarray | None = None,
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
        Initial guess of model expectation value in first channel, by default 0
    v0 : float, optional
        Initial guess of model std in first channel, by default None
    chan_mask : np.ndarray, optional
        mask of channels to ignore, by default None

    Returns
    -------
    float
        score, which is the Log likelihood ratio of the NP hypothesis test.

    Notes
    -----
    Number of changes is sqrt(nchan)*sig_eta/mean(spec_std).
    Frequency scale is 1/sig_eta**2.
    For details, see Eq. 10--12 in Kumar, Zackay & Law (2024).
    """
    if v0 is None:
        v0 = np.median(spec_std) ** 2

    if chan_mask is None:
        chan_mask = np.zeros(len(spec), dtype=np.bool_)

    e_cur = e0
    v_cur = v0
    v_transit = sig_eta**2
    nchans = len(spec)
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
