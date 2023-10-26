from __future__ import annotations
from numba import jit
import numpy as np


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
        1d spectra
    spec_std : np.ndarray
        1d spectra noise std
    sig_eta : float
        State transition std or Process noise. Sets the smoothness scale of model change.
    e0 : float
        initial guess of model expectation value in first channel, by default 0
    v0 : float, optional
        initial guess of model std in first channel, by default None
    chan_mask : np.ndarray, optional
        mask of channels to ignore, by default None

    Returns
    -------
    float
        score, which is the likelihood of presence of signal.

    Notes
    -----
    Number of changes is sqrt(nchan)*sig_eta/mean(spec_std). Frequency scale is 1/sig_eta**2.
    For details, see Eq. 10--12 in Kumar, Zackay & Law (2023).
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
