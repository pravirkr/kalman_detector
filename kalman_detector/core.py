import numpy as np
from numba import jit


@jit(nopython=True)
def kalman_filter(spec, spec_std, sig_t, e0=0, v0=None, chan_mask=None):
    """Core calculation of Kalman estimator of input 1d spectrum data.

    Number of changes is sqrt(nchan)*sig_t/mean(spec_std). Frequency scale is 1/sig_t**2.

    Parameters
    ----------
    spec : _type_
        1d spectra
    spec_std : _type_
        1d spectra noise
    sig_t : _type_
        sets the smoothness scale of model (A) change.
    e0 : int, optional
        initial guess of model expectation value in first channel, by default 0
    v0 : _type_, optional
        initial guess of model standard deviation in first channel, by default None
    chan_mask : _type_, optional
        mask of channels to ignore, by default None

    Returns
    -------
    _type_
        score, which is the likelihood of presence of signal.
    """
    if v0 is None:
        v0 = np.median(spec_std) ** 2

    if chan_mask is None:
        chan_mask = np.zeros(len(spec), dtype=np.bool_)

    e_cur = e0
    v_cur = v0
    v_transit = sig_t**2
    log_l_cur = 0
    log_l_h0 = 0

    for ichan in range(len(spec)):
        if not chan_mask[ichan]:
            v_spec = spec_std[ichan] ** 2
            log_l_cur += -((spec[ichan] - e_cur) ** 2) / (
                v_cur + v_spec
            ) / 2 - 0.5 * np.log(2 * np.pi * (v_cur + v_spec))
            e_cur = e_cur + (v_cur / (v_cur + v_spec)) * (spec[ichan] - e_cur)
            v_cur = v_transit + (v_cur / (v_cur + v_spec)) * v_spec

            log_l_h0 += -(
                spec[ichan] * spec[ichan] / (spec_std[ichan] * spec_std[ichan]) / 2
            ) - 0.5 * np.log(2 * np.pi * spec_std[ichan] * spec_std[ichan])

        else:
            v_cur += v_transit

    return log_l_cur - log_l_h0
