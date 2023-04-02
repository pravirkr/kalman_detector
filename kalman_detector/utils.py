import numpy as np
from scipy import stats


def add_noise(spec, spec_std, current_snr, target_snr):
    """Add noise to spectrum to achieve target_snr.

    Parameters
    ----------
    spec : _type_
        1d spectra
    spec_std : _type_
        1d spectra noise
    current_snr : _type_
        Current S/N of the spectrum.
    target_snr : _type_
        Target S/N of the spectrum.

    Returns
    -------
    _type_
        Noise added spectrum.
    """
    current_variance = np.sum(spec_std**2) / len(spec_std)
    needed_std = (
        (current_snr**2 - target_snr**2) * current_variance / target_snr**2
    ) ** 0.5
    spec_noise = spec + np.random.normal(0, needed_std, len(spec))
    spec_std_noise = (spec_std**2 + needed_std**2) ** 0.5
    return spec_noise, spec_std_noise


def normalize_spectrum(spec, spec_std, chan_mask=None):
    """Normalize spectrum to zero mean. Likelihood calc expects zero mean spec

    Parameters
    ----------
    spec : _type_
        _description_
    spec_std : _type_
        _description_
    chan_mask : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if chan_mask is None:
        chan_mask = np.zeros(len(spec), dtype=bool)
    spec_mean = np.average(spec[~chan_mask], weights=spec_std[~chan_mask] ** -2)
    spec[~chan_mask] -= spec_mean
    return spec


def get_snr_from_logsf(logsf):
    """Get S/N from log of significance.

    Parameters
    ----------
    logsf : _type_
        log of significance

    Returns
    -------
    _type_
        S/N

    Notes
    -----
    isf function returns bad results if we try to feed it with np.exp(-600) and beyond. This
    is because the double epsilon is reached. Needless to say, at this point the exact significance
    does not mean anything. In this case, we return a good approximation for S/N.
    """
    if np.abs(logsf) > 600:
        return np.abs(2 * logsf - np.log(np.abs(2 * logsf))) ** 0.5
    return stats.norm.isf(np.exp(logsf))


def simulate_1d_gaussian_process(mean, noise_std, nchan, corr_len, amp):
    """Simulate 1d Gaussian process.

    Parameters
    ----------
    mean : _type_
        Mean of the noise.
    noise_std : _type_
        Standard deviation of the noise.
    nchan : _type_
        Number of channels.
    corr_len : _type_
        Correlation length.
    amp : _type_
        Amplitude of the process.

    Returns
    -------
    _type_
        Simulated process.
    """
    noise = np.random.normal(mean, noise_std, nchan)
    kernel = np.exp(-np.linspace(-nchan / corr_len, nchan / corr_len, nchan) ** 2)
    kernel /= np.sum(kernel)
    base_process = np.random.normal(0, amp / np.sqrt(corr_len), nchan)
    signal = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(base_process))
    return np.abs(signal) + noise


def simulate_1d_abs_complex_process(noise_std, nchan, corr_len, snr_int):
    """Simulate 1d complex process.

    Parameters
    ----------
    noise_std : _type_
        Noise standard deviation.
    nchan : _type_
        Number of channels.
    corr_len : _type_
        Correlation length.
    snr_int : _type_
        S/N of the process.

    Returns
    -------
    _type_
        Simulated process.
    """
    noise = np.random.normal(0, noise_std, nchan)
    kernel = np.exp(-np.linspace(-nchan / corr_len, nchan / corr_len, nchan) ** 2)
    kernel /= np.sum(kernel)
    base_process = np.random.normal(0, 1 / np.sqrt(corr_len), nchan) + complex(
        0, 1
    ) * np.random.normal(0, 1 / np.sqrt(corr_len), nchan)
    signal = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(base_process))
    signal = np.abs(signal)
    signal = signal / np.mean(signal) * snr_int / nchan**0.5 / noise_std
    return signal + noise
