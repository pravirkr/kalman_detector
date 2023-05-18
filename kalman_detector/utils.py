from __future__ import annotations
import numpy as np
from scipy import stats


def add_noise(
    spec: np.ndarray, spec_std: np.ndarray, current_snr: float, target_snr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Add noise to spectrum to achieve target_snr.

    Parameters
    ----------
    spec : np.ndarray
        1d spectra
    spec_std : np.ndarray
        1d spectra noise
    current_snr : float
        Current S/N of the spectrum.
    target_snr : float
        Target S/N of the spectrum.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Noise added spectrum and spectrum noise.
    """
    current_variance = np.sum(spec_std**2) / len(spec_std)
    needed_std = (
        (current_snr**2 - target_snr**2) * current_variance / target_snr**2
    ) ** 0.5
    spec_noise = spec + np.random.normal(0, needed_std, len(spec))
    spec_std_noise = (spec_std**2 + needed_std**2) ** 0.5
    return spec_noise, spec_std_noise


def normalize_spectrum(
    spec: np.ndarray, spec_std: np.ndarray, chan_mask: np.ndarray | None = None
) -> np.ndarray:
    """Normalize spectrum to zero mean. Likelihood calc expects zero mean spec

    Parameters
    ----------
    spec : np.ndarray
        1d spectra
    spec_std : np.ndarray
        1d spectra noise
    chan_mask : np.ndarray, optional
        Mask of channels to ignore, by default None

    Returns
    -------
    np.ndarray
        Normalized spectrum.
    """
    if chan_mask is None:
        chan_mask = np.zeros(len(spec), dtype=bool)
    spec_mean = np.average(spec[~chan_mask], weights=spec_std[~chan_mask] ** -2)
    norm_spec = spec - spec_mean
    return norm_spec


def get_snr_from_logsf(logsf: float) -> float:
    """Get S/N from log of significance.

    Parameters
    ----------
    logsf : float
        log of significance

    Returns
    -------
    float
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


def simulate_1d_gaussian_process(
    mean: float, noise_std: float, nchan: int, corr_len: float, amp: float
) -> np.ndarray:
    """Simulate 1d Gaussian process.

    Parameters
    ----------
    mean : float
        Mean of the noise.
    noise_std : float
        Standard deviation of the noise.
    nchan : int
        Number of channels.
    corr_len : float
        Correlation length.
    amp : float
        Amplitude of the process.

    Returns
    -------
    np.ndarray
        Simulated process.
    """
    noise = np.random.normal(mean, noise_std, nchan)
    kernel = np.exp(-np.linspace(-nchan / corr_len, nchan / corr_len, nchan) ** 2)
    kernel /= np.sum(kernel)
    base_process = np.random.normal(0, amp / np.sqrt(corr_len), nchan)
    signal = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(base_process))
    return np.abs(signal) + noise


def simulate_1d_abs_complex_process(
    noise_std: float, nchan: int, corr_len: float, snr_int: float
) -> np.ndarray:
    """Simulate 1d complex process.

    Parameters
    ----------
    noise_std : float
        Noise standard deviation.
    nchan : int
        Number of channels.
    corr_len : float
        Correlation length.
    snr_int : float
        S/N of the process.

    Returns
    -------
    np.ndarray
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
