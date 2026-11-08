from __future__ import annotations

import numpy as np
from scipy import stats


def add_noise(
    spec: np.ndarray,
    spec_std: np.ndarray,
    current_snr: float,
    target_snr: float,
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
    current_variance = np.mean(spec_std**2)
    needed_std = np.sqrt(
        current_variance * (current_snr**2 - target_snr**2) / target_snr**2,
    )
    rng = np.random.default_rng()
    spec_noise = spec + rng.normal(0, needed_std, len(spec))
    spec_std_noise = (spec_std**2 + needed_std**2) ** 0.5
    return spec_noise, spec_std_noise


def normalize_spectrum(
    spec: np.ndarray,
    spec_std: np.ndarray,
    chan_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize spectrum to zero mean. Likelihood calc expects zero mean spec.

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
    return spec - spec_mean


def normalize(
    spec: np.ndarray,
    spec_std: np.ndarray,
    spec_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize spectrum to zero mean and unit std.

    Parameters
    ----------
    spec : np.ndarray
        1d spectra
    spec_std : np.ndarray
        1d spectra noise
    spec_mean : np.ndarray | None, optional
        Mean of the spectrum, by default None

    Returns
    -------
    np.ndarray
        Normalized spectrum.
    """
    if spec_mean is None:
        spec_mean = np.zeros_like(spec)
    return np.divide(
        spec - spec_mean,
        spec_std,
        out=np.zeros_like(spec),
        where=~np.isclose(spec_std, 0, atol=1e-5),
    )


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
    isf function returns bad results if we try to feed it with np.exp(-600) and beyond.
    This is because the double epsilon is reached. Needless to say, at this point the
    exact significance does not mean anything. In this case, we return a good
    approximation for S/N.
    """
    if np.abs(logsf) > 600:
        return np.abs(2 * logsf - np.log(np.abs(2 * logsf))) ** 0.5
    return stats.norm.isf(np.exp(logsf))


def simulate_gaussian_signal(
    nchans: int,
    corr_len: float,
    *,
    complex_process: bool = False,
) -> np.ndarray:
    """Simulate 1d Gaussian process.

    Parameters
    ----------
    nchans : int
        Number of frequency channels.
    corr_len : float
        Correlation length of the Gaussian process.
    complex_process : bool, optional
        whether to use comlex numbers as the base process, by default False

    Returns
    -------
    np.ndarray
        Normalized mean-subtracted signal array.
    """
    rng = np.random.default_rng()
    kernel = np.exp(-(np.linspace(-nchans / corr_len, nchans / corr_len, nchans) ** 2))
    kernel /= np.dot(kernel, kernel) ** 0.5
    if complex_process:
        base_process = rng.normal(0, 1 / np.sqrt(corr_len), nchans) + 1.0j * rng.normal(
            0,
            1 / np.sqrt(corr_len),
            nchans,
        )
    else:
        base_process = rng.normal(0, 1 / np.sqrt(corr_len), nchans)
    signal = np.abs(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(base_process)))
    signal -= np.mean(signal)
    return signal / np.dot(signal, signal) ** 0.5


class SnrResult:
    def __init__(self, name: str, snr_box: float, sig_kalman: float) -> None:
        self.name = name
        self.snr_box = snr_box
        self.sig_kalman = sig_kalman

    @property
    def sig_box(self) -> float:
        return stats.norm.logsf(self.snr_box)

    @property
    def snr_kalman(self) -> float:
        return get_snr_from_logsf(self.sig_box + self.sig_kalman)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "sig_box": self.sig_box,
            "sig_kalman": self.sig_kalman,
            "snr_box": self.snr_box,
            "snr_kalman": self.snr_kalman,
        }

    def __str__(self) -> str:
        return (
            f"{self.name} - Box S/N: {self.snr_box:.1f}, "
            f"Kalman S/N: {self.snr_kalman:.1f}\n"
            f"            Box Sig: {-self.sig_box:.1f}, "
            f"Kalman Sig: {-self.sig_kalman:.1f}"
        )
