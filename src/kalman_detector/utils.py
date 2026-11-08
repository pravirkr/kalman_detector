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


def effective_number_of_tests(corr: np.ndarray) -> float:
    """Effective number of independent tests from a correlation matrix.

    Uses the Li & Ji (2005, Heredity 95, 221) eigenvalue estimator,
    ``N_eff = sum_i [I(|e_i| >= 1) + (|e_i| - floor(|e_i|))]``, where ``e_i``
    are the eigenvalues of the correlation matrix.

    Parameters
    ----------
    corr : np.ndarray
        Square correlation matrix of the test statistics.

    Returns
    -------
    float
        Effective number of tests, between 1 and ``len(corr)``.
    """
    corr = np.atleast_2d(np.asarray(corr, dtype=np.float64))
    eigs = np.abs(np.linalg.eigvalsh(corr))
    # Snap round-off (e.g. 4.9999999999) so floor() is not off by one.
    nearest = np.round(eigs)
    eigs = np.where(np.abs(eigs - nearest) < 1e-9, nearest, eigs)
    n_eff = np.sum((eigs >= 1).astype(float) + (eigs - np.floor(eigs)))
    return float(np.clip(n_eff, 1, len(corr)))


def sidak_logsf(logsf: float, n_trials: float) -> float:
    """Trials-corrected log p-value of the minimum of ``n_trials`` tests.

    Computes ``log(1 - (1 - p)**n_trials)`` stably for ``p = exp(logsf)``.

    Parameters
    ----------
    logsf : float
        log p-value of the best (minimum p-value) test.
    n_trials : float
        (Effective) number of independent trials.

    Returns
    -------
    float
        Corrected log p-value (<= 0).
    """
    if logsf >= 0:
        return 0.0
    if logsf < -30:
        # 1 - (1 - p)^n = n p (1 + O(n p)), with n p < 1e-12 relative error.
        return float(min(0.0, np.log(n_trials) + logsf))
    return float(min(0.0, _log1mexp(n_trials * _log1mexp(logsf))))


def _log1mexp(x: float) -> float:
    """Compute ``log(1 - exp(x))`` accurately for ``x < 0`` (Maechler 2012)."""
    if x > -np.log(2):
        return float(np.log(-np.expm1(x)))
    return float(np.log1p(-np.exp(x)))


def fisher_logsf(logsf_a: float, logsf_b: float) -> float:
    """Calibrated log p-value of the product of two independent p-values.

    For independent uniform p-values, ``P(p_a p_b <= p_x) = p_x (1 - log p_x)``
    (Fisher's method with 4 degrees of freedom).

    Parameters
    ----------
    logsf_a, logsf_b : float
        log p-values of the two independent tests.

    Returns
    -------
    float
        Combined log p-value (<= 0).
    """
    log_px = logsf_a + logsf_b
    return float(min(0.0, log_px + np.log1p(-log_px)))


class SnrResult:
    """Combined box-car and Kalman detection significance.

    Parameters
    ----------
    name : str
        Name of the candidate.
    snr_box : float
        Box-car (broadband) S/N.
    sig_kalman : float
        Best Kalman significance (log p-value) over the transit sigmas.
    n_eff : float, optional
        Effective number of trials over the transit sigmas used to correct
        ``sig_kalman`` (see :attr:`KalmanDetector.n_eff`), by default 1
        (no correction).

    Notes
    -----
    The calibrated combination corrects the best Kalman p-value for the
    selection over transit sigmas (Sidak with ``n_eff``) and combines it with
    the box-car p-value with Fisher's method, ``p_x (1 - log p_x)``. The legacy
    uncalibrated product ``p_box * p_kalman`` is kept as
    :attr:`snr_kalman_legacy`.
    """

    def __init__(
        self,
        name: str,
        snr_box: float,
        sig_kalman: float,
        n_eff: float = 1.0,
    ) -> None:
        self.name = name
        self.snr_box = snr_box
        self.sig_kalman = sig_kalman
        self.n_eff = n_eff

    @property
    def sig_box(self) -> float:
        return stats.norm.logsf(self.snr_box)

    @property
    def sig_kalman_corrected(self) -> float:
        """Kalman log p-value corrected for the selection over transit sigmas."""
        return sidak_logsf(self.sig_kalman, self.n_eff)

    @property
    def sig_combined(self) -> float:
        """Calibrated Fisher combination of box-car and Kalman log p-values."""
        return fisher_logsf(self.sig_box, self.sig_kalman_corrected)

    @property
    def snr_kalman(self) -> float:
        """Gaussian-equivalent S/N of the calibrated combined significance."""
        return get_snr_from_logsf(self.sig_combined)

    @property
    def snr_kalman_legacy(self) -> float:
        """Legacy (uncalibrated) S/N from the plain product of p-values."""
        return get_snr_from_logsf(self.sig_box + self.sig_kalman)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "sig_box": self.sig_box,
            "sig_kalman": self.sig_kalman,
            "n_eff": self.n_eff,
            "sig_kalman_corrected": self.sig_kalman_corrected,
            "sig_combined": self.sig_combined,
            "snr_box": self.snr_box,
            "snr_kalman": self.snr_kalman,
            "snr_kalman_legacy": self.snr_kalman_legacy,
        }

    def __str__(self) -> str:
        return (
            f"{self.name} - Box S/N: {self.snr_box:.1f}, "
            f"Kalman S/N: {self.snr_kalman:.1f} "
            f"(legacy: {self.snr_kalman_legacy:.1f})\n"
            f"            Box Sig: {-self.sig_box:.1f}, "
            f"Kalman Sig: {-self.sig_kalman:.1f}, "
            f"Combined Sig: {-self.sig_combined:.1f} (n_eff={self.n_eff:.2f})"
        )
