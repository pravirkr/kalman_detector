from __future__ import annotations

import numpy as np
from scipy import optimize, stats


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


def simulate_ar1_signal(
    nchans: int,
    alpha: float = 0.98,
    sig_eta: float = 0.1,
    beta: float = 0.0,
    *,
    normalize_norm: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a 1D AR(1) / Ornstein-Uhlenbeck or random-walk spectral signal.

    Evolves ``A_f = alpha * A_{f-1} + (1 - alpha) * beta + eta_f`` with
    ``eta_f ~ N(0, sig_eta**2)``. Setting ``alpha = 1`` yields a pure random
    walk; ``0 < alpha < 1`` yields a stationary mean-reverting
    Ornstein-Uhlenbeck process with correlation length ``1 / (1 - alpha)``.

    Parameters
    ----------
    nchans : int
        Number of frequency channels.
    alpha : float, optional
        AR(1) persistence parameter in ``(0, 1]``, by default 0.98.
    sig_eta : float, optional
        Innovation standard deviation per channel step, by default 0.1.
    beta : float, optional
        Long-run mean of the state process, by default 0.0.
    normalize_norm : bool, optional
        Subtract the sample mean and normalize to unit L2 norm, by default True.
    rng : np.random.Generator, optional
        Random number generator, by default None.

    Returns
    -------
    np.ndarray
        Simulated 1D spectral signal of length ``nchans``.
    """
    if rng is None:
        rng = np.random.default_rng()
    innovations = rng.normal(0.0, sig_eta, nchans)
    signal = np.empty(nchans, dtype=np.float64)
    if 0.0 < alpha < 1.0:
        stat_std = sig_eta / np.sqrt(1.0 - alpha**2)
        state = float(rng.normal(beta, stat_std))
    else:
        state = float(beta)
    for idx in range(nchans):
        signal[idx] = state
        state = alpha * state + innovations[idx] + beta * (1.0 - alpha)
    if normalize_norm:
        signal -= np.mean(signal)
        norm = float(np.linalg.norm(signal))
        if norm > 0:
            signal /= norm
    return signal


def simulate_scintillation_2d(
    nchans: int = 1024,
    n_pix: int = 128,
    power: float = 5.0 / 3.0,
    lam_range: tuple[float, float] = (1.0, 2.0),
    delay_pow: float = 2.0,
    phase_scale: float = 2e6,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate multi-island scintillation spectra via a 2D turbulent phase screen.

    Generates a 2D spatial phase screen with power-law spectrum
    ``(k_x**2 + k_y**2)**(-power / 2)`` (``power = 5/3`` for Kolmogorov
    turbulence) and propagates the wavelength-dependent phase
    ``phase_map * phase_scale * lam**(delay_pow - 1)`` via 2D FFT.

    Parameters
    ----------
    nchans : int, optional
        Number of frequency/wavelength channels, by default 1024.
    n_pix : int, optional
        Spatial grid size ``(n_pix, n_pix)`` of the 2D phase screen, by default 128.
    power : float, optional
        Turbulent power-law index (5/3 for Kolmogorov), by default 5/3.
    lam_range : tuple[float, float], optional
        Dimensionless wavelength range ``(lam_min, lam_max)``, by default (1.0, 2.0).
    delay_pow : float, optional
        Wavelength scaling exponent of the dispersive/refractive phase, by default 2.0.
    phase_scale : float, optional
        Overall phase screen amplitude factor, by default 2e6.
    rng : np.random.Generator, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(phase_map, spectra)`` where ``phase_map`` has shape ``(n_pix, n_pix)``
        and ``spectra`` has shape ``(nchans, n_pix)`` (each column is a 1D
        scintillated spectrum across the ``nchans`` channels).
    """
    if rng is None:
        rng = np.random.default_rng()
    grid = np.arange(-n_pix / 2.0, n_pix / 2.0)
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="ij")
    r2 = np.fft.fftshift(x_grid**2 + y_grid**2)
    with np.errstate(divide="ignore"):
        amp_map = np.where(r2 > 0, r2 ** (-0.5 * power), 0.0)
    amp_map[0, 0] = np.max(amp_map[1:, :])
    rand_phase = np.exp(2j * np.pi * rng.random((n_pix, n_pix)))
    phase_map = np.real(np.fft.ifft2(rand_phase * amp_map))
    lam_grid = phase_scale * np.linspace(lam_range[0], lam_range[1], nchans)
    spectra = np.empty((nchans, n_pix), dtype=np.float64)
    for idx, lam in enumerate(lam_grid):
        field = np.exp(1j * phase_map * (lam ** (delay_pow - 1.0)))
        spectra[idx] = np.abs(np.fft.fft2(field)[0])
    return phase_map, spectra


def extract_waterfall_spectrum(
    waterfall: np.ndarray,
    on_pulse_bins: tuple[int, int],
    *,
    profile_weighted: bool = False,
    guard_bins: int = 50,
    mask_tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Extract on-pulse spectrum, per-channel std, boxcar S/N, and off-pulse spectra.

    Supports both flat boxcar integration (``profile_weighted=False``) and
    cross-correlation against the normalized broadband temporal pulse profile
    (``profile_weighted=True``, which preserves exact independence between the
    broadband S/N and the mean-free Kalman contrast score under Gaussian noise).

    Parameters
    ----------
    waterfall : np.ndarray
        2D dedispersed dynamic spectrum of shape ``(nchans, ntime)``.
    on_pulse_bins : tuple[int, int]
        ``(start_bin, stop_bin)`` of the on-pulse window ``[start_bin, stop_bin)``.
    profile_weighted : bool, optional
        Weight time bins inside ``on_pulse_bins`` by the broadband temporal
        profile instead of a flat top-hat boxcar, by default False.
    guard_bins : int, optional
        Guard margin in time bins around ``on_pulse_bins`` excluded from the
        off-pulse windows, by default 50.
    mask_tol : float, optional
        Tolerance for flagging zero-variance channels, by default 1e-5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, np.ndarray]
        ``(spec, spec_std, snr_boxcar, off_pulse_specs)`` where ``spec`` and
        ``spec_std`` have shape ``(nchans,)``, ``snr_boxcar`` is the inverse-variance
        weighted broadband S/N, and ``off_pulse_specs`` has shape ``(n_off, nchans)``.
    """
    wf = np.asarray(waterfall, dtype=np.float64)
    if wf.ndim != 2:
        msg = "waterfall must be a 2D array of shape (nchans, ntime)."
        raise ValueError(msg)
    nchans, ntime = wf.shape
    start, stop = int(on_pulse_bins[0]), int(on_pulse_bins[1])
    width = stop - start
    if width < 1 or start < 0 or stop > ntime:
        msg = f"Invalid on_pulse_bins={on_pulse_bins} for ntime={ntime}."
        raise ValueError(msg)

    is_off = np.ones(ntime, dtype=bool)
    is_off[max(0, start - guard_bins) : min(ntime, stop + guard_bins)] = False
    if np.sum(is_off) < 2:
        is_off = np.ones(ntime, dtype=bool)
        is_off[start:stop] = False

    chan_mean = wf[:, is_off].mean(axis=1)
    chan_std_1bin = wf[:, is_off].std(axis=1)
    wf_centered = wf - chan_mean[:, None]

    if profile_weighted and width > 1:
        prof = wf_centered[:, start:stop].mean(axis=0)
        norm_prof = float(np.linalg.norm(prof))
        weights_t = (
            prof / norm_prof if norm_prof > 0 else np.full(width, 1.0 / np.sqrt(width))
        )
        scale = 1.0
    else:
        weights_t = np.ones(width, dtype=np.float64)
        scale = float(np.sqrt(width))

    spec = wf_centered[:, start:stop] @ weights_t
    spec_std = chan_std_1bin * scale

    off_starts = [
        t
        for t in range(0, ntime - width + 1, width)
        if t + width <= start - guard_bins or t >= stop + guard_bins
    ]
    if off_starts:
        off_pulse_specs = np.array(
            [wf_centered[:, t : t + width] @ weights_t for t in off_starts],
            dtype=np.float64,
        )
    else:
        off_pulse_specs = np.empty((0, nchans), dtype=np.float64)

    valid = ~np.isclose(spec_std, 0.0, atol=mask_tol)
    if np.any(valid):
        inv_var = 1.0 / spec_std[valid] ** 2
        snr_boxcar = float(np.sum(spec[valid] * inv_var) / np.sqrt(np.sum(inv_var)))
    else:
        snr_boxcar = 0.0
    return spec, spec_std, snr_boxcar, off_pulse_specs


def fit_spectral_running(
    freqs: np.ndarray,
    spec: np.ndarray,
    spec_std: np.ndarray | None = None,
    f0: float | None = None,
    mask_tol: float = 1e-5,
) -> dict[str, float]:
    """Fit the CHIME/FRB log-parabola spectral running model to a 1D spectrum.

    Fits ``S(f) = amp * (f / f0)**(gamma + beta * log(f / f0))`` across unmasked
    channels using non-linear least squares (adapted from the FRB 20201124A
    analysis in Paper 1).

    Parameters
    ----------
    freqs : np.ndarray
        1D array of channel frequencies (positive, e.g. in MHz).
    spec : np.ndarray
        1D observed on-pulse spectrum.
    spec_std : np.ndarray, optional
        1D per-channel noise standard deviation, by default None (uniform weights).
    f0 : float, optional
        Pivot frequency, by default the geometric mean of unmasked ``freqs``.
    mask_tol : float, optional
        Tolerance for flagging zero-std channels, by default 1e-5.

    Returns
    -------
    dict[str, float]
        Dictionary with fitted ``amp``, ``gamma``, ``beta`` (spectral running),
        ``f0``, ``peak_freq``, and reduced chi-square ``chi2_dof``.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    if spec_std is None:
        spec_std = np.ones_like(spec)
    else:
        spec_std = np.asarray(spec_std, dtype=np.float64)
    valid = (
        (~np.isclose(spec_std, 0.0, atol=mask_tol)) & np.isfinite(spec) & (freqs > 0)
    )
    f_obs = freqs[valid]
    y_obs = spec[valid]
    s_obs = spec_std[valid]
    if f0 is None:
        f0 = float(np.exp(np.mean(np.log(f_obs))))
    x = f_obs / f0

    def model(x_val: np.ndarray, amp: float, gamma: float, beta: float) -> np.ndarray:
        lx = np.log(x_val)
        exponent = np.clip((gamma + beta * lx) * lx, -50.0, 50.0)
        return amp * np.exp(exponent)

    amp0 = float(max(np.percentile(y_obs, 90), np.std(y_obs), 1e-6))
    try:
        popt, _ = optimize.curve_fit(
            model,
            x,
            y_obs,
            p0=[amp0, 0.0, -5.0],
            sigma=s_obs,
            absolute_sigma=True,
            maxfev=5000,
        )
        amp, gamma, beta = float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError:
        amp, gamma, beta = amp0, 0.0, 0.0

    resid = (y_obs - model(x, amp, gamma, beta)) / s_obs
    dof = max(len(y_obs) - 3, 1)
    chi2_dof = float(np.sum(resid**2) / dof)
    if beta < -1e-6:
        peak_freq = float(f0 * np.exp(np.clip(-gamma / (2.0 * beta), -10.0, 10.0)))
    else:
        peak_freq = float("nan")
    return {
        "amp": amp,
        "gamma": gamma,
        "beta": beta,
        "f0": float(f0),
        "peak_freq": peak_freq,
        "chi2_dof": chi2_dof,
    }


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
