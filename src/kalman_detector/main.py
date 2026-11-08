from __future__ import annotations

import logging
from typing import ClassVar, Literal, NamedTuple

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.polynomial import Polynomial
from scipy import integrate, interpolate, optimize, special, stats
from scipy.linalg import eigvalsh_tridiagonal

from kalman_detector.core import kalman_filter_cached, prepare_gain_cache
from kalman_detector.utils import (
    effective_number_of_tests,
    fisher_logsf,
    get_snr_from_logsf,
    normalize_spectrum,
    sidak_logsf,
)

logger = logging.getLogger(__name__)

_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


class OffPulseAnchor(NamedTuple):
    """Empirical null moments of the Kalman score from off-pulse spectra.

    Attributes
    ----------
    n_off : int
        Number of off-pulse spectra.
    emp_mean, emp_std : np.ndarray
        Sample mean and std (ddof=1) of the off-pulse scores, per transit sigma.
    null_mean, null_std : np.ndarray
        Mean and std of the calibrated (analytic or MC) null, per transit sigma.
    """

    n_off: int
    emp_mean: np.ndarray
    emp_std: np.ndarray
    null_mean: np.ndarray
    null_std: np.ndarray

    @property
    def scale_ratio(self) -> np.ndarray:
        """``emp_std / null_std``; ~1 when the noise model holds."""
        return self.emp_std / self.null_std

    @property
    def mean_z(self) -> np.ndarray:
        """Standardized offset of the empirical mean, ``~N(0, 1)`` under the model."""
        return (self.emp_mean - self.null_mean) / (self.null_std / np.sqrt(self.n_off))

    def anchor(self, scores: np.ndarray) -> np.ndarray:
        """Map scores into the calibrated null frame by location-scale matching.

        ``K' = null_mean + (K - emp_mean) * null_std / (emp_std * sqrt(1 + 1/n))``;
        the ``sqrt(1 + 1/n)`` factor is the predictive inflation for the
        estimated mean.
        """
        inflate = np.sqrt(1 + 1 / self.n_off)
        return self.null_mean + (scores - self.emp_mean) * self.null_std / (
            self.emp_std * inflate
        )


class LocalizedResult(NamedTuple):
    """Subband-localized significance of a spectrum.

    See :meth:`KalmanDetector.get_localized_significance`.

    Attributes
    ----------
    chan_start, chan_end : int
        Selected subband ``[chan_start, chan_end)``.
    snr_box : float
        Inverse-variance weighted S/N of the subband mean.
    sig_box : float
        One-sided normal log p-value of ``snr_box``.
    sig_kalman : float
        Diffuse Kalman contrast log p-value inside the subband, best over the
        transit sigmas and corrected with the subband's ``n_eff``.
    sig_combined : float
        Fisher combination of ``sig_box`` and ``sig_kalman`` (independent for a
        fixed subband).
    n_subbands : int
        Number of searched subbands.
    sig_corrected : float
        ``sig_combined`` corrected for the subband search (Sidak/Bonferroni over
        ``n_subbands``).
    """

    chan_start: int
    chan_end: int
    snr_box: float
    sig_box: float
    sig_kalman: float
    sig_combined: float
    n_subbands: int
    sig_corrected: float

    @property
    def snr(self) -> float:
        """Gaussian-equivalent S/N of :attr:`sig_corrected`."""
        return get_snr_from_logsf(self.sig_corrected)


def subband_edges(
    nchans: int,
    min_subband_chans: int = 32,
    *,
    staggered: bool = True,
) -> list[tuple[int, int]]:
    """Dyadic (and half-shifted staggered) subbands ``[a, b)`` of a band.

    Level ``k`` splits the band into ``2**k`` near-equal parts, down to the
    finest level whose parts have at least ``min_subband_chans`` channels. The
    full band is level 0. With ``staggered``, level ``k >= 1`` also includes the
    ``2**k - 1`` subbands of the same size shifted by half a subband.
    """
    bands = []
    level = 0
    while nchans / 2**level >= min_subband_chans or level == 0:
        nb = 2**level
        edges = np.rint(np.linspace(0, nchans, nb + 1)).astype(int)
        bands.extend(zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True))
        if staggered and nb > 1:
            half = np.rint(np.linspace(0, nchans, 2 * nb + 1)).astype(int)
            bands.extend(
                zip(half[1:-2:2].tolist(), half[3::2].tolist(), strict=True),
            )
        level += 1
    return list(dict.fromkeys(bands))


class KalmanDetector:
    """Calculates the kalman significance for given 1d spectrum and noise.

    Parameters
    ----------
    spec_std : numpy.ndarray
        Estimated 1D per-channel standard deviation, presumably from the
        "off-pulse" region.
    q_par : :py:obj:`~numpy.typing.ArrayLike`, optional
        q parameter values for the transit sigma, by default None.
    mask_tol: float
        The absolute tolerance parameter to flag standard deviation values,
        by default 1e-5.
    diffuse : bool, optional
        Use the diffuse initial prior (1/v0 = 0) and score the mean-free contrast
        log likelihood ratio, by default True. If False, the legacy statistic
        (finite v0 on the weighted mean-subtracted spectrum) is used.

    Raises
    ------
    ValueError
        if all values of spec_std are zeros or negligible.

    Notes
    -----
    Zeros in the input spec_std will be flagged.
    """

    q_default: ClassVar[list[float]] = [3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

    def __init__(
        self,
        spec_std: np.ndarray,
        q_par: np.ndarray | list[float] | float | None = None,
        mask_tol: float = 1e-5,
        *,
        diffuse: bool = True,
    ) -> None:
        self._mask_tol = mask_tol
        # Check the values of per-channel standard deviation and mitigate any zero.
        self._mask = np.isclose(spec_std, 0, atol=mask_tol)
        if np.all(self._mask):
            msg = (
                "spectrum stds are all zeros or negligible."
                "Not preparing kalman significance distribution."
            )
            raise ValueError(msg)
        self._spec_std = np.asarray(spec_std, dtype=np.float64)
        self._diffuse = diffuse
        self._sig_ts = self._get_sig_ts(q_par)
        self._q_arr = (self._sig_ts / np.median(self._spec_std[~self._mask])) ** 2
        self._subdetectors: dict[tuple[int, int], KalmanDetector | None] = {}
        self._distributions: list | None = None
        self._caches: list | None = None
        self._n_eff: float | None = None
        self._method: str | None = None

    @property
    def mask_tol(self) -> float:
        """The absolute tolerance parameter to flag standard deviation values."""
        return self._mask_tol

    @property
    def mask(self) -> np.ndarray:
        """The mask of flagged standard deviation values."""
        return self._mask

    @property
    def spec_std(self) -> np.ndarray:
        """The per-channel standard deviation."""
        return self._spec_std

    @property
    def nchans(self) -> int:
        """The number of frequency channels."""
        return len(self.spec_std)

    @property
    def diffuse(self) -> bool:
        """Whether the diffuse initial prior (1/v0 = 0) is used."""
        return self._diffuse

    @property
    def method(self) -> str | None:
        """Calibration method used in :meth:`prepare_fits` (None if not prepared)."""
        return self._method

    @property
    def transit_sigmas(self) -> np.ndarray:
        """The trial transition std for the intrinsic markov process."""
        return self._sig_ts

    @property
    def distributions(self) -> list:
        """Null score distributions (one per transit sigma)."""
        if self._distributions is None:
            msg = "Distributions are not prepared. Call prepare_fits() first."
            raise RuntimeError(msg)
        return self._distributions

    @property
    def n_eff(self) -> float:
        """Effective number of independent trials across the transit sigmas.

        For ``method="saddlepoint"`` this is the Li & Ji (2005) estimate from the
        exact null correlation matrix of the scores. For ``method="mc"`` it is
        the conservative number of trials ``len(transit_sigmas)``.
        """
        if self._n_eff is None:
            msg = "n_eff is not prepared. Call prepare_fits() first."
            raise RuntimeError(msg)
        return self._n_eff

    def prepare_fits(
        self,
        ntrials: int = 10000,
        method: Literal["saddlepoint", "mc"] = "saddlepoint",
    ) -> None:
        """Prepare the null distributions of the kalman score.

        Parameters
        ----------
        ntrials : int, optional
            number of random instances to be used for ``method="mc"``,
            by default 10000. Ignored for ``method="saddlepoint"``.
        method : {"saddlepoint", "mc"}, optional
            ``"saddlepoint"`` (default) computes the exact null distribution
            analytically from the tridiagonal difference-space eigenvalues and
            evaluates its tail with the Lugannani-Rice saddlepoint formula
            (requires ``diffuse=True``). ``"mc"`` simulates ``ntrials`` gaussian
            noise spectra and fits a linear exponential tail (legacy).

        Raises
        ------
        ValueError
            if the method is unknown, or saddlepoint is requested for the legacy
            (non-diffuse) statistic.
        """
        with np.printoptions(precision=3, suppress=True):
            logger.debug(
                f"Preparing Kalman significance distribution ({method}) for "
                f"sig_ts {self.transit_sigmas}",
            )
        if method == "saddlepoint":
            if not self.diffuse:
                msg = (
                    "method='saddlepoint' calibrates the diffuse contrast statistic; "
                    "use diffuse=True or method='mc'."
                )
                raise ValueError(msg)
            mu = contrast_eigenvalues(self.spec_std, self.mask)
            distributions = [
                TridiagonalSaddlepointDistribution(mu, sig_eta)
                for sig_eta in self.transit_sigmas
            ]
            lambdas = np.array([dist.eigenvalues for dist in distributions])
            self._n_eff = effective_number_of_tests(_null_correlation(lambdas))
        elif method == "mc":
            # Pass the full channel array (masked channels have zero std) so that
            # the simulated null preserves gap lengths between unmasked channels.
            distributions = [
                KalmanDistribution(
                    self.spec_std,
                    sig_eta,
                    ntrials=ntrials,
                    mask_tol=self.mask_tol,
                    diffuse=self.diffuse,
                )
                for sig_eta in self.transit_sigmas
            ]
            self._n_eff = float(len(distributions))
        else:
            msg = f"Unknown method {method!r}. Use 'saddlepoint' or 'mc'."
            raise ValueError(msg)
        self._distributions = distributions
        self._caches = [
            prepare_gain_cache(
                self.spec_std,
                sig_eta,
                chan_mask=self.mask,
                diffuse=self.diffuse,
            )
            for sig_eta in self.transit_sigmas
        ]
        self._method = method

    def get_best_significance(
        self,
        spec: np.ndarray,
        off_pulse_specs: np.ndarray | None = None,
    ) -> float:
        """Calculate the best kalman significance of 1d spectrum.

        Parameters
        ----------
        spec : numpy.ndarray
            1d spectrum data
        off_pulse_specs : numpy.ndarray, optional
            2d array ``(n_off, nchans)`` of off-pulse spectra used to anchor the
            null (see :meth:`get_significance`), by default None.

        Returns
        -------
        float
            The best kalman significance in logsf units (not corrected for the
            selection over transit sigmas; see :attr:`n_eff`).
        """
        sigs, _scores = self.get_significance(spec, off_pulse_specs)
        return np.min(sigs)

    def get_significance(
        self,
        spec: np.ndarray,
        off_pulse_specs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the kalman score and significance of 1d spectrum.

        Parameters
        ----------
        spec : numpy.ndarray
            1d spectrum data
        off_pulse_specs : numpy.ndarray, optional
            2d array ``(n_off, nchans)`` of off-pulse spectra (e.g. 30-100
            spectra from the waterfall around the candidate), by default None.
            If given, each score is mapped into the calibrated null frame by
            matching the empirical mean and std of the off-pulse scores (see
            :meth:`off_pulse_anchor`) before its significance is evaluated.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Kalman significance (logsf units) and the (un-anchored) kalman score
            for each transit sigma.

        Notes
        -----
        Anchoring is a robustness check against an imperfect noise model
        (baseline drifts, channel covariance, misestimated ``spec_std``), not a
        replacement for the analytic null. With ``n_off`` spectra the empirical
        std carries a relative error of about ``1 / sqrt(2 n_off)`` (~10% for
        50 spectra), which moves a 6 sigma significance by roughly 0.5 sigma.
        Compare :attr:`OffPulseAnchor.scale_ratio` and
        :attr:`OffPulseAnchor.mean_z` with their nominal values (1 and N(0, 1))
        to judge whether anchoring is needed.
        """
        spec_in = self._prepare_spec(spec)
        scores = self._scores(spec_in)
        anchored = scores
        if off_pulse_specs is not None:
            anchored = self.off_pulse_anchor(off_pulse_specs).anchor(scores)
        significances = np.array(
            [
                dist.logsf(score)
                for dist, score in zip(self.distributions, anchored, strict=True)
            ],
        )
        return significances, scores

    def off_pulse_anchor(self, off_pulse_specs: np.ndarray) -> OffPulseAnchor:
        """Empirical null moments of the score from off-pulse spectra.

        Parameters
        ----------
        off_pulse_specs : numpy.ndarray
            2d array ``(n_off, nchans)`` of off-pulse spectra, ``n_off >= 2``.

        Returns
        -------
        OffPulseAnchor
            Empirical and calibrated null moments per transit sigma.
        """
        off = np.atleast_2d(np.asarray(off_pulse_specs, dtype=np.float64))
        if off.ndim != 2 or off.shape[1] != self.nchans:
            msg = f"off_pulse_specs must have shape (n_off, {self.nchans})."
            raise ValueError(msg)
        n_off = off.shape[0]
        if n_off < 2:
            msg = "Need at least 2 off-pulse spectra."
            raise ValueError(msg)
        if not self.diffuse:
            off = np.array([self._prepare_spec(row) for row in off])
        off_scores = np.array(
            [kalman_filter_cached(off, cache) for cache in self._caches_checked()],
        )
        null = np.array([_null_moments(dist) for dist in self.distributions])
        return OffPulseAnchor(
            n_off=n_off,
            emp_mean=off_scores.mean(axis=1),
            emp_std=off_scores.std(axis=1, ddof=1),
            null_mean=null[:, 0],
            null_std=null[:, 1],
        )

    def get_localized_significance(
        self,
        spec: np.ndarray,
        min_subband_chans: int = 32,
        *,
        staggered: bool = True,
    ) -> LocalizedResult:
        """Significance of the best subband for narrowband bursts.

        The subband ``[a*, b*)`` with the largest positive inverse-variance
        weighted S/N is selected among the dyadic (and staggered) subbands of
        :func:`subband_edges`, including the full band. Inside it, the subband
        S/N and the diffuse Kalman contrast are combined with Fisher's method
        (for a fixed subband the two are exactly independent under gaussian
        noise, because the contrast is mean-free), and the result is corrected
        for the number of searched subbands. The correction bounds the
        probability of the minimum p-value over all subbands, so it is valid
        (conservative) for this data-dependent selection.

        Parameters
        ----------
        spec : numpy.ndarray
            1d spectrum data
        min_subband_chans : int, optional
            Smallest subband size, by default 32.
        staggered : bool, optional
            Include half-shifted subbands, by default True.

        Returns
        -------
        LocalizedResult
            The selected subband and its significances.

        Notes
        -----
        Requires the diffuse statistic. The Kalman nulls of the subbands are
        calibrated lazily (saddlepoint) and cached.
        """
        if not self.diffuse:
            msg = "get_localized_significance requires diffuse=True."
            raise ValueError(msg)
        spec_in = self._prepare_spec(spec)
        safe_std = np.where(self.mask, 1.0, self.spec_std)
        weights = np.where(self.mask, 0.0, 1 / safe_std**2)
        info = np.where(self.mask, 0.0, weights * spec_in)
        cum_w = np.concatenate(([0.0], np.cumsum(weights)))
        cum_h = np.concatenate(([0.0], np.cumsum(info)))
        cum_n = np.concatenate(([0], np.cumsum(~self.mask)))
        min_obs = 3
        edges = subband_edges(self.nchans, min_subband_chans, staggered=staggered)
        bands = [(a, b) for a, b in edges if cum_n[b] - cum_n[a] >= min_obs]
        snrs = np.array(
            [(cum_h[b] - cum_h[a]) / np.sqrt(cum_w[b] - cum_w[a]) for a, b in bands],
        )
        ibest = int(np.argmax(snrs))
        a, b = bands[ibest]
        sub = self._subdetector(a, b)
        sig_box = float(stats.norm.logsf(snrs[ibest]))
        sig_k = sidak_logsf(sub.get_best_significance(spec_in[a:b]), sub.n_eff)
        sig_comb = fisher_logsf(sig_box, sig_k)
        return LocalizedResult(
            chan_start=a,
            chan_end=b,
            snr_box=float(snrs[ibest]),
            sig_box=sig_box,
            sig_kalman=sig_k,
            sig_combined=sig_comb,
            n_subbands=len(bands),
            sig_corrected=sidak_logsf(sig_comb, len(bands)),
        )

    def _subdetector(self, start: int, end: int) -> KalmanDetector:
        key = (start, end)
        if key not in self._subdetectors:
            std = np.where(self.mask[start:end], 0.0, self.spec_std[start:end])
            sub = KalmanDetector(std, q_par=self._q_arr, mask_tol=self.mask_tol)
            sub.prepare_fits(method="saddlepoint")
            self._subdetectors[key] = sub
        return self._subdetectors[key]

    def _prepare_spec(self, spec: np.ndarray) -> np.ndarray:
        if len(spec) != self.nchans:
            msg = (
                f"Length of signal spectrum ({len(spec)}) is not equal "
                f"to the noise variance ({self.nchans})."
            )
            raise ValueError(msg)
        if self.diffuse:
            # The diffuse contrast score is exactly invariant to constant offsets.
            return np.asarray(spec, dtype=np.float64)
        return normalize_spectrum(spec, self.spec_std, chan_mask=self.mask)

    def _caches_checked(self) -> list:
        _ = self.distributions  # raises if not prepared
        return self._caches

    def _scores(self, spec_in: np.ndarray) -> np.ndarray:
        return np.array(
            [kalman_filter_cached(spec_in, cache) for cache in self._caches_checked()],
        )

    def _get_sig_ts(
        self,
        q_par: np.ndarray | list[float] | float | None = None,
    ) -> np.ndarray:
        if q_par is None:
            q_arr = np.array(self.q_default)
        elif isinstance(q_par, (np.ndarray, list)):
            q_arr = np.array(q_par, dtype=float)
        else:
            q_arr = np.array([q_par], dtype=float)
        # Scale by the typical noise of the unmasked channels only.
        return np.sqrt(np.median(self.spec_std[~self.mask]) ** 2 * q_arr)


def _null_moments(dist: object) -> tuple[float, float]:
    """Mean and std of a calibrated null distribution."""
    if isinstance(dist, QuadraticFormTail):
        return dist.mean, float(np.sqrt(dist.var))
    scores = np.asarray(dist.scores)
    return float(scores.mean()), float(scores.std(ddof=1))


def contrast_eigenvalues(
    spec_std: np.ndarray,
    chan_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Generalized eigenvalues of the difference-space noise covariance.

    For unmasked channel indices ``idx`` with gaps ``g = diff(idx)``, the
    adjacent differences ``d_i = x[idx[i+1]] - x[idx[i]]`` scaled by
    ``1/sqrt(g_i)`` have a symmetric tridiagonal noise covariance and a
    random-walk signal covariance ``q * I``. Its eigenvalues ``mu_j`` give the
    null distribution of the diffuse contrast score for every ``q`` at once.

    Parameters
    ----------
    spec_std : np.ndarray
        Per-channel noise standard deviation.
    chan_mask : np.ndarray, optional
        Mask of channels to ignore, by default None.

    Returns
    -------
    np.ndarray
        The ``N_obs - 1`` eigenvalues ``mu_j`` (ascending).
    """
    spec_std = np.asarray(spec_std, dtype=np.float64)
    if chan_mask is None:
        chan_mask = np.zeros(len(spec_std), dtype=bool)
    idx = np.flatnonzero(~np.asarray(chan_mask, dtype=bool))
    if len(idx) < 2:
        return np.empty(0)
    r_obs = spec_std[idx] ** 2
    gaps = np.diff(idx).astype(np.float64)
    diag = (r_obs[:-1] + r_obs[1:]) / gaps
    if len(diag) == 1:
        return diag
    off = -r_obs[1:-1] / np.sqrt(gaps[:-1] * gaps[1:])
    return eigvalsh_tridiagonal(diag, off, lapack_driver="sterf")


def _null_correlation(lambdas: np.ndarray) -> np.ndarray:
    """Null correlation of ``K_q = c_q + 1/2 sum_j lambda_j(q) Z_j^2`` across q."""
    cov = 0.5 * lambdas @ lambdas.T
    norm = np.sqrt(np.diag(cov))
    norm[norm == 0] = 1
    return cov / np.outer(norm, norm)


class QuadraticFormTail:
    """Survival function of ``K = const + 1/2 * sum_j lambda_j Z_j**2``.

    The tail is evaluated with the Lugannani-Rice saddlepoint approximation
    in log space (accurate far into the tail, e.g. 1e-300), with Imhof's exact
    numerical inversion in a small window around the mean where the
    Lugannani-Rice formula is singular. Results are tabulated on a grid
    parametrized by the saddlepoint ``t`` (no root finding) and interpolated
    with a monotone cubic (PCHIP) in log survival probability.

    Parameters
    ----------
    lambdas : np.ndarray
        Non-negative weights ``lambda_j``.
    const : float, optional
        Constant offset of the score, by default 0.
    n_grid : int, optional
        Number of saddlepoint grid points, by default 160.
    imhof_window : float, optional
        Half width (in standardized ``w`` units) around the mean replaced by
        Imhof integration, by default 0.2.
    """

    def __init__(
        self,
        lambdas: np.ndarray,
        const: float = 0.0,
        *,
        n_grid: int = 160,
        imhof_window: float = 0.2,
    ) -> None:
        lam = np.asarray(lambdas, dtype=np.float64)
        if np.any(lam < 0):
            msg = "lambdas must be non-negative."
            raise ValueError(msg)
        self._lambdas = lam
        self._lam = lam[lam > 0]
        self._const = float(const)
        self._n_grid = n_grid
        self._imhof_window = imhof_window
        self._build_table()

    @property
    def eigenvalues(self) -> np.ndarray:
        """Weights ``lambda_j`` of the quadratic form."""
        return self._lambdas

    @property
    def const(self) -> float:
        """Constant offset of the score."""
        return self._const

    @property
    def mean(self) -> float:
        """Null mean of the score."""
        return self._const + 0.5 * float(np.sum(self._lam))

    @property
    def var(self) -> float:
        """Null variance of the score."""
        return 0.5 * float(np.sum(self._lam**2))

    def logsf(self, score: float | np.ndarray) -> float | np.ndarray:
        """Natural log of the survival function ``P(K >= score)`` (tabulated).

        Parameters
        ----------
        score : float | np.ndarray
            Score value(s).

        Returns
        -------
        float | np.ndarray
            log survival probability (<= 0).
        """
        q_val = np.asarray(score, dtype=np.float64) - self._const
        out = np.zeros_like(q_val)
        if self._lam.size == 0:
            return out if out.ndim else float(out)
        mid = (q_val > 0) & (q_val <= self._x_max)
        out[mid] = self._interp(q_val[mid])
        high = q_val > self._x_max
        out[high] = self._logsf_max + self._slope_max * (q_val[high] - self._x_max)
        out = np.minimum(out, 0)
        return out if out.ndim else float(out)

    def sf(self, score: float | np.ndarray) -> float | np.ndarray:
        """Survival function ``P(K >= score)``."""
        return np.exp(self.logsf(score))

    def saddlepoint_logsf(self, score: float) -> float:
        """Direct (untabulated) Lugannani-Rice log survival probability."""
        q_val = float(score) - self._const
        if q_val <= 0 or self._lam.size == 0:
            return 0.0
        lam_max = self._lam.max()
        rho = self._lam / lam_max
        half_sum = 0.5 * np.sum(self._lam)
        if q_val >= half_sum:

            def one_minus(s: float) -> np.ndarray:
                return (1 - rho) + rho * np.exp(-s)

            def t_of(s: float) -> float:
                return -np.expm1(-s) / lam_max

            s_hi = 1.0
            while self._kprime(one_minus(s_hi)) < q_val and s_hi < 700:
                s_hi *= 2
        else:

            def one_minus(s: float) -> np.ndarray:
                return 1 + rho * s

            def t_of(s: float) -> float:
                return -s / lam_max

            s_hi = 1.0
            while self._kprime(one_minus(s_hi)) > q_val and s_hi < 1e300:
                s_hi *= 2
        s_root = optimize.brentq(
            lambda s: self._kprime(one_minus(s)) - q_val,
            0.0,
            s_hi,
            xtol=1e-300,
            rtol=4 * np.finfo(float).eps,
        )
        om = one_minus(s_root)[None, :]
        _, logsf, w_hat = self._lugannani_rice(np.array([t_of(s_root)]), om)
        if abs(w_hat[0]) <= self._imhof_window:
            # Lugannani-Rice is singular at the mean: use Imhof inversion.
            return float(np.log(max(self.imhof_sf(score), 1e-300)))
        return float(min(logsf[0], 0.0))

    def imhof_sf(
        self,
        score: float,
        epsabs: float = 1e-14,
        epsrel: float = 1e-10,
        limit: int = 2000,
    ) -> float:
        """Exact survival probability via Imhof (1961) numerical inversion.

        Absolute accuracy is limited by ``epsabs``; use for moderate tails only.
        """
        q_val = float(score) - self._const
        if self._lam.size == 0:
            return float(q_val <= 0)
        a = 0.5 * self._lam

        def integrand(u: float) -> float:
            if u == 0:
                return 0.5 * (np.sum(a) - q_val)
            theta = 0.5 * np.sum(np.arctan(a * u)) - 0.5 * q_val * u
            log_rho = 0.25 * np.sum(np.log1p((a * u) ** 2))
            return np.sin(theta) * np.exp(-log_rho) / u

        val, _ = integrate.quad(
            integrand,
            0,
            np.inf,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        return float(np.clip(0.5 + val / np.pi, 0, 1))

    def _kprime(self, one_minus: np.ndarray) -> float:
        return 0.5 * float(np.sum(self._lam / one_minus))

    def _lugannani_rice(
        self,
        t: np.ndarray,
        one_minus: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, logsf, w) at saddlepoints ``t`` with ``one_minus = 1 - t*lam``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._lugannani_rice_unsafe(t, one_minus)

    def _lugannani_rice_unsafe(
        self,
        t: np.ndarray,
        one_minus: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ratio = self._lam / one_minus
        x = 0.5 * np.sum(ratio, axis=1)
        kappa = -0.5 * np.sum(np.log(one_minus), axis=1)
        kpp = 0.5 * np.sum(ratio**2, axis=1)
        w = np.sign(t) * np.sqrt(np.maximum(2 * (t * x - kappa), 0))
        u = t * np.sqrt(kpp)
        logsf = np.empty_like(t)
        pos = w > 0
        wp, up = w[pos], u[pos]
        mills = np.sqrt(np.pi / 2) * special.erfcx(wp / np.sqrt(2))
        logsf[pos] = -0.5 * wp**2 - _LOG_SQRT_2PI + np.log(mills + 1 / up - 1 / wp)
        neg = ~pos
        wn, un = w[neg], u[neg]
        sf_neg = special.ndtr(-wn) + np.exp(-0.5 * wn**2 - _LOG_SQRT_2PI) * (
            1 / un - 1 / wn
        )
        logsf[neg] = np.log(np.clip(sf_neg, 1e-300, 1))
        return x, logsf, w

    def _build_table(self) -> None:
        if self._lam.size == 0:
            return
        lam_max = self._lam.max()
        rho = self._lam / lam_max
        n_neg = self._n_grid // 3
        n_pos = self._n_grid - n_neg
        # Upper tail: t = (1 - e^{-s}) / lam_max, 1 - t*lam computed without
        # cancellation as (1 - rho) + rho e^{-s}.
        s_pos = np.geomspace(1e-3, 60, n_pos)
        t_pos = -np.expm1(-s_pos) / lam_max
        om_pos = (1 - rho)[None, :] + rho[None, :] * np.exp(-s_pos)[:, None]
        # Lower tail: t = -v / lam_max.
        v_neg = np.geomspace(1e-3, 1e6, n_neg)
        t_neg = -v_neg / lam_max
        om_neg = 1 + rho[None, :] * v_neg[:, None]
        t_all = np.concatenate([t_neg, t_pos])
        om_all = np.concatenate([om_neg, om_pos])
        x_all, logsf_all, w_all = self._lugannani_rice(t_all, om_all)
        keep = np.abs(w_all) > self._imhof_window
        x_grid = list(x_all[keep])
        logsf_grid = list(logsf_all[keep])
        # Imhof at the mean, where Lugannani-Rice has a removable singularity.
        # SF(mean) is O(1), so a modest absolute tolerance is sufficient.
        q_mean = 0.5 * float(np.sum(self._lam))
        sf_im = self.imhof_sf(q_mean + self._const, epsabs=1e-9, epsrel=1e-7)
        x_grid.append(q_mean)
        logsf_grid.append(np.log(max(sf_im, 1e-300)))
        x_grid.append(0.0)
        logsf_grid.append(0.0)
        order = np.argsort(x_grid)
        x_arr = np.asarray(x_grid)[order]
        logsf_arr = np.minimum.accumulate(np.minimum(np.asarray(logsf_grid)[order], 0))
        x_arr, uniq = np.unique(x_arr, return_index=True)
        logsf_arr = logsf_arr[uniq]
        self._interp = interpolate.PchipInterpolator(
            x_arr, logsf_arr, extrapolate=False
        )
        self._x_max = float(x_arr[-1])
        self._logsf_max = float(logsf_arr[-1])
        self._slope_max = -float(t_pos[-1])


class TridiagonalSaddlepointDistribution(QuadraticFormTail):
    """Analytic null distribution of the diffuse Kalman contrast score.

    Under the gaussian null,
    ``K_q = c_q + 1/2 sum_j lambda_j(q) Z_j**2`` with
    ``lambda_j(q) = q / (q + mu_j)`` and
    ``c_q = -1/2 sum_j log(1 + q / mu_j)``, where ``mu_j`` are the
    difference-space eigenvalues from :func:`contrast_eigenvalues` and
    ``q = sig_eta**2``.

    Parameters
    ----------
    mu : np.ndarray
        Difference-space eigenvalues (from :func:`contrast_eigenvalues`).
    sig_eta : float
        State transition std for the intrinsic markov process.
    **kwargs
        Passed to :class:`QuadraticFormTail`.
    """

    def __init__(self, mu: np.ndarray, sig_eta: float, **kwargs: float) -> None:
        mu = np.asarray(mu, dtype=np.float64)
        q_var = sig_eta**2
        self._sig_eta = sig_eta
        self._mu = mu
        lambdas = q_var / (q_var + mu)
        const = -0.5 * float(np.sum(np.log1p(q_var / mu)))
        super().__init__(lambdas, const, **kwargs)

    @property
    def sig_eta(self) -> float:
        """State transition std for the intrinsic markov process."""
        return self._sig_eta

    @property
    def nchans(self) -> int:
        """Number of unmasked frequency channels."""
        return len(self._mu) + 1

    def __str__(self) -> str:
        return (
            f"TridiagonalSaddlepointDistribution(sig_eta={self.sig_eta:.3f}, "
            f"nobs={self.nchans})"
        )

    def __repr__(self) -> str:
        return str(self)


class KalmanDistribution:
    """Generate a monte-carlo Kalman score distribution for a given transition std.

    Parameters
    ----------
    sigma_arr : numpy.ndarray
        Measurement noise std per frequency channel. Channels with zero
        (negligible) std are treated as masked, preserving their position.
    sig_eta : float
        State transition std for the intrinsic markov process.
    ntrials : int, optional
        number of gaussian noise instances to be used, by default 10000.
    mask_tol : float, optional
        The absolute tolerance to flag standard deviation values, by default 1e-5.
    diffuse : bool, optional
        Simulate the diffuse contrast score instead of the legacy statistic,
        by default False.

    Raises
    ------
    ValueError
        if all sigma_arr values are zero or negligible.
    """

    def __init__(
        self,
        sigma_arr: np.ndarray,
        sig_eta: float,
        ntrials: int = 10000,
        mask_tol: float = 1e-5,
        *,
        diffuse: bool = False,
    ) -> None:
        self._sigma_arr = np.asarray(sigma_arr, dtype=np.float64)
        self._sig_eta = sig_eta
        self._ntrials = ntrials
        self._diffuse = diffuse
        # Check the values of per-channel standard deviation and mitigate any zero.
        self._mask_tol = mask_tol
        self._mask = np.isclose(sigma_arr, 0, atol=mask_tol)
        if np.all(self._mask):
            msg = (
                "sigma_arr are all zeros or negligible."
                "Not preparing kalman significance distribution."
            )
            raise ValueError(msg)

        self._generate()
        self._fit_distribution()

    @property
    def mask_tol(self) -> float:
        """The absolute tolerance parameter to flag standard deviation values."""
        return self._mask_tol

    @property
    def mask(self) -> np.ndarray:
        """Mask for the negligible standard deviation values."""
        return self._mask

    @property
    def sigma_arr(self) -> np.ndarray:
        """Per-channel standard deviation."""
        return self._sigma_arr

    @property
    def sig_eta(self) -> float:
        """State transition std for the intrinsic markov process."""
        return self._sig_eta

    @property
    def ntrials(self) -> int:
        """Number of gaussian noise instances to be used."""
        return self._ntrials

    @property
    def diffuse(self) -> bool:
        """Whether the diffuse contrast score is simulated."""
        return self._diffuse

    @property
    def nchans(self) -> int:
        """Number of frequency channels."""
        return len(self.sigma_arr)

    @property
    def scores(self) -> np.ndarray:
        """Kalman scores for the gaussian noise instances."""
        return self._scores

    @property
    def theoretical_quantiles(self) -> np.ndarray:
        """Theoretical quantiles for the kalman scores fit."""
        return np.arange(3, 10, 0.2)

    @property
    def sample_quantiles(self) -> np.ndarray:
        """Sample quantiles for the kalman scores fit with a scaled exponential."""
        return np.percentile(
            self.scores,
            100 * (1 - 2.0 ** (-self.theoretical_quantiles)),
        )

    @property
    def polyfit(self) -> Polynomial:
        """Polynomial fit to the tail of the kalman scores distribution."""
        return self._polyfit

    def logsf(self, score: float | np.ndarray) -> float | np.ndarray:
        """Natural log survival probability from the fitted exponential tail."""
        return -np.log(2) * np.maximum(0, self.polyfit(score))

    def plot_diagnostic(
        self,
        bins: int = 30,
        figsize: tuple[float, float] = (13, 5.5),
        dpi: int = 100,
        *,
        logy: bool = False,
        outfile: str | None = None,
    ) -> plt.Figure:
        """Make a plot of the distribution."""
        fig = plt.figure(figsize=figsize, dpi=dpi)
        grid = fig.add_gridspec(nrows=1, ncols=2)
        grid.update(left=0.07, right=0.98, bottom=0.1, top=0.95, wspace=0.12)

        ax_hist = plt.subplot(grid[0, 0])
        axins = inset_axes(ax_hist, width="35%", height="30%")
        ax_qq = plt.subplot(grid[0, 1])

        cutoff = self.sample_quantiles[0]
        ax_hist.hist(
            self.scores,
            bins=bins,
            density=True,
            histtype="step",
            ec="tab:blue",
            lw=2,
        )
        ax_hist.axvline(cutoff, color="tab:red", lw=1.5)
        ax_hist.set_xlabel("Kalman Score")
        ax_hist.set_ylabel("Probability")

        tail_samples = self.scores[self.scores > cutoff]
        axins.hist(
            tail_samples,
            bins=bins,
            density=True,
            histtype="step",
            ec="tab:blue",
            lw=2,
        )
        axins.set_xlabel("Kalman Score tail end")
        axins.set_ylabel("Probability")
        if logy:
            axins.set_yscale("log")

        ax_qq.plot(
            self.theoretical_quantiles,
            self.sample_quantiles,
            "o",
            ms=8,
            mec="tab:blue",
            mfc="white",
        )
        ax_qq.plot(
            self.polyfit(self.sample_quantiles),
            self.sample_quantiles,
            "tab:red",
            lw=1.5,
        )
        ax_qq.set_xlabel("Theoretical Quantiles (exponential)")
        ax_qq.set_ylabel("Sample Quantiles")

        fig.suptitle(str(self))
        if outfile is not None:
            plt.savefig(outfile, dpi=dpi)
        return fig

    def __str__(self) -> str:
        return (
            f"KalmanDistribution(sig_eta={self.sig_eta:.3f}, "
            f"ntrials={self.ntrials}, nchans={self.nchans})"
        )

    def __repr__(self) -> str:
        return str(self)

    def _generate(self, chunk: int = 2048) -> None:
        cache = prepare_gain_cache(
            self.sigma_arr,
            self.sig_eta,
            chan_mask=self.mask,
            diffuse=self.diffuse,
        )
        weights = np.where(
            self.mask, 0.0, 1 / np.where(self.mask, 1, self.sigma_arr) ** 2
        )
        scores = np.empty(self.ntrials)
        rng = np.random.default_rng()
        for start in range(0, self.ntrials, chunk):
            nrow = min(chunk, self.ntrials - start)
            random_specs = rng.normal(0, self.sigma_arr, size=(nrow, self.nchans))
            if not self.diffuse:
                # Vectorized normalize_spectrum (weighted mean over unmasked).
                spec_mean = random_specs @ weights / np.sum(weights)
                random_specs -= spec_mean[:, None]
            scores[start : start + nrow] = kalman_filter_cached(random_specs, cache)
        self._scores = scores

    def _fit_distribution(self) -> None:
        """Approximating the tail of the distribution as an exponential tail."""
        self._polyfit = Polynomial.fit(
            self.sample_quantiles,
            self.theoretical_quantiles,
            1,
        )


def secondary_spectrum_cumulative_chi2_score(
    spec: np.ndarray,
    spec_std: np.ndarray,
    mask_tol: float = 1e-5,
) -> float:
    """Compute the cumulative-chi2 test statistic on sig.

    Parameters
    ----------
    spec : numpy.ndarray
        1d array of the observed spectrum.
    spec_std : numpy.ndarray
        1D array of the standard deviation of the observed spectrum.
    mask_tol : float, optional
        The absolute tolerance to flag standard deviation values, by default 1e-5.

    Returns
    -------
    float
        Best significance in logsf units (max_f0 of ln(P(sig|H1,ff0)/P(sig|H0)))

    Notes
    -----
    Would compute the cumulative-chi2 test statistic on sig.
    Assumes the signal is composed of i.i.d N(E,1) variables ($E$ would be ignored).
    Would return the following statistical test between:
    H0: sig(f) ~ N(E,1)
    H1: sig(f) ~ N(E,1) + FRB with A(f) that have secondary spectrum (DFT(A(f))
    with freq cutoff ff0
    """
    signal = np.divide(
        spec,
        spec_std,
        out=np.zeros_like(spec),
        where=~np.isclose(spec_std, 0, atol=mask_tol),
    )
    fft_sig = np.abs(np.fft.rfft(signal)) ** 2
    score_arr = np.cumsum(fft_sig[1:] / (len(signal) / 2))
    dof_arr = 2 * np.arange(1, len(score_arr) + 1)
    significance_arr = stats.chi2.logsf(score_arr, dof_arr)
    return np.min(significance_arr)
