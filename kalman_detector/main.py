from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.polynomial import Polynomial
from scipy import stats

from kalman_detector.core import kalman_filter
from kalman_detector.utils import normalize_spectrum

logger = logging.getLogger(__name__)


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
        self._spec_std = spec_std
        self._sig_ts = self._get_sig_ts(q_par)
        self._polyfits = None

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
    def transit_sigmas(self) -> np.ndarray:
        """The trial transition std for the intrinsic markov process."""
        return self._sig_ts

    @property
    def distributions(self) -> list:
        """Polynomial fits for the tail distribution of the kalman detector."""
        return self._distributions

    def prepare_fits(self, ntrials: int = 10000) -> None:
        """Prepare the polynomial fits for the tail distribution of the kalman detector.

        Measure kalman significance distribution in pure gaussian noise random data.
        For each individual transit_sigma, prepare polynomial fit of the exponential
        tail of the distribution.

        Parameters
        ----------
        ntrials : int, optional
            number of random instances to be used, by default 10000
        """
        spec_std = self.spec_std[~self.mask]
        with np.printoptions(precision=3, suppress=True):
            logger.debug(
                f"Measuring Kalman significance distribution for "
                f"sig_ts {self.transit_sigmas}",
            )
        distributions = []
        for transit_sigma in self.transit_sigmas:
            dist = KalmanDistribution(
                spec_std,
                transit_sigma,
                ntrials=ntrials,
                mask_tol=self.mask_tol,
            )
            distributions.append(dist)
        self._distributions = distributions

    def get_best_significance(self, spec: np.ndarray) -> float:
        """Calculate the best kalman significance of 1d spectrum.

        Parameters
        ----------
        spec : numpy.ndarray
            1d spectrum data

        Returns
        -------
        float
            The best kalman significance in logsf units.
        """
        sigs, scores = self.get_significance(spec)
        return np.min(sigs)

    def get_significance(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the kalman score and significance of 1d spectrum.

        Parameters
        ----------
        spec : numpy.ndarray
            1d spectrum data

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Kalman significance and the kalman score for each transit sigma.
        """
        if len(spec) != self.nchans:
            msg = (
                f"Length of signal spectrum ({len(spec)}) is not equal "
                f"to the noise variance ({self.nchans})."
            )
            raise ValueError(msg)
        scores = np.empty(len(self.transit_sigmas))
        significances = np.empty(len(self.transit_sigmas))
        for ii, transit_sigma in enumerate(self.transit_sigmas):
            norm_spec = normalize_spectrum(spec, self.spec_std, chan_mask=self.mask)
            score = kalman_filter(
                norm_spec,
                self.spec_std,
                transit_sigma,
                chan_mask=self.mask,
            )
            dist = self.distributions[ii]
            scores[ii] = score
            significances[ii] = max(0, dist.polyfit(score))
        significances *= -np.log(2)  # convert to logsf units
        return significances, scores

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
        return np.sqrt(np.median(self.spec_std) ** 2 * q_arr)


class KalmanDistribution:
    """Generate a monte-carlo Kalman score distribution for a given transition std.

    Parameters
    ----------
    sigma_arr : numpy.ndarray
        Measurement noise std per frequency channel.
    sig_eta : float
        State transition std for the intrinsic markov process.
    ntrials : int, optional
        number of gaussian noise instances to be used, by default 10000.
    mask_tol : float, optional
        The absolute tolerance to flag standard deviation values, by default 1e-5.

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
    ) -> None:
        self._sigma_arr = sigma_arr
        self._sig_eta = sig_eta
        self._ntrials = ntrials
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

    def _generate(self) -> None:
        scores = np.zeros(self.ntrials)
        rng = np.random.default_rng()
        for itrial in range(self.ntrials):
            random_spec = rng.normal(0, self.sigma_arr, size=self.nchans)
            norm_random_spec = normalize_spectrum(
                random_spec,
                self.sigma_arr,
                chan_mask=self.mask,
            )
            scores[itrial] = kalman_filter(
                norm_random_spec,
                self.sigma_arr,
                self.sig_eta,
                chan_mask=self.mask,
            )
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
