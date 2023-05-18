from __future__ import annotations
import numpy as np

from scipy import stats
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from kalman_detector.core import kalman_filter
from kalman_detector.utils import normalize_spectrum


class KalmanDetector(object):
    """Calculates the kalman significance for given 1d spec and per-channel error.

    Parameters
    ----------
    spec_std : numpy.ndarray
        1d numpy array of the estimated per-channel standard deviation, presumably from the "off-pulse" region.
    sig_ts : :py:obj:`~numpy.typing.ArrayLike`, optional
        List of trial transition standard deviations for the intrinsic markov process to be used, by default None.
    mask_tol: float
        The absolute tolerance parameter to flag standard deviation values, by default 1e-5.

    Raises
    ------
    ValueError
        if all values of spec_std are zeros or negligible.
    ValueError
        if sig_ts is not None and is not a list or numpy array.

    Notes
    -----
    Zeros in the input spec_std will be flagged. Best for zeros to be similar to those
    of real spectrum to be used.
    """

    q_default = [3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]

    def __init__(
        self,
        spec_std: np.ndarray,
        sig_ts: float | list | None = None,
        mask_tol: float = 1e-5,
    ) -> None:
        self._mask_tol = mask_tol
        # Check the values of per-channel standard deviation and mitigate any zero.
        self._mask = np.isclose(spec_std, 0, atol=mask_tol)
        if np.all(self._mask):
            raise ValueError(
                "spectrum stds are all zeros or negligible. Not preparing kalman significance distribution."
            )
        self._spec_std = spec_std
        self._sig_ts = self._get_sig_ts(sig_ts)
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
        """The trial transition standard deviations for the intrinsic markov process."""
        return self._sig_ts

    @property
    def distributions(self) -> list:
        """Polynomial fits for the tail distribution of the kalman detector."""
        return self._distributions

    def prepare_fits(self, ntrials: int = 10000) -> None:
        """Prepares the polynomial fits for the tail distribution of the kalman detector.

        Measure kalman significance distribution in pure gaussian noise random data.
        For each individual sig_t, prepare polynomial fit of the exponential tail of
        the distribution.

        Parameters
        ----------
        ntrials : int, optional
            number of random instances to be used, by default 10000
        """
        spec_std = self.spec_std[~self.mask]
        with np.printoptions(precision=3, suppress=True):
            print(
                f"Measuring Kalman significance distribution for sig_ts {self.transit_sigmas}"
            )
        distributions = []
        for transit_sigma in self.transit_sigmas:
            dist = KalmanDistribution(
                spec_std, transit_sigma, ntrials=ntrials, mask_tol=self.mask_tol
            )
            distributions.append(dist)
        self._distributions = distributions

    def get_significance(self, spec: np.ndarray) -> float:
        """Calculates the kalman significance of 1d spectrum in the background noise.

        Parameters
        ----------
        spec : numpy.ndarray
            1d numpy array with the spectrum of the candidate burst.

        Returns
        -------
        float
            The kalman significance of the candidate burst.
        """
        if len(spec) != self.nchans:
            raise ValueError(
                f"Length of signal spectrum {len(spec)} is not equal to the noise variance {self.nchans}."
            )
        significances = []
        for ii, transit_sigma in enumerate(self.transit_sigmas):
            norm_spec = normalize_spectrum(spec, self.spec_std, chan_mask=self.mask)
            score = kalman_filter(
                norm_spec, self.spec_std, transit_sigma, chan_mask=self.mask
            )
            dist = self.distributions[ii]
            significances.append(dist.polyfit(score))

        # return prob in units of nats = ln(P(D|H1)/P(D|H0)). ignore negative probs
        return -max(0, np.max(significances)) * np.log(2)

    def _get_sig_ts(self, sig_ts: float | list | None = None) -> np.ndarray:
        if sig_ts is None:
            sig_ts = np.sqrt(np.median(self.spec_std) ** 2 * np.array(self.q_default))
        elif isinstance(sig_ts, float):
            sig_ts = np.array([sig_ts])
        elif isinstance(sig_ts, list):
            sig_ts = np.array(sig_ts)
        else:
            raise ValueError("Not sure what to do with sig_ts {0}".format(sig_ts))
        if not np.all(np.nan_to_num(sig_ts)):
            raise ValueError("sig_ts are nans. Not estimating coeffs.")
        return sig_ts


class KalmanDistribution(object):
    """Generate a monte-carlo simulated Kalman score distribution for a given state transition std.

    Parameters
    ----------
    sigma_arr : numpy.ndarray
        per-channel standard deviation.
    t_sig : float
        transition std for the intrinsic markov process.
    ntrials : int, optional
        number of gaussian noise instances to be used, by default 10000
    mask_tol : float, optional
        The absolute tolerance parameter to flag standard deviation values, by default 1e-5

    Raises
    ------
    ValueError
        if all sigma_arr values are zero or negligible.
    """

    def __init__(
        self,
        sigma_arr: np.ndarray,
        t_sig: float,
        ntrials: int = 10000,
        mask_tol: float = 1e-5,
    ) -> None:
        self._sigma_arr = sigma_arr
        self._t_sig = t_sig
        self._ntrials = ntrials
        # Check the values of per-channel standard deviation and mitigate any zero.
        self._mask_tol = mask_tol
        self._mask = np.isclose(sigma_arr, 0, atol=mask_tol)
        if np.all(self._mask):
            raise ValueError(
                "sigma_arr are all zeros or negligible. Not preparing kalman significance distribution."
            )

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
    def t_sig(self) -> float:
        """Transition std for the intrinsic markov process."""
        return self._t_sig

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
            self.scores, 100 * (1 - 2.0 ** (-self.theoretical_quantiles))
        )

    @property
    def polyfit(self) -> np.poly1d:
        """Polynomial fit to the tail of the kalman scores distribution."""
        return self._polyfit

    def plot_diagnostic(
        self, bins=30, figsize=(13, 5.5), dpi=100, logy=False, outfile=None
    ):
        """
        Make a plot of the distribution.
        """
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
            tail_samples, bins=bins, density=True, histtype="step", ec="tab:blue", lw=2
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

    def _generate(self) -> None:
        scores = []
        for _itrial in range(self.ntrials):
            random_spec = np.random.normal(0, self.sigma_arr, size=self.nchans)
            norm_random_spec = normalize_spectrum(
                random_spec, self.sigma_arr, chan_mask=self.mask
            )
            score = kalman_filter(
                norm_random_spec, self.sigma_arr, self.t_sig, chan_mask=self.mask
            )
            scores.append(score)
        self._scores = np.array(scores)

    def _fit_distribution(self) -> None:
        """Approximating the tail of the distribution as an exponential tail (probably is justified)."""
        self._polyfit = Polynomial.fit(
            self.sample_quantiles, self.theoretical_quantiles, 1
        )

    def __str__(self) -> str:
        return f"KalmanDistribution(t_sig={self.t_sig:.3f}, ntrials={self.ntrials}, nchans={self.nchans})"

    def __repr__(self) -> str:
        return str(self)


def secondary_spectrum_cumulative_chi2_score(
    spec: np.ndarray, spec_std: np.ndarray, mask_tol: float = 1e-5
) -> float:
    """Compute the cumulative-chi2 test statistic on sig.

    Parameters
    ----------
    spec : numpy.ndarray
        1D array of the observed spectrum I(f) of the candidate FRB.
    spec_std : numpy.ndarray
        1D array of the standard deviation of the observed spectrum.
    mask_tol : float, optional
        The absolute tolerance parameter to flag standard deviation values, by default 1e-5.

    Returns
    -------
    float
        Maximum value of the significance. max_f0 of ln(P(sig|H1,ff0)/P(sig|H0))

    Notes
    -----
    Would compute the cumulative-chi2 test statistic on sig.
    Assumes the signal is composed of i.i.d N(E,1) variables ($E$ would be ignored).
    Would return the following statistical test between:
    H0: sig(f) ~ N(E,1)
    H1: sig(f) ~ N(E,1) + FRB with A(f) that have secondary spectrum (DFT(A(f)) with freq cutoff ff0
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
    return np.max(-significance_arr)
