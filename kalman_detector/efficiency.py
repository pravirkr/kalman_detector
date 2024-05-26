from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track
from scipy import stats
from uncertainties import unumpy

from kalman_detector import utils
from kalman_detector.main import KalmanDetector

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Result:
    snr_emp: np.ndarray
    kal_score: np.ndarray
    kal_sig: np.ndarray

    @property
    def snr_sig(self) -> np.ndarray:
        return stats.norm.logsf(self.snr_emp)


class Results:
    def __init__(self, results: list[Result]) -> None:
        self.results = results

    @property
    def kal_score(self) -> np.ndarray:
        return self.get_unumpy("kal_score")

    @property
    def kal_sig(self) -> np.ndarray:
        return self.get_unumpy("kal_sig")

    @property
    def snr_emp(self) -> np.ndarray:
        return self.get_unumpy("snr_emp")

    @property
    def snr_sig(self) -> np.ndarray:
        return self.get_unumpy("snr_sig")

    @property
    def efficiency(self) -> np.ndarray:
        return 2 * self.kal_score / self.snr_emp**2

    def get_unumpy(self, key: str) -> unumpy.uarray:
        return unumpy.uarray(self.get_item(key, np.mean), self.get_item(key, np.std))

    def get_item(self, key: str, func: Callable = np.mean) -> np.ndarray:
        return np.array([func(getattr(result, key)) for result in self.results])


def monte_carlo(
    template: np.ndarray,
    target_snr: float,
    q_arr: np.ndarray,
    niters: int = 10000,
) -> list[Result]:
    snr_emp_arr = np.empty(niters)
    kal_sig_arr = np.empty(shape=(niters, len(q_arr)))
    kal_score_arr = np.empty(shape=(niters, len(q_arr)))

    spec_mean = np.zeros_like(template)
    spec_std = np.ones_like(template)
    kalman = KalmanDetector(spec_std, q_par=q_arr)
    kalman.prepare_fits(ntrials=10000)
    rng = np.random.default_rng()
    for ii in range(niters):
        spec = target_snr * template + rng.normal(
            spec_mean,
            spec_std,
            len(template),
        )
        snr_emp_arr[ii] = np.dot(template, utils.normalize(spec, spec_std)) / np.sqrt(
            np.dot(template, template),
        )
        sigs, scores = kalman.get_significance(spec)
        kal_sig_arr[ii] = sigs
        kal_score_arr[ii] = scores
    results = []
    for iq, _ in enumerate(q_arr):
        results.append(Result(snr_emp_arr, kal_score_arr[:, iq], kal_sig_arr[:, iq]))
    return results


def sim_efficiency(
    template: np.ndarray,
    snr_arr: np.ndarray,
    q_arr: np.ndarray,
    niters: int = 10000,
) -> list[Results]:
    results_arr = np.empty((len(q_arr), len(snr_arr)), dtype=object)
    for ii in track(range(len(snr_arr)), description="Simulating efficiency"):
        target_snr = snr_arr[ii]
        results = monte_carlo(template, target_snr, q_arr, niters=niters)
        results_arr[:, ii] = results
    results_ts = []
    for iresult, _ in enumerate(q_arr):
        results_ts.append(Results(results_arr[iresult]))
    return results_ts


def eff_plot(
    template: np.ndarray,
    freqs: np.ndarray,
    ax_eff: plt.Axes,
    ax_prof: plt.Axes,
    niters: int = 10000,
    snr_arr: np.ndarray | None = None,
    q_arr: np.ndarray | None = None,
) -> list[Results]:
    if snr_arr is None:
        snr_arr = np.arange(6, 40, 2)
    if q_arr is None:
        q_arr = np.array([0.5, 0.1, 0.05])
    results_ts = sim_efficiency(template, snr_arr, q_arr**2, niters=niters)

    colors = ["#8da0cb", "#fc8d62", "#66c2a5"]
    ax_prof.plot(freqs, template, label="filter")
    ax_prof.set_xlabel("Frequency (MHz)")
    ax_prof.set_ylabel("Amplitude")
    ax_prof.set_ylim(-0.15, 0.15)
    for iresult, results in enumerate(results_ts):
        ax_eff.errorbar(
            unumpy.nominal_values(results.snr_emp),
            unumpy.nominal_values(results.efficiency),
            xerr=unumpy.std_devs(results.snr_emp),
            yerr=unumpy.std_devs(results.efficiency),
            fmt="o",
            mec="k",
            mfc=colors[iresult],
            ms=7,
            capsize=1.7,
            ecolor="darkgrey",
            elinewidth=1.2,
            label=f"$q^{2}$ = {q_arr[iresult]:.2f}",
        )
    ax_eff.axhline(1, ls="--", color="k", lw=1)
    ax_eff.set_xlabel("S/N")
    ax_eff.set_ylabel("Efficiency")
    ax_eff.set_ylim(0.4, 1.1)
    ax_eff.set_xlim(7, 40)
    ax_eff.legend(loc="lower right", frameon=True)
    return results_ts


if __name__ == "__main__":
    niters = 10000
    nchans = 336
    corr_len = 300
    freqs = np.arange(nchans) + 1104
    template = utils.simulate_gaussian_signal(nchans, corr_len, complex_process=True)

    fig, (ax_eff, ax_prof) = plt.subplots(2, 1, height_ratios=(2, 1), figsize=(6, 6.5))
    results = eff_plot(template, freqs, ax_eff, ax_prof, niters=niters)
    fig.tight_layout()
    plt.show()
