import pytest
import numpy as np
from scipy import stats
from pathlib import Path
from numpy.polynomial import Polynomial

from kalman_detector.main import (
    KalmanDetector,
    KalmanDistribution,
    secondary_spectrum_cumulative_chi2_score,
)


class TestKalmanDetector(object):
    def test_sig_ts_float(self):
        sig_ts = 0.1
        std_vec = np.arange(0.1, 1, 0.01)
        kalman = KalmanDetector(std_vec, sig_ts)
        np.testing.assert_array_equal(kalman.transit_sigmas, np.array([sig_ts]))

    def test_sig_ts_list(self):
        sig_ts = [0.1, 0.2]
        std_vec = np.arange(0.1, 1, 0.01)
        kalman = KalmanDetector(std_vec, sig_ts)
        np.testing.assert_array_equal(kalman.transit_sigmas, np.array(sig_ts))

    def test_sig_ts_fail(self):
        sig_ts = {"a": 0.1, "b": 0.2}
        std_vec = np.arange(0.1, 1, 0.01)
        with pytest.raises(ValueError):
            kalman = KalmanDetector(std_vec, sig_ts)

    def test_sig_ts_nan_fail(self):
        sig_ts = [0.1, np.nan]
        std_vec = np.arange(0.1, 1, 0.01)
        with pytest.raises(ValueError):
            kalman = KalmanDetector(std_vec, sig_ts)

    def test_initialization_fail(self):
        std_vec = np.zeros(1024)
        with pytest.raises(ValueError):
            kalman = KalmanDetector(std_vec)

    def test_prepare_fits(self):
        sig_ts = [0.1, 0.2]
        std_vec = np.arange(0.1, 1, 0.01)
        kalman = KalmanDetector(std_vec, sig_ts)
        kalman.prepare_fits(ntrials=1000)
        assert isinstance(kalman.distributions[0], KalmanDistribution)
        np.testing.assert_equal(len(kalman.distributions), len(sig_ts))

    def test_get_significance(self):
        nchans = 128
        target = 5
        rng = np.random.default_rng()
        std_vec = rng.normal(1, 0.1, size=nchans)
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits(ntrials=1000)
        spectrum = rng.normal(target, std_vec)
        sig_kalman = kalman.get_significance(spectrum)
        assert sig_kalman <= 0

    def test_get_significance_fail(self):
        nchans = 128
        rng = np.random.default_rng()
        std_vec = rng.normal(1, 0.1, size=nchans)
        spectrum = np.ones(nchans + 1)
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits(ntrials=1000)
        with pytest.raises(ValueError):
            sig_kalman = kalman.get_significance(spectrum)

    def test_sensitivity_up_down(self):
        nchans = 800
        signal = np.concatenate([np.ones(10), np.zeros(10)] * 40)
        std_noise = np.sin(0.1 * np.arange(nchans)) + 2
        optimal_snr = np.sum(signal**2 / std_noise**2) ** 0.5
        snr_sum = np.sum(signal / std_noise**2) / np.sum(1 / std_noise**2) ** 0.5

        rng = np.random.default_rng()
        kalman = KalmanDetector(std_noise)
        kalman.prepare_fits(ntrials=10000)
        sig_kalman = []
        sig_naive = []
        for i in range(1000):
            spectrum = signal + rng.normal(0, std_noise)
            snr_naive = (
                np.sum(spectrum / std_noise**2) / np.sum(1 / std_noise**2) ** 0.5
            )
            sig_naive.append(stats.norm.logsf(snr_naive))
            sig_kalman.append(kalman.get_significance(spectrum))

        sig_naive_mean = np.mean(sig_naive)
        improvement = (np.array(sig_naive) + np.array(sig_kalman)) / stats.norm.logsf(
            optimal_snr
        )
        np.testing.assert_almost_equal(
            sig_naive_mean, stats.norm.logsf(snr_sum), decimal=0
        )
        np.testing.assert_almost_equal(np.mean(improvement), 0.6, decimal=1)

    def test_sin(self):
        nchans = 400
        rng = np.random.default_rng()
        sig_list = [
            rng.random()
            * np.sin(
                rng.random() * 2 * np.pi + rng.random(nchans) * np.arange(nchans) / 4
            )
            for i in range(4)
        ]
        signal = np.abs((1.5 + np.sum(sig_list, axis=0)) * 0.6)
        std_noise = np.sin(0.1 * np.arange(nchans)) + 2
        optimal_snr = np.sum(signal**2 / std_noise**2) ** 0.5
        snr_sum = np.sum(signal / std_noise**2) / np.sum(1 / std_noise**2) ** 0.5

        kalman = KalmanDetector(std_noise)
        kalman.prepare_fits(ntrials=10000)
        sig_kalman = []
        sig_naive = []
        optimal_sig_list = []

        for i in range(1000):
            spectrum = signal + rng.normal(0, std_noise)
            snr_naive = (
                np.sum(spectrum / std_noise**2) / np.sum(1 / std_noise**2) ** 0.5
            )
            sig_naive.append(stats.norm.logsf(snr_naive))
            optimal_sig_list.append(
                stats.norm.logsf(
                    np.sum(signal * spectrum / std_noise**2)
                    / np.sum(signal**2 / std_noise**2) ** 0.5
                )
            )
            sig_kalman.append(kalman.get_significance(spectrum))

        sig_naive_mean = np.mean(sig_naive)
        improvement = (np.array(sig_naive) + np.array(sig_kalman)) / stats.norm.logsf(
            optimal_snr
        )
        np.testing.assert_almost_equal(
            sig_naive_mean, stats.norm.logsf(snr_sum), decimal=0
        )
        np.testing.assert_almost_equal(np.mean(improvement), 0.8, decimal=1)


class TestKalmanDistribution(object):
    def test_initialization(self):
        std_vec = np.arange(0.1, 1, 0.01)
        t_sig = 0.1
        mask_tol = 0.1
        ntrials = 1000
        kdist = KalmanDistribution(std_vec, t_sig, ntrials=ntrials, mask_tol=mask_tol)
        np.testing.assert_equal(kdist.mask_tol, mask_tol)
        np.testing.assert_equal(len(kdist.mask), len(std_vec))
        np.testing.assert_equal(kdist.t_sig, t_sig)
        np.testing.assert_equal(kdist.ntrials, ntrials)

    def test_initialization_fail(self):
        std_vec = np.zeros(1024)
        with pytest.raises(ValueError):
            kdist = KalmanDistribution(std_vec, 0.1)

    def test_plot_diagnostic(self, tmpfile):
        outfile_path = Path(f"{tmpfile}.pdf")
        std_vec = np.arange(0.1, 1, 0.01)
        kdist = KalmanDistribution(std_vec, 0.01, ntrials=1000)
        fig = kdist.plot_diagnostic(logy=True, outfile=outfile_path)
        assert outfile_path.is_file()
        outfile_path.unlink()

    def test_polyfit(self):
        std_vec = np.arange(0.1, 1, 0.01)
        kdist = KalmanDistribution(std_vec, 0.01, ntrials=1000)
        assert isinstance(kdist.polyfit, Polynomial)

    def test_str(self):
        std_vec = np.arange(0.1, 1, 0.01)
        kdist = KalmanDistribution(std_vec, 0.01, ntrials=1000)
        assert str(kdist).startswith("KalmanDistribution")

    def test_repr(self):
        std_vec = np.arange(0.1, 1, 0.01)
        kdist = KalmanDistribution(std_vec, 0.01, ntrials=1000)
        assert repr(kdist).startswith("KalmanDistribution")


class TestSecondarySpectrym(object):
    def test_secondary_spectrum_cumulative_chi2_score(self):
        nchans = 128
        ntrials = 1000
        scores_arr = []
        target = 5
        rng = np.random.default_rng()
        for _ in range(ntrials):
            std_vec = rng.normal(1, 0.1, size=nchans)
            spectrum = rng.normal(target, std_vec)
            score = secondary_spectrum_cumulative_chi2_score(spectrum, std_vec)
            scores_arr.append(score)
        new_score_mean = np.mean(scores_arr)
        np.testing.assert_almost_equal(new_score_mean, target, decimal=0)
