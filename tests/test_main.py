from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from kalman_detector import utils
from kalman_detector.main import (
    KalmanDetector,
    KalmanDistribution,
    secondary_spectrum_cumulative_chi2_score,
)


class TestKalmanDetector:
    def test_q_par_float(self) -> None:
        q_par = 0.1
        std_vec = np.ones(336)
        kalman = KalmanDetector(std_vec, q_par**2)
        np.testing.assert_array_equal(kalman.transit_sigmas, np.array([q_par]))

    def test_q_par_list(self) -> None:
        q_par = [0.1, 0.2]
        q_par_sq = [qq**2 for qq in q_par]
        std_vec = np.ones(336)
        kalman = KalmanDetector(std_vec, q_par_sq)
        np.testing.assert_array_equal(kalman.transit_sigmas, np.array(q_par))

    def test_initialization_fail(self) -> None:
        std_vec = np.zeros(1024)
        with pytest.raises(ValueError):
            KalmanDetector(std_vec)

    def test_prepare_fits(self) -> None:
        q_par = [0.1, 0.2]
        std_vec = np.arange(0.1, 1, 0.01)
        kalman = KalmanDetector(std_vec, q_par)
        kalman.prepare_fits(ntrials=1000)
        assert isinstance(kalman.distributions[0], KalmanDistribution)
        np.testing.assert_equal(len(kalman.distributions), len(q_par))

    def test_get_significance(self) -> None:
        nchans = 128
        target = 5
        rng = np.random.default_rng()
        std_vec = rng.normal(1, 0.1, size=nchans)
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits(ntrials=1000)
        spectrum = rng.normal(target, std_vec)
        sigs, scores = kalman.get_significance(spectrum)
        np.testing.assert_equal(len(sigs), len(scores))
        np.testing.assert_array_less(sigs, 0.1)

    def test_get_significance_fail(self) -> None:
        nchans = 128
        rng = np.random.default_rng()
        std_vec = rng.normal(1, 0.1, size=nchans)
        spectrum = np.ones(nchans + 1)
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits(ntrials=1000)
        with pytest.raises(ValueError):
            kalman.get_significance(spectrum)

    def test_get_best_significance(self) -> None:
        nchans = 336
        corr_len = 300
        ntrials = 1000
        target_snr = 20

        scores_arr = []
        template = utils.simulate_gaussian_signal(
            nchans,
            corr_len,
            complex_process=True,
        )
        spec_mean = np.zeros_like(template)
        spec_std = np.ones_like(template)
        kalman = KalmanDetector(spec_std)
        kalman.prepare_fits(ntrials=1000)
        rng = np.random.default_rng()
        for _ in range(ntrials):
            spec = target_snr * template + rng.normal(
                spec_mean,
                spec_std,
                len(template),
            )
            best_sig = kalman.get_best_significance(spec)
            scores_arr.append(best_sig)
        score_mean = utils.get_snr_from_logsf(np.array(scores_arr).mean())
        np.testing.assert_allclose(score_mean, target_snr, rtol=0.2)


class TestKalmanDistribution:
    def test_initialization(self) -> None:
        sigma_arr = np.arange(0.1, 1, 0.01)
        sig_eta = 0.1
        mask_tol = 0.1
        ntrials = 1000
        kdist = KalmanDistribution(
            sigma_arr,
            sig_eta,
            ntrials=ntrials,
            mask_tol=mask_tol,
        )
        np.testing.assert_equal(kdist.mask_tol, mask_tol)
        np.testing.assert_equal(len(kdist.mask), len(sigma_arr))
        np.testing.assert_equal(kdist.sig_eta, sig_eta)
        np.testing.assert_equal(kdist.ntrials, ntrials)

    def test_initialization_fail(self) -> None:
        sigma_arr = np.zeros(1024)
        with pytest.raises(ValueError):
            KalmanDistribution(sigma_arr, 0.1)

    def test_plot_diagnostic(self, tmpfile: str) -> None:
        outfile_path = Path(f"{tmpfile}.pdf")
        sigma_arr = np.arange(0.1, 1, 0.01)
        kdist = KalmanDistribution(sigma_arr, 0.01, ntrials=1000)
        kdist.plot_diagnostic(logy=True, outfile=outfile_path.as_posix())
        assert outfile_path.is_file()
        outfile_path.unlink()

    def test_polyfit(self) -> None:
        sigma_arr = np.arange(0.1, 1, 0.01)
        kdist = KalmanDistribution(sigma_arr, 0.01, ntrials=1000)
        assert isinstance(kdist.polyfit, Polynomial)

    def test_str(self) -> None:
        sigma_arr = np.arange(0.1, 1, 0.01)
        kdist = KalmanDistribution(sigma_arr, 0.01, ntrials=1000)
        assert str(kdist).startswith("KalmanDistribution")
        assert repr(kdist) == str(kdist)


class TestSecondarySpectrym:
    def test_secondary_spectrum_cumulative_chi2_score(self) -> None:
        nchans = 336
        corr_len = 300
        ntrials = 1000
        target_snr = 20

        scores_arr = []
        template = utils.simulate_gaussian_signal(
            nchans,
            corr_len,
            complex_process=True,
        )
        spec_mean = np.zeros_like(template)
        spec_std = np.ones_like(template)
        rng = np.random.default_rng()
        for _ in range(ntrials):
            spec = target_snr * template + rng.normal(
                spec_mean,
                spec_std,
                len(template),
            )
            score = secondary_spectrum_cumulative_chi2_score(spec, spec_std)
            scores_arr.append(score)
        score_mean = utils.get_snr_from_logsf(np.array(scores_arr).mean())
        np.testing.assert_allclose(score_mean, target_snr, rtol=0.2)
