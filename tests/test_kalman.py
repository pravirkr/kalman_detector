import pytest
import numpy as np

from pathlib import Path

from kalman_detector.main import (
    KalmanDetector,
    KalmanDistribution,
    secondary_spectrum_cumulative_chi2_score,
)


class TestKalmanDetector(object):
    def test_sig_ts_float(self):
        sig_ts = 0.1
        std_vec = np.arange(0, 1, 0.01)
        kalman = KalmanDetector(std_vec, sig_ts)
        assert kalman.transit_sigmas == np.array([sig_ts])

    def test_sig_ts_list(self):
        sig_ts = [0.1, 0.2]
        std_vec = np.arange(0, 1, 0.01)
        kalman = KalmanDetector(std_vec, sig_ts)
        np.testing.assert_array_equal(kalman.transit_sigmas, np.array(sig_ts))

    def test_sig_ts_fail(self):
        sig_ts = {"a": 0.1, "b": 0.2}
        std_vec = np.arange(0, 1, 0.01)
        with pytest.raises(ValueError):
            kalman = KalmanDetector(std_vec, sig_ts)

    def test_sig_ts_nan_fail(self):
        sig_ts = [0.1, np.nan]
        std_vec = np.arange(0, 1, 0.01)
        with pytest.raises(ValueError):
            kalman = KalmanDetector(std_vec, sig_ts)

    def test_initialization_fail(self):
        std_vec = np.zeros(1024)
        with pytest.raises(ValueError):
            kalman = KalmanDetector(std_vec)

    def test_get_significance_fail(self):
        std_vec = np.arange(0, 1, 0.01)
        spectrum = np.ones(1024)
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits(ntrials=1000)
        with pytest.raises(ValueError):
            sig_kalman = kalman.get_significance(spectrum)



class TestKalmanDistribution(object):
    def test_initialization(self):
        std_vec = np.arange(0, 1, 0.01)
        t_sig = 0.1
        mask_tol = 0.1
        ntrials = 1000
        kdist = KalmanDistribution(std_vec, t_sig, ntrials=ntrials, mask_tol=mask_tol)
        assert kdist.t_sig == t_sig
        assert kdist.mask_tol == mask_tol
        assert kdist.ntrials == ntrials

    def test_initialization_fail(self):
        std_vec = np.zeros(1024)
        with pytest.raises(ValueError):
            kdist = KalmanDistribution(std_vec, 0.1)

    def test_plot_diagnostic(self, tmpfile):
        outfile_path = Path(f"{tmpfile}.pdf")
        std_vec = np.arange(0, 1, 0.01)
        kdist = KalmanDistribution(std_vec, 0.01, ntrials=1000)
        fig = kdist.plot_diagnostic(logy=True, outfile=outfile_path)
        assert outfile_path.is_file()
        outfile_path.unlink()

    def test_str(self):
        std_vec = np.arange(0, 1, 0.01)
        kdist = KalmanDistribution(std_vec, 0.01, ntrials=1000)
        assert str(kdist).startswith("KalmanDistribution")

    def test_repr(self):
        std_vec = np.arange(0, 1, 0.01)
        kdist = KalmanDistribution(std_vec, 0.01, ntrials=1000)
        assert repr(kdist).startswith("KalmanDistribution")


class TestSecondarySpectrym(object):
    def test_secondary_spectrum_cumulative_chi2_score(self):
        rng = np.random.default_rng()
        std_vec = np.arange(0, 1, 0.01)
        spectrum = rng.normal(0, std_vec)
        score = secondary_spectrum_cumulative_chi2_score(spectrum, std_vec)
        assert score > 0
