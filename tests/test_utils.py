import pytest
import numpy as np
from scipy import stats

from kalman_detector import utils


class TestUtils(object):
    def test_add_noise_len(self):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 64)
        spec_std = rng.normal(0, 1, 64)
        cur_snr = np.sum(spec) / (np.sum(spec_std**2)) ** 0.5
        target_snr = cur_snr * 0.8
        spec_noise, spec_std_noise = utils.add_noise(spec, spec_std, cur_snr, target_snr)
        np.testing.assert_equal(len(spec_noise), len(spec))
        np.testing.assert_equal(len(spec_std_noise), len(spec_std))

    def test_add_noise(self):
        spec = np.ones(64) * 10
        spec_std = np.ones(64)
        cur_snr = np.sum(spec) / (np.sum(spec_std**2)) ** 0.5
        target_snr = cur_snr * 0.5
        ntrials = 10000
        snr_arr = []
        for _ in range(ntrials):
            spec_noise, spec_std_noise = utils.add_noise(
                spec, spec_std, cur_snr, target_snr
            )
            new_snr = np.sum(spec_noise) / (np.sum(spec_std_noise**2)) ** 0.5
            snr_arr.append(new_snr)
        new_snr_mean = np.mean(snr_arr)
        np.testing.assert_almost_equal(new_snr_mean, target_snr, decimal=1)

    def test_normalize_spectrum(self):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 1024) + 5
        spec_std = rng.normal(0, 1, 1024)
        spec_norm = utils.normalize_spectrum(spec, spec_std)
        normalized_mean = np.average(spec_norm, weights=spec_std**-2)
        np.testing.assert_equal(len(spec_norm), len(spec))
        np.testing.assert_almost_equal(normalized_mean, 0, decimal=3)

    @pytest.mark.parametrize("sigma", [1, 3, 5, 10, 20, 30])
    def test_get_snr_from_logsf(self, sigma):
        logsf = np.log(stats.norm.sf(sigma))
        snr = utils.get_snr_from_logsf(logsf)
        np.testing.assert_almost_equal(snr, sigma, decimal=5)

    @pytest.mark.parametrize("sigma", [35, 36, 37])
    def test_get_snr_from_logsf_extreme(self, sigma):
        logsf = np.log(stats.norm.sf(sigma))
        snr = utils.get_snr_from_logsf(logsf)
        np.testing.assert_almost_equal(snr, sigma, decimal=0)

    def test_simulate_1d_gaussian_process(self):
        mean = 0.0
        noise_std = 0.1
        nchan = 1024
        corr_len = 100
        amp = 1.0
        process = utils.simulate_1d_gaussian_process(
            mean, noise_std, nchan, corr_len, amp
        )
        np.testing.assert_equal(len(process), nchan)

    def test_simulate_1d_abs_complex_process(self):
        noise_std = 0.1
        nchan = 1024
        corr_len = 100
        snr_int = 10
        process = utils.simulate_1d_abs_complex_process(
            noise_std, nchan, corr_len, snr_int
        )
        np.testing.assert_equal(len(process), nchan)
