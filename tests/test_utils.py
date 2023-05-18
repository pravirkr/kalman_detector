import numpy as np

from kalman_detector import utils


class TestUtils(object):
    def test_add_noise_len(self):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 64)
        spec_std = rng.normal(0, 1, 64)
        current_snr = np.sum(spec) / (np.sum(spec_std**2)) ** 0.5
        spec_noise, spec_std_noise = utils.add_noise(
            spec, spec_std, current_snr, current_snr * 0.8
        )
        assert len(spec_noise) == len(spec)
        assert len(spec_std_noise) == len(spec_std)

    def test_add_noise(self):
        spec = np.ones(64)
        spec_std = np.ones(64)
        current_snr = np.sum(spec) / (np.sum(spec_std**2)) ** 0.5
        spec_noise, spec_std_noise = utils.add_noise(
            spec, spec_std, current_snr, current_snr * 0.5
        )
        new_snr = np.sum(spec_noise) / (np.sum(spec_std_noise**2)) ** 0.5
        np.testing.assert_allclose(new_snr, current_snr * 0.5, rtol=0.8)

    def test_normalize_spectrum(self):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 1024) + 5
        spec_std = rng.normal(0, 1, 1024)
        spec_norm = utils.normalize_spectrum(spec, spec_std)
        normalized_mean = np.average(spec_norm, weights=spec_std ** -2)
        np.testing.assert_equal(len(spec_norm), len(spec))
        np.testing.assert_almost_equal(normalized_mean, 0, decimal=3)

    def test_get_snr_from_logsf(self):
        logsf = -10
        snr = utils.get_snr_from_logsf(logsf)
        np.testing.assert_allclose(snr, 3.9, atol=0.1)

        logsf = -800
        snr = utils.get_snr_from_logsf(logsf)
        np.testing.assert_allclose(snr, 40, atol=0.1)

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
