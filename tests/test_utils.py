import numpy as np
import pytest
from scipy import stats

from kalman_detector import utils


class TestUtils:
    def test_add_noise_len(self) -> None:
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 64)
        spec_std = rng.normal(0, 1, 64)
        cur_snr = np.sum(spec) / (np.sum(spec_std**2)) ** 0.5
        target_snr = cur_snr * 0.8
        spec_noise, spec_std_noise = utils.add_noise(
            spec,
            spec_std,
            cur_snr,
            target_snr,
        )
        np.testing.assert_equal(len(spec_noise), len(spec))
        np.testing.assert_equal(len(spec_std_noise), len(spec_std))

    def test_add_noise(self) -> None:
        spec = np.ones(64) * 10
        spec_std = np.ones(64)
        cur_snr = np.sum(spec) / (np.sum(spec_std**2)) ** 0.5
        target_snr = cur_snr * 0.5
        ntrials = 10000
        snr_arr = []
        for _ in range(ntrials):
            spec_noise, spec_std_noise = utils.add_noise(
                spec,
                spec_std,
                cur_snr,
                target_snr,
            )
            new_snr = np.sum(spec_noise) / (np.sum(spec_std_noise**2)) ** 0.5
            snr_arr.append(new_snr)
        new_snr_mean = np.mean(snr_arr)
        np.testing.assert_almost_equal(new_snr_mean, target_snr, decimal=1)

    def test_normalize_spectrum(self) -> None:
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 1024) + 5
        spec_std = rng.normal(0, 1, 1024)
        spec_norm = utils.normalize_spectrum(spec, spec_std)
        normalized_mean = np.average(spec_norm, weights=spec_std**-2)
        np.testing.assert_equal(len(spec_norm), len(spec))
        np.testing.assert_almost_equal(normalized_mean, 0, decimal=3)

    def test_normalize(self) -> None:
        rng = np.random.default_rng()
        spec_mean = rng.normal(50, 5, 1024)
        spec_std = rng.normal(20, 3, 1024)
        spec = rng.normal(spec_mean, spec_std, 1024)
        spec_norm = utils.normalize(spec, spec_std, spec_mean=spec_mean)
        normalized_mean = np.mean(spec_norm)
        normalized_std = np.std(spec_norm)
        np.testing.assert_equal(len(spec_norm), len(spec))
        np.testing.assert_almost_equal(normalized_mean, 0, decimal=1)
        np.testing.assert_almost_equal(normalized_std, 1, decimal=1)

    def test_normalize_nomean(self) -> None:
        rng = np.random.default_rng()
        spec_mean = np.zeros(1024)
        spec_std = rng.normal(20, 3, 1024)
        spec = rng.normal(spec_mean, spec_std, 1024)
        spec_norm = utils.normalize(spec, spec_std)
        normalized_std = np.std(spec_norm)
        np.testing.assert_equal(len(spec_norm), len(spec))
        np.testing.assert_almost_equal(normalized_std, 1, decimal=1)

    @pytest.mark.parametrize("sigma", [1, 3, 5, 10, 20, 30])
    def test_get_snr_from_logsf(self, sigma: float) -> None:
        logsf = np.log(stats.norm.sf(sigma))
        snr = utils.get_snr_from_logsf(logsf)
        np.testing.assert_almost_equal(snr, sigma, decimal=5)

    @pytest.mark.parametrize("sigma", [35, 36, 37])
    def test_get_snr_from_logsf_extreme(self, sigma: float) -> None:
        logsf = np.log(stats.norm.sf(sigma))
        snr = utils.get_snr_from_logsf(logsf)
        np.testing.assert_almost_equal(snr, sigma, decimal=0)

    def test_simulate_gaussian_signal(self) -> None:
        nchans = 336
        corr_len = 100
        signal = utils.simulate_gaussian_signal(nchans, corr_len)
        signal_mean = np.mean(signal)
        np.testing.assert_equal(len(signal), nchans)
        np.testing.assert_almost_equal(signal_mean, 0, decimal=1)

    def test_simulate_gaussian_signal_complex(self) -> None:
        nchans = 336
        corr_len = 100
        signal = utils.simulate_gaussian_signal(nchans, corr_len, complex_process=True)
        np.testing.assert_equal(len(signal), nchans)
        np.testing.assert_almost_equal(np.mean(signal), 0, decimal=1)


class TestSnrResult:
    def test_init(self) -> None:
        name = "test"
        snr_box = 5
        sig_kalman = 0
        result = utils.SnrResult(name, snr_box, sig_kalman)
        np.testing.assert_almost_equal(result.sig_box, stats.norm.logsf(snr_box))
        np.testing.assert_almost_equal(result.snr_kalman, snr_box)

    def test_to_dict(self) -> None:
        name = "test"
        snr_box = 5
        sig_kalman = 10
        result = utils.SnrResult(name, snr_box, sig_kalman)
        result_dict = result.to_dict()
        np.testing.assert_equal(result_dict["name"], name)
        assert isinstance(result_dict, dict)

    def test_str(self) -> None:
        name = "test"
        snr_box = 5
        sig_kalman = 10
        result = utils.SnrResult(name, snr_box, sig_kalman)
        assert str(result).startswith("test")
