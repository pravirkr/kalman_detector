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

    def test_simulate_ar1_signal(self) -> None:
        rng = np.random.default_rng(42)
        sig = utils.simulate_ar1_signal(256, alpha=0.95, sig_eta=0.1, rng=rng)
        assert sig.shape == (256,)
        np.testing.assert_allclose(np.mean(sig), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(sig), 1.0, atol=1e-12)
        # Random-walk branch (alpha = 1) without normalization.
        rw = utils.simulate_ar1_signal(
            128, alpha=1.0, sig_eta=0.2, beta=1.0, normalize_norm=False, rng=rng
        )
        assert rw.shape == (128,)
        np.testing.assert_allclose(rw[0], 1.0)

    def test_simulate_scintillation_2d(self) -> None:
        rng = np.random.default_rng(7)
        phase_map, spectra = utils.simulate_scintillation_2d(
            nchans=64, n_pix=32, rng=rng
        )
        assert phase_map.shape == (32, 32)
        assert spectra.shape == (64, 32)
        assert np.all(spectra >= 0)

    def test_extract_waterfall_spectrum(self) -> None:
        rng = np.random.default_rng(11)
        wf = rng.normal(0.0, 1.0, size=(64, 200))
        wf[:16, 95:105] += 3.0
        spec, spec_std, snr_box, off_specs = utils.extract_waterfall_spectrum(
            wf, (95, 105), guard_bins=20
        )
        assert spec.shape == (64,)
        assert spec_std.shape == (64,)
        assert snr_box > 3.0
        assert off_specs.ndim == 2
        assert off_specs.shape[1] == 64

        spec_pw, std_pw, snr_pw, off_pw = utils.extract_waterfall_spectrum(
            wf, (95, 105), profile_weighted=True, guard_bins=20
        )
        assert spec_pw.shape == (64,)
        assert std_pw.shape == (64,)
        assert snr_pw > 3.0
        assert off_pw.shape == off_specs.shape

    def test_fit_spectral_running(self) -> None:
        freqs = np.linspace(400.0, 800.0, 128)
        f0 = 600.0
        x = freqs / f0
        true_spec = 10.0 * (x ** (1.5 - 8.0 * np.log(x)))
        fit = utils.fit_spectral_running(freqs, true_spec, f0=f0)
        np.testing.assert_allclose(fit["amp"], 10.0, rtol=1e-2)
        np.testing.assert_allclose(fit["gamma"], 1.5, atol=5e-2)
        np.testing.assert_allclose(fit["beta"], -8.0, rtol=2e-2)
        expected_peak = f0 * np.exp(-1.5 / (2.0 * -8.0))
        np.testing.assert_allclose(fit["peak_freq"], expected_peak, rtol=1e-2)


class TestSnrResult:
    def test_init(self) -> None:
        name = "test"
        snr_box = 5
        sig_kalman = 0
        result = utils.SnrResult(name, snr_box, sig_kalman)
        np.testing.assert_almost_equal(result.sig_box, stats.norm.logsf(snr_box))
        np.testing.assert_almost_equal(result.snr_kalman_legacy, snr_box)
        # Fisher's combination penalises an uninformative second p-value.
        log_px = result.sig_box
        np.testing.assert_almost_equal(
            result.sig_combined,
            log_px + np.log(1 - log_px),
        )
        assert result.snr_kalman < snr_box

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

    def test_n_eff(self) -> None:
        result = utils.SnrResult("test", 6, -20, n_eff=4)
        np.testing.assert_allclose(
            result.sig_kalman_corrected,
            np.log(1 - (1 - np.exp(-20)) ** 4),
            rtol=1e-6,
        )
        assert result.sig_combined > result.sig_box + result.sig_kalman
        assert result.snr_kalman < result.snr_kalman_legacy
        result_dict = result.to_dict()
        for key in ("n_eff", "sig_combined", "snr_kalman_legacy"):
            assert key in result_dict


class TestCalibration:
    def test_fisher_logsf_uniform(self) -> None:
        rng = np.random.default_rng(0)
        ntrials = 200000
        log_pa = np.log(rng.uniform(size=ntrials))
        log_pb = np.log(rng.uniform(size=ntrials))
        combined = np.array(
            [utils.fisher_logsf(a, b) for a, b in zip(log_pa, log_pb, strict=True)],
        )
        for alpha in (0.1, 0.01, 1e-3):
            rate = np.mean(combined < np.log(alpha))
            np.testing.assert_allclose(rate, alpha, rtol=4 / np.sqrt(alpha * ntrials))
        # The uncalibrated product is anti-conservative (REPORT 4.1).
        assert np.mean(log_pa + log_pb < np.log(1e-3)) > 5e-3

    def test_fisher_logsf_values(self) -> None:
        np.testing.assert_allclose(utils.fisher_logsf(0, 0), 0)
        log_px = -50.0
        np.testing.assert_allclose(
            utils.fisher_logsf(-20.0, -30.0),
            log_px + np.log(1 - log_px),
        )

    @pytest.mark.parametrize("n_trials", [1, 2.5, 8])
    def test_sidak_logsf(self, n_trials: float) -> None:
        for logp in (-1e-3, -0.5, -3.0, -50.0, -800.0):
            if logp < -30:
                expected = np.log(n_trials) + logp
            else:
                expected = np.log1p(-((1 - np.exp(logp)) ** n_trials))
            np.testing.assert_allclose(
                utils.sidak_logsf(logp, n_trials),
                expected,
                rtol=1e-6,
            )
        np.testing.assert_equal(utils.sidak_logsf(0.0, n_trials), 0.0)

    def test_sidak_logsf_uniform(self) -> None:
        rng = np.random.default_rng(1)
        n_trials = 5
        min_p = rng.uniform(size=(200000, n_trials)).min(axis=1)
        corrected = np.array([utils.sidak_logsf(np.log(p), n_trials) for p in min_p])
        np.testing.assert_allclose(np.mean(corrected < np.log(0.01)), 0.01, rtol=0.1)

    def test_effective_number_of_tests(self) -> None:
        np.testing.assert_allclose(utils.effective_number_of_tests(np.eye(5)), 5)
        np.testing.assert_allclose(utils.effective_number_of_tests(np.ones((5, 5))), 1)
        corr = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
        n_eff = utils.effective_number_of_tests(corr)
        assert 1 <= n_eff <= 4
