from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy import stats

from kalman_detector import utils
from kalman_detector.core import kalman_filter_cached, prepare_gain_cache
from kalman_detector.main import (
    KalmanDetector,
    KalmanDistribution,
    QuadraticFormTail,
    TridiagonalSaddlepointDistribution,
    contrast_eigenvalues,
    secondary_spectrum_cumulative_chi2_score,
)


def _masked_profile(nchans: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec_std = rng.lognormal(0, 0.3, nchans)
    spec_std[rng.choice(nchans, nchans // 16, replace=False)] = 0
    spec_std[nchans // 3 : nchans // 3 + nchans // 8] = 0
    return spec_std


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
        kalman.prepare_fits(ntrials=1000, method="mc")
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

    def test_prepare_fits_saddlepoint(self) -> None:
        std_vec = _masked_profile(256)
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits()
        assert kalman.method == "saddlepoint"
        assert all(
            isinstance(dist, TridiagonalSaddlepointDistribution)
            for dist in kalman.distributions
        )
        assert 1 <= kalman.n_eff <= len(kalman.transit_sigmas)
        kalman_mc = KalmanDetector(std_vec)
        kalman_mc.prepare_fits(ntrials=1000, method="mc")
        np.testing.assert_equal(kalman_mc.n_eff, len(kalman_mc.transit_sigmas))

    def test_prepare_fits_fail(self) -> None:
        std_vec = np.ones(64)
        with pytest.raises(ValueError):
            KalmanDetector(std_vec).prepare_fits(method="imhof")
        with pytest.raises(ValueError):
            KalmanDetector(std_vec, diffuse=False).prepare_fits(method="saddlepoint")
        with pytest.raises(RuntimeError):
            KalmanDetector(std_vec).get_significance(np.ones(64))
        with pytest.raises(RuntimeError):
            _ = KalmanDetector(std_vec).n_eff

    def test_get_significance_offset_invariant(self) -> None:
        std_vec = _masked_profile(256)
        rng = np.random.default_rng(1)
        spectrum = rng.normal(0, np.where(std_vec > 0, std_vec, 1))
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits()
        sigs, scores = kalman.get_significance(spectrum)
        sigs_off, scores_off = kalman.get_significance(spectrum + 10)
        np.testing.assert_allclose(scores_off, scores, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(sigs_off, sigs, rtol=1e-9, atol=1e-9)
        assert np.all(sigs <= 0)

    def test_get_significance_legacy_mc(self) -> None:
        std_vec = _masked_profile(128)
        rng = np.random.default_rng(2)
        spectrum = rng.normal(5, np.where(std_vec > 0, std_vec, 1))
        kalman = KalmanDetector(std_vec, diffuse=False)
        kalman.prepare_fits(ntrials=1000, method="mc")
        sigs, scores = kalman.get_significance(spectrum)
        np.testing.assert_equal(len(sigs), len(scores))
        assert np.all(sigs <= 0)

    def test_mc_preserves_mask_gaps(self) -> None:
        """MC calibration must keep gap lengths (REPORT 4.4, main.py:105 bug)."""
        std_vec = np.ones(64)
        std_vec[20:37] = 0
        kalman = KalmanDetector(std_vec, q_par=0.1)
        kalman.prepare_fits(ntrials=40000, method="mc")
        mc_scores = kalman.distributions[0].scores
        sig_eta = kalman.transit_sigmas[0]
        exact = TridiagonalSaddlepointDistribution(
            contrast_eigenvalues(std_vec, kalman.mask),
            sig_eta,
        )
        compressed = TridiagonalSaddlepointDistribution(
            contrast_eigenvalues(std_vec[~kalman.mask]),
            sig_eta,
        )
        se_mean = np.sqrt(exact.var / len(mc_scores))
        np.testing.assert_allclose(mc_scores.mean(), exact.mean, atol=5 * se_mean)
        np.testing.assert_allclose(mc_scores.var(), exact.var, rtol=0.05)
        assert abs(compressed.mean - exact.mean) > 20 * se_mean


class TestSaddlepointDistribution:
    def test_contrast_eigenvalues(self) -> None:
        spec_std = _masked_profile(64)
        mask = spec_std == 0
        idx = np.flatnonzero(~mask)
        r_obs = spec_std[idx] ** 2
        gaps = np.diff(idx)
        cov = (
            np.diag(r_obs[:-1] + r_obs[1:])
            - np.diag(r_obs[1:-1], 1)
            - np.diag(r_obs[1:-1], -1)
        )
        scale = np.diag(1 / np.sqrt(gaps))
        expected = np.linalg.eigvalsh(scale @ cov @ scale)
        mu = contrast_eigenvalues(spec_std, mask)
        np.testing.assert_allclose(np.sort(mu), expected, rtol=1e-10)
        assert len(contrast_eigenvalues(np.ones(1))) == 0
        assert len(contrast_eigenvalues(np.ones(2))) == 1

    @pytest.mark.parametrize("q_rel", [3, 0.3, 0.03, 0.003])
    def test_matches_monte_carlo(self, q_rel: float) -> None:
        """Analytic table and cached filter share the same null distribution."""
        spec_std = _masked_profile(512, seed=3)
        mask = spec_std == 0
        sig_eta = np.sqrt(q_rel * np.median(spec_std[~mask]) ** 2)
        dist = TridiagonalSaddlepointDistribution(
            contrast_eigenvalues(spec_std, mask),
            sig_eta,
        )
        cache = prepare_gain_cache(spec_std, sig_eta, chan_mask=mask, diffuse=True)
        rng = np.random.default_rng(4)
        ntrials = 200000
        scores = np.concatenate(
            [
                kalman_filter_cached(rng.normal(0, spec_std, (20000, 512)), cache)
                for _ in range(ntrials // 20000)
            ],
        )
        np.testing.assert_allclose(
            scores.mean(),
            dist.mean,
            atol=5 * np.sqrt(dist.var / ntrials),
        )
        for prob in (1e-1, 1e-2, 1e-3):
            threshold = np.quantile(scores, 1 - prob)
            # Binomial 4-sigma tolerance in log SF.
            tol = 4 / np.sqrt(prob * ntrials)
            np.testing.assert_allclose(dist.logsf(threshold), np.log(prob), atol=tol)

    @pytest.mark.parametrize("q_rel", [3, 1, 0.3, 0.1, 0.03, 0.01, 0.003])
    def test_matches_imhof(self, q_rel: float) -> None:
        spec_std = _masked_profile(1024, seed=5)
        mask = spec_std == 0
        sig_eta = np.sqrt(q_rel * np.median(spec_std[~mask]) ** 2)
        dist = TridiagonalSaddlepointDistribution(
            contrast_eigenvalues(spec_std, mask),
            sig_eta,
        )
        sd = np.sqrt(dist.var)
        for nsd in (-1.0, 0.0, 0.5, 2.0, 4.0, 8.0):
            score = dist.mean + nsd * sd
            exact = np.log(dist.imhof_sf(score, epsabs=1e-16, epsrel=1e-11))
            if exact < np.log(1e-8):
                continue
            np.testing.assert_allclose(dist.logsf(score), exact, atol=0.02)

    def test_chi2_tail(self) -> None:
        """Deep tail vs exact chi-square (equal weights) to 1e-100."""
        dof = 10
        dist = QuadraticFormTail(np.ones(dof), const=-1.0)
        for logp in (-5, -30, -100, -230):
            x = stats.chi2.isf(np.exp(logp), dof) if logp > -200 else 300.0
            exact = stats.chi2.logsf(x, dof)
            np.testing.assert_allclose(dist.logsf(x / 2 - 1), exact, rtol=2e-3)

    def test_direct_saddlepoint_matches_table(self) -> None:
        spec_std = _masked_profile(256, seed=6)
        mu = contrast_eigenvalues(spec_std, spec_std == 0)
        dist = TridiagonalSaddlepointDistribution(mu, 0.2)
        sd = np.sqrt(dist.var)
        for nsd in (-1.5, 0.0, 1.0, 5.0, 20.0, 100.0):
            score = dist.mean + nsd * sd
            np.testing.assert_allclose(
                dist.logsf(score),
                dist.saddlepoint_logsf(score),
                atol=2e-3,
            )

    def test_logsf_properties(self) -> None:
        dist = TridiagonalSaddlepointDistribution(
            contrast_eigenvalues(np.ones(128)),
            0.1,
        )
        scores = np.linspace(dist.const - 5, dist.mean + 1e4, 2000)
        logsf = dist.logsf(scores)
        assert np.all(np.isfinite(logsf))
        assert np.all(logsf <= 0)
        assert np.all(np.diff(logsf) <= 1e-12)
        np.testing.assert_equal(dist.logsf(dist.const - 1), 0)
        assert isinstance(dist.logsf(dist.mean), float)
        np.testing.assert_allclose(dist.sf(dist.mean), np.exp(dist.logsf(dist.mean)))
        assert str(dist).startswith("TridiagonalSaddlepointDistribution")
        assert repr(dist) == str(dist)
        with pytest.raises(ValueError):
            QuadraticFormTail(np.array([1.0, -1.0]))


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
