import numpy as np
import pytest

from kalman_detector.core import (
    kalman_filter,
    kalman_filter_binary,
    kalman_filter_cached,
    prepare_gain_cache,
)


class TestKalmanFilter:
    def test_kalman_filter(self) -> None:
        nchans = 128
        target = 5
        rng = np.random.default_rng()
        spec_std = rng.normal(1, 0.1, size=nchans)
        spec = rng.normal(target, spec_std)
        score = kalman_filter.py_func(spec, spec_std, 0.1)
        mask = np.zeros(nchans, dtype=bool)
        mask[rng.choice(nchans, 2, replace=False)] = True
        score_masked = kalman_filter.py_func(spec, spec_std, 0.1, chan_mask=mask)
        np.testing.assert_allclose(score, score_masked, rtol=1e-1)


class TestKalmanDetectorBinary:
    @pytest.mark.parametrize("v0", [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    @pytest.mark.parametrize("nchans", [2, 128, 4096])
    def test_consistency_v0(self, v0: float, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        sig_t = rng.lognormal(0, 0.3)
        e0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter.py_func(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_filter_binary(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman1d, kalman2d, decimal=10)

    @pytest.mark.parametrize("e0", [-10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10])
    @pytest.mark.parametrize("nchans", [2, 128, 4096])
    def test_consistency_e0(self, e0: float, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        sig_t = rng.lognormal(0, 0.3)
        v0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter.py_func(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_filter_binary(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman2d, kalman1d, decimal=10)
        # Test without v0
        kalman1d_v0 = kalman_filter.py_func(spec, spec_std, sig_t, e0=e0)
        kalman2d_v0 = kalman_filter_binary(spec, spec_std, sig_t, e0=e0)
        np.testing.assert_almost_equal(kalman1d_v0, kalman2d_v0, decimal=10)

    @pytest.mark.parametrize("sig_t", [0.01, 0.1, 1, 10, 100, 1000])
    @pytest.mark.parametrize("nchans", [2, 128, 1024])
    def test_consistency_eta(self, sig_t: float, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        e0 = rng.lognormal(0, 0.3)
        v0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter.py_func(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_filter_binary(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman2d, kalman1d, decimal=10)

    @pytest.mark.parametrize("nchans", [2, 4, 16, 64, 256, 1024, 4096, 8192])
    def test_consistency_nchans(self, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        sig_t = rng.lognormal(0, 0.3)
        e0 = rng.lognormal(0, 0.3)
        v0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter.py_func(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_filter_binary(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman2d, kalman1d, decimal=10)


def _random_profile(
    rng: np.random.Generator,
    nchans: int,
) -> tuple[np.ndarray, np.ndarray]:
    spec_std = rng.lognormal(0, 0.3, nchans)
    mask = np.zeros(nchans, dtype=bool)
    if nchans >= 16:
        mask[0] = True
        mask[rng.choice(nchans, nchans // 16, replace=False)] = True
        mask[nchans // 3 : nchans // 3 + nchans // 8] = True
    return spec_std, mask


def _dense_contrast_llr(
    spec: np.ndarray,
    spec_std: np.ndarray,
    sig_eta: float,
    mask: np.ndarray,
) -> float:
    """Compute the reference contrast LR from gap-scaled adjacent differences."""
    idx = np.flatnonzero(~mask)
    r_obs = spec_std[idx] ** 2
    gaps = np.diff(idx)
    cov = (
        np.diag(r_obs[:-1] + r_obs[1:])
        - np.diag(r_obs[1:-1], 1)
        - np.diag(r_obs[1:-1], -1)
    )
    scale = np.diag(1 / np.sqrt(gaps))
    noise = scale @ cov @ scale
    signal = noise + sig_eta**2 * np.eye(len(gaps))
    y = scale @ np.diff(spec[idx])
    logdet_noise = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(noise))))
    logdet_signal = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(signal))))
    quad = y @ np.linalg.solve(noise, y) - y @ np.linalg.solve(signal, y)
    return -0.5 * (logdet_signal - logdet_noise) + 0.5 * quad


class TestKalmanFilterDiffuse:
    @pytest.mark.parametrize("sig_eta", [0.03, 0.3, 1, 5])
    @pytest.mark.parametrize("nchans", [2, 17, 128])
    def test_matches_dense_contrast_llr(self, sig_eta: float, nchans: int) -> None:
        rng = np.random.default_rng(42)
        spec_std, mask = _random_profile(rng, nchans)
        spec = rng.normal(0.5, spec_std)
        score = kalman_filter.py_func(
            spec,
            spec_std,
            sig_eta,
            chan_mask=mask,
            diffuse=True,
        )
        expected = _dense_contrast_llr(spec, spec_std, sig_eta, mask)
        np.testing.assert_allclose(score, expected, rtol=1e-10, atol=1e-10)

    def test_v0_inf_is_diffuse(self) -> None:
        rng = np.random.default_rng(1)
        spec_std, mask = _random_profile(rng, 256)
        spec = rng.normal(0, spec_std)
        np.testing.assert_allclose(
            kalman_filter(spec, spec_std, 0.3, v0=np.inf, chan_mask=mask),
            kalman_filter(spec, spec_std, 0.3, chan_mask=mask, diffuse=True),
            rtol=1e-14,
        )

    def test_shift_and_reversal_invariance(self) -> None:
        rng = np.random.default_rng(2)
        spec_std, mask = _random_profile(rng, 512)
        spec = rng.normal(0, spec_std)
        score = kalman_filter(spec, spec_std, 0.2, chan_mask=mask, diffuse=True)
        shifted = kalman_filter(
            spec + 7.5,
            spec_std,
            0.2,
            chan_mask=mask,
            diffuse=True,
        )
        reversed_score = kalman_filter(
            spec[::-1].copy(),
            spec_std[::-1].copy(),
            0.2,
            chan_mask=mask[::-1].copy(),
            diffuse=True,
        )
        np.testing.assert_allclose(shifted, score, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(reversed_score, score, rtol=1e-10, atol=1e-10)

    def test_single_channel(self) -> None:
        spec_std = np.ones(8)
        mask = np.ones(8, dtype=bool)
        mask[3] = False
        score = kalman_filter(np.ones(8), spec_std, 0.3, chan_mask=mask, diffuse=True)
        np.testing.assert_allclose(score, 0, atol=1e-14)


class TestKalmanFilterCached:
    @pytest.mark.parametrize("diffuse", [True, False])
    @pytest.mark.parametrize("sig_eta", [0.01, 0.3, 3, 100])
    @pytest.mark.parametrize("nchans", [2, 128, 1024])
    def test_matches_uncached(self, diffuse: bool, sig_eta: float, nchans: int) -> None:
        rng = np.random.default_rng(3)
        spec_std, mask = _random_profile(rng, nchans)
        spec = rng.normal(0.3, spec_std)
        e0 = 0.7
        cache = prepare_gain_cache(
            spec_std,
            sig_eta,
            chan_mask=mask,
            e0=e0,
            diffuse=diffuse,
        )
        expected = kalman_filter(
            spec,
            spec_std,
            sig_eta,
            e0=e0,
            chan_mask=mask,
            diffuse=diffuse,
        )
        np.testing.assert_allclose(
            kalman_filter_cached(spec, cache),
            expected,
            rtol=1e-12,
            atol=1e-10,
        )

    def test_default_v0_ignores_masked_channels(self) -> None:
        rng = np.random.default_rng(6)
        spec_std = rng.lognormal(0, 0.3, 128)
        mask = np.zeros(128, dtype=bool)
        mask[:60] = True
        spec_std[mask] = 0  # masked channels flagged by zero std
        spec = rng.normal(0, np.where(mask, 1, spec_std))
        v0 = np.median(spec_std[~mask]) ** 2
        expected = kalman_filter(spec, spec_std, 0.2, v0=v0, chan_mask=mask)
        np.testing.assert_allclose(
            kalman_filter(spec, spec_std, 0.2, chan_mask=mask),
            expected,
        )
        cache = prepare_gain_cache(spec_std, 0.2, chan_mask=mask)
        np.testing.assert_allclose(kalman_filter_cached(spec, cache), expected)

    def test_legacy_binary_requires_power_of_two(self) -> None:
        rng = np.random.default_rng(7)
        spec_std = rng.lognormal(0, 0.3, 64)
        spec = rng.normal(0, spec_std)
        np.testing.assert_allclose(
            kalman_filter_binary(spec, spec_std, 0.3),
            kalman_filter(spec, spec_std, 0.3),
            rtol=1e-8,
        )
        with pytest.raises(IndexError):
            kalman_filter_binary(spec[:48], spec_std[:48], 0.3)

    def test_explicit_v0(self) -> None:
        rng = np.random.default_rng(4)
        spec_std, mask = _random_profile(rng, 256)
        spec = rng.normal(0, spec_std)
        cache = prepare_gain_cache(spec_std, 0.2, chan_mask=mask, e0=-0.4, v0=2.5)
        expected = kalman_filter(spec, spec_std, 0.2, e0=-0.4, v0=2.5, chan_mask=mask)
        np.testing.assert_allclose(kalman_filter_cached(spec, cache), expected)

    def test_masked_channels_ignore_nan(self) -> None:
        rng = np.random.default_rng(5)
        spec_std, mask = _random_profile(rng, 128)
        spec = rng.normal(0, spec_std)
        spec_nan = spec.copy()
        spec_nan[mask] = np.nan
        cache = prepare_gain_cache(spec_std, 0.2, chan_mask=mask, diffuse=True)
        np.testing.assert_allclose(
            kalman_filter_cached(spec_nan, cache),
            kalman_filter_cached(spec, cache),
        )

    def test_batch(self) -> None:
        rng = np.random.default_rng(6)
        spec_std, mask = _random_profile(rng, 256)
        specs = rng.normal(0, spec_std, size=(50, 256))
        cache = prepare_gain_cache(spec_std, 0.1, chan_mask=mask, diffuse=True)
        batch = kalman_filter_cached(specs, cache)
        single = [kalman_filter_cached(spec, cache) for spec in specs]
        np.testing.assert_allclose(batch, single, rtol=1e-14)

    def test_fail(self) -> None:
        spec_std = np.ones(8)
        with pytest.raises(ValueError):
            prepare_gain_cache(spec_std, 0.1, chan_mask=np.ones(8, dtype=bool))
        cache = prepare_gain_cache(spec_std, 0.1)
        with pytest.raises(ValueError):
            kalman_filter_cached(np.ones((2, 2, 8)), cache)
