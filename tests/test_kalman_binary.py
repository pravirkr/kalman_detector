import numpy as np
import pytest

from kalman_detector.core import kalman_filter
from kalman_detector.svm import kalman_binary_hypothesis


class TestKalmanDetector2D:
    @pytest.mark.parametrize("v0", [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    @pytest.mark.parametrize("nchans", [2, 128, 4096])
    def test_consistency_v0(self, v0: float, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        sig_t = rng.lognormal(0, 0.3)
        e0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman1d, kalman2d, decimal=10)

    @pytest.mark.parametrize("e0", [-10, -1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10])
    @pytest.mark.parametrize("nchans", [2, 128, 4096])
    def test_consistency_e0(self, e0: float, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        sig_t = rng.lognormal(0, 0.3)
        v0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman2d, kalman1d, decimal=10)

    @pytest.mark.parametrize("sig_t", [0.01, 0.1, 1, 10, 100, 1000])
    @pytest.mark.parametrize("nchans", [2, 128, 1024])
    def test_consistency_eta(self, sig_t: float, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        e0 = rng.lognormal(0, 0.3)
        v0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman2d, kalman1d, decimal=10)

    @pytest.mark.parametrize("nchans", [2, 4, 16, 64, 256, 1024, 4096, 8192])
    def test_consistency_nchans(self, nchans: int) -> None:
        rng = np.random.default_rng()
        spec = rng.lognormal(0, 0.3, nchans)
        spec_std = rng.lognormal(0, 0.3, nchans)
        sig_t = rng.lognormal(0, 0.3)
        e0 = rng.lognormal(0, 0.3)
        v0 = rng.lognormal(0, 0.3)

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_almost_equal(kalman2d, kalman1d, decimal=10)
