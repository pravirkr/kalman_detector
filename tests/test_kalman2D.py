import pytest
import numpy as np

from kalman_detector.core import kalman_filter
from kalman_detector.svm import kalman_binary_hypothesis


class TestKalmanDetector2D(object):
    @pytest.mark.parametrize("v0", [0.01, 0.1, 1, 10, 100, 1000])
    def test_consistency_v0(self, v0):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 64)
        spec_std = rng.normal(0, 1, 64)
        sig_t = rng.normal(0, 1)
        e0 = rng.normal(0, 1)

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_array_almost_equal(kalman1d, kalman2d, decimal=8)

    @pytest.mark.parametrize("e0", [0.01, 0.1, 0.5, 1, 5, 10])
    def test_consistency_e0(self, e0):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 64)
        spec_std = rng.normal(0, 1, 64)
        sig_t = rng.normal(0, 1)
        v0 = np.abs(rng.normal(0, 1))

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_array_almost_equal(kalman1d, kalman2d, decimal=8)

    @pytest.mark.parametrize("sig_t", [0.01, 0.1, 1, 10, 100, 1000])
    def test_consistency_eta(self, sig_t):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, 64)
        spec_std = rng.normal(0, 1, 64)
        e0 = rng.normal(0, 1)
        v0 = np.abs(rng.normal(0, 1))

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_array_almost_equal(kalman1d, kalman2d, decimal=8)

    @pytest.mark.parametrize("nchans", [2, 4, 16, 32, 64, 512])
    def test_consistency_nchans(self, nchans):
        rng = np.random.default_rng()
        spec = rng.normal(0, 1, nchans)
        spec_std = rng.normal(0, 1, nchans)
        sig_t = rng.normal(0, 1)
        e0 = rng.normal(0, 1)
        v0 = np.abs(rng.normal(0, 1))

        kalman1d = kalman_filter(spec, spec_std, sig_t, e0=e0, v0=v0)
        kalman2d = kalman_binary_hypothesis(spec, spec_std, sig_t, e0=e0, v0=v0)
        np.testing.assert_array_almost_equal(kalman1d, kalman2d, decimal=8)
