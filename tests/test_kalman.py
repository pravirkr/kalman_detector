import numpy as np

from kalman_detector.main import KalmanDetector


class TestKalmanDetector(object):
    def test_random(self):
        rng = np.random.default_rng()
        std_vec = np.arange(0, 1, 0.01)
        spectrum = rng.normal(0, std_vec)
        kalman = KalmanDetector(std_vec)
        kalman.prepare_fits(ntrials=1000)
        sig_kalman = kalman.get_significance(spectrum)
