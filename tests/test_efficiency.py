import numpy as np

from kalman_detector.efficiency import Result, Results


class TestEfficiency:
    def test_trial_stats_match_numpy(self) -> None:
        snr = np.array([1.0, 3.0, 5.0])
        results = Results(
            [Result(snr_emp=snr, kal_score=np.array([2.0, 4.0, 6.0]), kal_sig=snr)],
        )

        snr_stats = results.snr_emp
        np.testing.assert_allclose(snr_stats.mean, np.mean(snr))
        np.testing.assert_allclose(snr_stats.std, np.std(snr))

    def test_efficiency_is_aggregated_per_trial(self) -> None:
        result = Result(
            snr_emp=np.array([1.0, 3.0]),
            kal_score=np.array([1.0, 3.0]),
            kal_sig=np.zeros(2),
        )
        per_trial = 2 * result.kal_score / result.snr_emp**2

        efficiency = Results([result]).efficiency

        np.testing.assert_allclose(efficiency.mean, np.mean(per_trial))
        np.testing.assert_allclose(efficiency.std, np.std(per_trial))
        np.testing.assert_allclose(per_trial, [2.0, 2.0 / 3.0])
