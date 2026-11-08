import numpy as np
import pytest
import sympy as sp

from kalman_detector import svm
from kalman_detector.core import kalman_filter
from kalman_detector.svm import State


class TestKalmanSVM:
    def test_init_state(self) -> None:
        rng = np.random.default_rng()
        d0, d1, var_0, var_1, var_t = rng.random(5) * 2 + 0.5
        x_vec = rng.normal(0, 1, [100, 2])
        num_result = (
            np.exp(
                -0.5
                * (
                    (x_vec[:, 0] - d0) ** 2 / var_0
                    + (x_vec[:, 1] - d1) ** 2 / var_1
                    + (x_vec[:, 0] - x_vec[:, 1]) ** 2 / var_t
                ),
            )
            / (2 * np.pi) ** (3 / 2)
            / (var_0 * var_1 * var_t) ** 0.5
        )

        state = State.init_from_data(d0, d1, var_0, var_1, var_t)
        return np.testing.assert_array_almost_equal(
            state.apply(x_vec),
            num_result,
            decimal=15,
        )

    def test_init_state_f01(self) -> None:
        rng = np.random.default_rng()
        d0, d1, var_0, var_1, var_t, e0, v0 = rng.random(7) * 2 + 0.5
        x_vec = rng.normal(0, 1, [100, 2])
        num_result = (
            np.exp(
                -0.5
                * (
                    (x_vec[:, 0] - d0) ** 2 / var_0
                    + (x_vec[:, 1] - d1) ** 2 / var_1
                    + (x_vec[:, 0] - x_vec[:, 1]) ** 2 / var_t
                    + (x_vec[:, 0] - e0) ** 2 / v0
                ),
            )
            / (2 * np.pi) ** (4 / 2)
            / (var_0 * var_1 * var_t * v0) ** 0.5
        )

        state = State.init_from_data_f01(d0, d1, var_0, var_1, var_t, e0, v0)
        return np.testing.assert_array_almost_equal(
            state.apply(x_vec),
            num_result,
            decimal=15,
        )

    def test_addition_rule(self) -> None:
        rng = np.random.default_rng()
        d0, d1, d2, d3, var_0, var_1, var_2, var_3, var_t = rng.random(9) * 2 + 0.5

        state0 = State.init_from_data(d0, d1, var_0, var_1, var_t)
        state1 = State.init_from_data(d2, d3, var_2, var_3, var_t)
        final_state = state0 + state1

        sig1 = var_1**0.5
        sig2 = var_2**0.5
        dx1 = sig1 / 15
        dx2 = sig2 / 15

        num_result = np.zeros([10, 10])
        x0_vec = rng.normal(0, 1, 10)
        x3_vec = rng.normal(0, 1, 10)

        for i, x0 in enumerate(x0_vec):
            for j, x3 in enumerate(x3_vec):
                grid_x1, grid_x2 = np.mgrid[
                    d1 - 5 * sig1 : d1 + 5 * sig1 : dx1,
                    d2 - 5 * sig2 : d2 + 5 * sig2 : dx2,
                ]
                inp0 = np.vstack(
                    [np.ones(np.prod(grid_x1.shape)) * x0, grid_x1.flatten()],
                ).T
                inp1 = np.vstack(
                    [grid_x2.flatten(), np.ones(np.prod(grid_x1.shape)) * x3],
                ).T
                num_result[i, j] += (
                    np.sum(
                        state0.apply(inp0)
                        * state1.apply(inp1)
                        * np.exp(-((inp0[:, 1] - inp1[:, 0]) ** 2) / (2 * var_t)),
                    )
                    / (2 * np.pi * var_t) ** 0.5
                    * dx1
                    * dx2
                )
        grid_03 = np.array(
            [
                (x0_vec[i], x3_vec[j])
                for i in range(len(x0_vec))
                for j in range(len(x3_vec))
            ],
        )
        state_result = final_state.apply(grid_03).reshape([len(x0_vec), len(x3_vec)])
        return np.testing.assert_array_almost_equal(
            state_result,
            num_result,
            decimal=10,
        )

    def test_addition_rule_simple(self) -> None:
        rng = np.random.default_rng()
        d0, d1, d2, d3, var_0, var_1, var_2, var_3, var_t = rng.random(9) * 2 + 0.5

        state0 = State.init_from_data(d0, d1, var_0, var_1, var_t)
        state1 = State.init_from_data(d2, d3, var_2, var_3, var_t)
        final_state = state0.add_simple(state1)

        sig1 = var_1**0.5
        sig2 = var_2**0.5
        dx1 = sig1 / 15
        dx2 = sig2 / 15

        num_result = np.zeros([10, 10])
        x0_vec = rng.normal(0, 1, 10)
        x3_vec = rng.normal(0, 1, 10)

        for i, x0 in enumerate(x0_vec):
            for j, x3 in enumerate(x3_vec):
                grid_x1, grid_x2 = np.mgrid[
                    d1 - 5 * sig1 : d1 + 5 * sig1 : dx1,
                    d2 - 5 * sig2 : d2 + 5 * sig2 : dx2,
                ]
                inp0 = np.vstack(
                    [np.ones(np.prod(grid_x1.shape)) * x0, grid_x1.flatten()],
                ).T
                inp1 = np.vstack(
                    [grid_x2.flatten(), np.ones(np.prod(grid_x1.shape)) * x3],
                ).T
                num_result[i, j] += (
                    np.sum(
                        state0.apply(inp0)
                        * state1.apply(inp1)
                        * np.exp(-((inp0[:, 1] - inp1[:, 0]) ** 2) / (2 * var_t)),
                    )
                    / (2 * np.pi * var_t) ** 0.5
                    * dx1
                    * dx2
                )
        grid_03 = np.array(
            [
                (x0_vec[i], x3_vec[j])
                for i in range(len(x0_vec))
                for j in range(len(x3_vec))
            ],
        )
        state_result = final_state.apply(grid_03).reshape([len(x0_vec), len(x3_vec)])
        return np.testing.assert_array_almost_equal(
            state_result,
            num_result,
            decimal=10,
        )


class TestSVM:
    def test_collect2(self) -> None:
        x, y = sp.symbols("x y")
        expr = (x + 2 * y) ** 3
        np.testing.assert_equal(svm.collect2(expr, x, 3, y, 0), 1)
        np.testing.assert_equal(svm.collect2(expr, x, 2, y, 1), 6)
        np.testing.assert_equal(svm.collect2(expr, x, 1, y, 2), 12)
        np.testing.assert_equal(svm.collect2(expr, x, 0, y, 3), 8)
        np.testing.assert_equal(svm.collect2(expr, x, 0, y, 0), 0)

    def test_derive_addition_rule(self) -> None:
        M2, M2V2, S2_factor, res_mul = svm.derive_addition_rule()
        assert isinstance(M2, sp.Matrix)
        assert isinstance(M2V2, sp.Matrix)
        assert isinstance(S2_factor, sp.Add)
        assert isinstance(res_mul, sp.Mul)

    def test_derive_addition_rule_simple(self) -> None:
        M2, M2V2, S2_factor, res_mul = svm.derive_addition_rule_simple()
        assert isinstance(M2, sp.Matrix)
        assert isinstance(M2V2, sp.Matrix)
        assert isinstance(S2_factor, sp.Add)
        assert isinstance(res_mul, sp.Mul)

    def test_addition_rul_equivalence(self) -> None:
        M2, M2V2, S2_factor, _res_mul = svm.derive_addition_rule()
        (
            M2_simple,
            M2V2_simple,
            S2_factor_simple,
            _res_mul_simple,
        ) = svm.derive_addition_rule_simple()
        assert M2.equals(M2_simple)
        assert M2V2.equals(M2V2_simple)
        assert S2_factor.equals(S2_factor_simple)


def _mask_case(nchans: int, case: str) -> np.ndarray:
    mask = np.zeros(nchans, dtype=bool)
    if case == "first":
        mask[:3] = True
    elif case == "last":
        mask[-2:] = True
    elif case == "quarter":
        mask[nchans // 4 : nchans // 2] = True
    elif case == "random":
        mask[np.random.default_rng(nchans).random(nchans) < 0.3] = True
    mask[nchans // 2] = False  # keep at least one observed channel
    return mask


class TestBinaryCompressDiffuse:
    @pytest.mark.parametrize("nchans", [2, 3, 17, 64, 1000, 1024])
    @pytest.mark.parametrize("case", ["none", "first", "last", "quarter", "random"])
    @pytest.mark.parametrize("q", [1e-6, 1e-3, 0.1, 10.0])
    def test_matches_kalman_filter(self, nchans: int, case: str, q: float) -> None:
        rng = np.random.default_rng(nchans)
        spec_std = 10 ** rng.uniform(-1, 1, nchans)
        mask = _mask_case(nchans, case)
        spec = rng.normal(0, spec_std) + 2 * np.sin(np.arange(nchans) / 7)
        sig_eta = np.sqrt(q) * np.median(spec_std)
        expected = kalman_filter(spec, spec_std, sig_eta, chan_mask=mask, diffuse=True)
        result = svm.kalman_binary_compress_diffuse(spec, spec_std, sig_eta, mask)
        # Measured worst relative error ~4e-12 (stiff chains included).
        assert abs(result - expected) <= 1e-10 * max(1.0, abs(expected))

    def test_masked_data_ignored_and_edges_unit_factor(self) -> None:
        rng = np.random.default_rng(7)
        spec_std = rng.lognormal(0, 0.3, 64)
        spec = rng.normal(0, spec_std)
        mask = np.zeros(64, dtype=bool)
        mask[:5] = True
        mask[-9:] = True
        spec_nan = spec.copy()
        spec_nan[mask] = np.nan
        full = svm.kalman_binary_compress_diffuse(spec_nan, spec_std, 0.3, mask)
        cut = svm.kalman_binary_compress_diffuse(spec[5:-9], spec_std[5:-9], 0.3)
        np.testing.assert_allclose(full, cut, rtol=1e-12, atol=1e-12)

    def test_bank_reuse_and_validation(self) -> None:
        rng = np.random.default_rng(8)
        spec_std = rng.lognormal(0, 0.3, 100)
        bank = svm.binary_tree_bank(spec_std, 0.2)
        assert len(bank.weights) == 128
        assert set(bank.levels) == {4, 8, 16, 32, 64, 128}
        assert bank.levels[8].shape == (16, 5)
        np.testing.assert_allclose(bank.inv_var_sum, np.sum(1 / spec_std**2))
        spec = rng.normal(0, spec_std)
        np.testing.assert_allclose(
            svm.kalman_binary_compress_diffuse(spec, spec_std, 0.2, bank=bank),
            svm.kalman_binary_compress_diffuse(spec, spec_std, 0.2),
        )
        with pytest.raises(ValueError, match="channels"):
            svm.kalman_binary_compress_diffuse(spec[:50], spec_std, 0.2, bank=bank)
        with pytest.raises(ValueError, match="positive"):
            svm.binary_tree_bank(spec_std, 0)
        with pytest.raises(ValueError, match="masked"):
            svm.binary_tree_bank(spec_std, 0.2, np.ones(100, dtype=bool))

    def test_offset_invariance(self) -> None:
        rng = np.random.default_rng(9)
        spec_std = rng.lognormal(0, 0.3, 256)
        spec = rng.normal(0, spec_std)
        base = svm.kalman_binary_compress_diffuse(spec, spec_std, 0.1)
        shifted = svm.kalman_binary_compress_diffuse(spec + 5.0, spec_std, 0.1)
        np.testing.assert_allclose(base, shifted, rtol=1e-9)
