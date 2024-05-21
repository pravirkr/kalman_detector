import numpy as np
import sympy as sp

from kalman_detector import svm
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
        M2, M2V2, S2_factor, res_mul = svm.derive_addition_rule()
        (
            M2_simple,
            M2V2_simple,
            S2_factor_simple,
            res_mul_simple,
        ) = svm.derive_addition_rule_simple()
        assert M2.equals(M2_simple)
        assert M2V2.equals(M2V2_simple)
        assert S2_factor.equals(S2_factor_simple)
