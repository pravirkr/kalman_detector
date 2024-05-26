from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


def collect2(expr: sp.Expr, v1: sp.Symbol, p1: int, v2: sp.Symbol, p2: int) -> sp.Expr:
    """Collect the coefficient of v1**p1 and v2**p2 in expr.

    Parameters
    ----------
    expr : sp.Expr
        The expression in which to collect the coefficients.
    v1 : sp.Symbol
        The variable for which to collect the coefficient.
    p1 : int
        The power of the variable v1 for which to collect the coefficient.
    v2 : sp.Symbol
        The variable for which to collect the coefficient.
    p2 : int
        The power of the variable v2 for which to collect the coefficient.

    Returns
    -------
    sp.Expr
        The coefficient of v1**p1 and v2**p2 in expr.
    """
    return expr.expand().collect(v1).coeff(v1, p1).collect(v2).coeff(v2, p2)


def derive_addition_rule() -> tuple:
    """Derive the addition rule for the Binary Kalman.

    Returns
    -------
    _type_
        Addition rule expressions for the Binary Kalman.
    """
    M000, M010, M011 = sp.symbols("M000, M010, M011")
    M0 = sp.Matrix([[M000, M010], [M010, M011]])
    V0 = sp.Matrix([sp.symbols("v00, v01")]).T
    A0 = sp.Matrix([sp.symbols("x00, x01")]).T

    M100, M110, M111 = sp.symbols("M100, M110, M111")
    M1 = sp.Matrix([[M100, M110], [M110, M111]])
    V1 = sp.Matrix([sp.symbols("v10, v11")]).T
    A1 = sp.Matrix([sp.symbols("x10, x11")]).T

    # Dealing with the first Gaussian (Eqn. B11-B12)
    exp_term0 = -1 / 2 * ((A0 - V0).T * M0 * (A0 - V0)).det()
    exp_term0_expand = sp.collect(exp_term0.expand(), A0[1])
    alpha_0 = -1 / (2 * exp_term0_expand.coeff(A0[1], 2))
    beta_0 = alpha_0 * exp_term0_expand.coeff(A0[1], 1)
    gamma_0 = -(beta_0**2) / alpha_0 - 2 * exp_term0_expand.coeff(A0[1], 0)

    pred = -0.5 * ((A0[1] - beta_0) ** 2 / alpha_0 + gamma_0)
    logger.debug(f"is it zero?: {(pred - exp_term0).expand()}")

    # Dealing with the last Gaussian (Eqn. B14-B15)
    exp_term1 = -1 / 2 * ((A1 - V1).T * M1 * (A1 - V1)).det()
    exp_term1_expand = sp.collect(exp_term1.expand(), A1[0])
    alpha_1 = -1 / (2 * exp_term1_expand.coeff(A1[0], 2))
    beta_1 = exp_term1_expand.coeff(A1[0], 1) * alpha_1
    gamma_1 = -(beta_1**2) / alpha_1 - 2 * exp_term1_expand.coeff(A1[0], 0)

    pred = -0.5 * ((A1[0] - beta_1) ** 2 / alpha_1 + gamma_1)
    logger.debug(f"is it zero?: {(pred - exp_term1).expand()}")

    # Dealing with the middle Gaussian
    s_t = sp.var("s_t")

    # Combining terms (Eqn. B19)
    alpha_2 = alpha_0 + alpha_1 + s_t**2
    beta_2 = beta_1 - beta_0
    gamma_2 = gamma_0 + gamma_1

    # final integral (Eqn. B20)
    res_exp = (-(beta_2**2) / (2 * alpha_2) - gamma_2 / 2).expand()
    res_mul = sp.sqrt((2 * np.pi * alpha_0 * alpha_1) / alpha_2)

    # (After equating coeffecients of Eqn. B20 and B22)
    M200 = -2 * collect2(res_exp, A0[0], 2, A1[1], 0)
    M211 = -2 * collect2(res_exp, A0[0], 0, A1[1], 2)
    M210 = -collect2(res_exp, A0[0], 1, A1[1], 1)
    M201 = -collect2(res_exp, A0[0], 1, A1[1], 1)
    M2V20 = collect2(res_exp, A0[0], 1, A1[1], 0)
    M2V21 = collect2(res_exp, A0[0], 0, A1[1], 1)
    S2_factor = collect2(res_exp, A0[0], 0, A1[1], 0)

    M2 = sp.Matrix([[M200, M201], [M210, M211]])
    M2V2 = sp.Matrix([M2V20, M2V21])

    return M2, M2V2, S2_factor, res_mul


def derive_addition_rule_simple() -> tuple:
    M000, M010, M011 = sp.symbols("M000, M010, M011")
    M100, M110, M111 = sp.symbols("M100, M110, M111")
    v00, v01 = sp.symbols("v00, v01")
    v10, v11 = sp.symbols("v10, v11")
    s_t = sp.var("s_t")

    c = 1 / (M011 * M100 * s_t**2 + M011 + M100)
    n = sp.Matrix([[M000 - M010**2 / M011, 0], [0, M111 - M110**2 / M100]])
    q = sp.Matrix(
        [
            [M010**2 * M100 / M011, -M010 * M110],
            [-M010 * M110, M110**2 * M011 / M100],
        ],
    )
    w = sp.Matrix([[v00 + v01 * M011 / M010], [v10 * M100 / M110 + v11]])
    u = sp.Matrix([[v00], [v11]])

    M2 = n + c * q
    M2V2 = n @ u + c * q @ w
    res_mul = sp.sqrt(2 * np.pi * c)
    S2_factor = -0.5 * (np.squeeze(u.T @ n @ u + c * w.T @ q @ w))
    return M2, M2V2, S2_factor, res_mul


@dataclass
class State:
    """Complex State of the Binary Kalman.

    Parameters
    ----------
    log_s : float
        Scalar log of the state.
    m : np.ndarray
        Noise covariance matrix.
    v : np.ndarray
        Mean vector.
    var_t : float
        Variance of the transition.

    Notes
    -----
    Implements the distribution of
    int(dydz * exp(-0.5*((x,y) - v0)^t A0 ((x,y) - v0))
             * exp(-0.5*((z,w) - v1)^t A1 ((z,w) - v1))
             * exp(-0.5*(z-y)**2/V)).
    Assumes that M1, M2 are diagonal matrices.
    """

    log_s: float = 0
    m: np.ndarray = field(default_factory=lambda: np.zeros(shape=(2, 2)))
    v: np.ndarray = field(default_factory=lambda: np.zeros(shape=(2, 1)))
    var_t: float = 0

    @property
    def s_t(self) -> float:
        """Returns the standard deviation of the transition."""
        return self.var_t**0.5

    def apply(self, x_arr: np.ndarray) -> np.ndarray:
        """Evaluate the SVM distribution for given signal model A.

        Parameters
        ----------
        x_arr : np.ndarray
            Signal model A.

        Returns
        -------
        np.ndarray
            SVM distribution evaluated for given signal model A.
        """
        return np.array(
            [
                np.exp(self.log_s)
                * np.exp(
                    -0.5
                    * (
                        (x_arr[i] - self.v.squeeze()).T
                        @ self.m
                        @ (x_arr[i] - self.v.squeeze())
                    ),
                )
                for i in range(len(x_arr))
            ],
        )

    def add(self, other: State) -> State:
        """Add another state to the current state.

        Parameters
        ----------
        other : State
            State to be added.

        Returns
        -------
        State
            Sum of the two states.
        """
        M000, M001, M010, M011 = self.m.flatten()
        M100, M101, M110, M111 = other.m.flatten()
        v00, v01 = self.v.flatten()
        v10, v11 = other.v.flatten()
        s_t = self.s_t

        # The following formulae were derived using the "derive_addition_rule" function
        m_res = np.array(
            [
                [
                    M000
                    + 2.0
                    * M010**2
                    / (2 * M011**2 * s_t**2 + 2.0 * M011**2 / M100 + 2.0 * M011)
                    - 1.0 * M010**2 / M011,
                    -2.0
                    * M010
                    * M110
                    / (2 * M011 * M100 * s_t**2 + 2.0 * M011 + 2.0 * M100),
                ],
                [
                    -2.0
                    * M010
                    * M110
                    / (2 * M011 * M100 * s_t**2 + 2.0 * M011 + 2.0 * M100),
                    2.0
                    * M110**2
                    / (2 * M100**2 * s_t**2 + 2.0 * M100 + 2.0 * M100**2 / M011)
                    + M111
                    - 1.0 * M110**2 / M100,
                ],
            ],
        )

        mv_res = np.array(
            [
                [
                    1.0 * M000 * v00
                    + 2.0
                    * M010**2
                    * v00
                    / (2 * M011**2 * s_t**2 + 2.0 * M011**2 / M100 + 2.0 * M011)
                    - 1.0 * M010**2 * v00 / M011
                    + 2.0
                    * M010
                    * M011
                    * v01
                    / (2 * M011**2 * s_t**2 + 2.0 * M011**2 / M100 + 2.0 * M011)
                    - 2.0
                    * M010
                    * M110
                    * v11
                    / (2 * M011 * M100 * s_t**2 + 2.0 * M011 + 2.0 * M100)
                    - 2.0 * M010 * v10 / (2 * M011 * s_t**2 + 2.0 * M011 / M100 + 2.0),
                ],
                [
                    -2.0
                    * M010
                    * M110
                    * v00
                    / (2 * M011 * M100 * s_t**2 + 2.0 * M011 + 2.0 * M100)
                    + 2.0
                    * M100
                    * M110
                    * v10
                    / (2 * M100**2 * s_t**2 + 2.0 * M100 + 2.0 * M100**2 / M011)
                    + 2.0
                    * M110**2
                    * v11
                    / (2 * M100**2 * s_t**2 + 2.0 * M100 + 2.0 * M100**2 / M011)
                    - 2.0 * M110 * v01 / (2 * M100 * s_t**2 + 2.0 + 2.0 * M100 / M011)
                    + 1.0 * M111 * v11
                    - 1.0 * M110**2 * v11 / M100,
                ],
            ],
        )

        s_factor = (
            -0.5 * M000 * v00**2
            - 1.0
            * M010**2
            * v00**2
            / (2 * M011**2 * s_t**2 + 2.0 * M011**2 / M100 + 2.0 * M011)
            + 0.5 * M010**2 * v00**2 / M011
            - 2.0
            * M010
            * M011
            * v00
            * v01
            / (2 * M011**2 * s_t**2 + 2.0 * M011**2 / M100 + 2.0 * M011)
            + 2.0
            * M010
            * M110
            * v00
            * v11
            / (2 * M011 * M100 * s_t**2 + 2.0 * M011 + 2.0 * M100)
            + 2.0 * M010 * v00 * v10 / (2 * M011 * s_t**2 + 2.0 * M011 / M100 + 2.0)
            - 1.0
            * M011**2
            * v01**2
            / (2 * M011**2 * s_t**2 + 2.0 * M011**2 / M100 + 2.0 * M011)
            - 1.0
            * M100**2
            * v10**2
            / (2 * M100**2 * s_t**2 + 2.0 * M100 + 2.0 * M100**2 / M011)
            - 2.0
            * M100
            * M110
            * v10
            * v11
            / (2 * M100**2 * s_t**2 + 2.0 * M100 + 2.0 * M100**2 / M011)
            - 1.0
            * M110**2
            * v11**2
            / (2 * M100**2 * s_t**2 + 2.0 * M100 + 2.0 * M100**2 / M011)
            + 2.0 * M110 * v01 * v11 / (2 * M100 * s_t**2 + 2.0 + 2.0 * M100 / M011)
            - 0.5 * M111 * v11**2
            + 2.0 * v01 * v10 / (2 * s_t**2 + 2.0 / M100 + 2.0 / M011)
            + 0.5 * M110**2 * v11**2 / M100
        )

        res_mul = np.sqrt(2 * np.pi) * np.sqrt(
            1 / (M011 * M100 * (s_t**2 + 1.0 / M100 + 1.0 / M011)),
        )
        v_res = np.linalg.inv(m_res) @ mv_res
        log_s_res = (
            self.log_s
            + other.log_s
            + np.log(res_mul)
            + s_factor
            + 0.5 * np.squeeze(v_res.T @ mv_res)
        )
        return State(log_s_res, m_res, v_res, self.var_t)

    def add_simple(self, other: State) -> State:
        """Add another state to the current state.

        Parameters
        ----------
        other : State
            State to be added.

        Returns
        -------
        State
            Sum of the two states.
        """
        M000, M001, M010, M011 = self.m.flatten()
        M100, M101, M110, M111 = other.m.flatten()
        v00, v01 = self.v.flatten()
        v10, v11 = other.v.flatten()
        s_t = self.s_t

        c = 1 / (M011 * M100 * s_t**2 + M011 + M100)
        n = np.array([[M000 - M010**2 / M011, 0], [0, M111 - M110**2 / M100]])
        q = np.array(
            [
                [M010**2 * M100 / M011, -M010 * M110],
                [-M010 * M110, M110**2 * M011 / M100],
            ],
        )
        w = np.array([[v00 + v01 * M011 / M010], [v10 * M100 / M110 + v11]])
        u = np.array([[v00], [v11]])

        m_res = n + c * q
        mv_res = n @ u + c * q @ w
        v_res = np.linalg.inv(m_res) @ mv_res

        res_mul = np.sqrt(2 * np.pi * c)
        log_s_res = (
            self.log_s
            + other.log_s
            + np.log(res_mul)
            - 0.5 * (np.squeeze(u.T @ n @ u + c * w.T @ q @ w - v_res.T @ mv_res))
        )
        return State(log_s_res, m_res, v_res, self.var_t)

    @classmethod
    def init_from_data(
        cls,
        d0: float,
        d1: float,
        var_0: float,
        var_1: float,
        var_t: float,
    ) -> State:
        """Initialize the state for a pair of frequency channels.

        Parameters
        ----------
        d0 : float
            Observed spectrum for the first frequency channel.
        d1 : float
            Observed spectrum for the second frequency channel.
        var_0 : float
            Variance of the first frequency channel.
        var_1 : float
            Variance of the second frequency channel.
        var_t : float
            Variance of the state transition between two frequency channels.

        Returns
        -------
        State
            Initialized state.
        """
        m_init = np.array(
            [[1 / var_0 + 1 / var_t, -1 / var_t], [-1 / var_t, 1 / var_1 + 1 / var_t]],
        )
        v_init = np.linalg.inv(m_init) @ np.array([[d0 / var_0], [d1 / var_1]])
        s_init = np.squeeze(
            1
            / np.sqrt((2 * np.pi) ** 3 * var_0 * var_1 * var_t)
            * np.exp(
                0.5 * (v_init.T @ m_init @ v_init)
                - d0**2 / (2 * var_0)
                - d1**2 / (2 * var_1),
            ),
        )
        return cls(np.log(s_init), m_init, v_init, var_t)

    @classmethod
    def init_from_data_f01(
        cls,
        d0: float,
        d1: float,
        var_0: float,
        var_1: float,
        var_t: float,
        e0: float,
        v0: float,
    ) -> State:
        """Initialize the state for the first two frequency channels.

        Parameters
        ----------
        d0 : float
            Observed spectrum for the first frequency channel.
        d1 : float
            Observed spectrum for the second frequency channel.
        var_0 : float
            Variance of the first frequency channel.
        var_1 : float
            Variance of the second frequency channel.
        var_t : float
            Variance of the state transition between two frequency channels.
        e0 : float
            Expected value of the first hiden state A0.
        v0 : float
            Variance of the first hiden state A0.

        Returns
        -------
        State
            Initialized state.
        """
        m_init = np.array(
            [
                [1 / var_0 + 1 / var_t + 1 / v0, -1 / var_t],
                [-1 / var_t, 1 / var_1 + 1 / var_t],
            ],
        )
        v_init = np.linalg.inv(m_init) @ np.array(
            [[d0 / var_0 + e0 / v0], [d1 / var_1]],
        )
        s_init = np.squeeze(
            1
            / np.sqrt((2 * np.pi) ** 4 * var_0 * var_1 * var_t * v0)
            * np.exp(
                0.5 * (v_init.T @ m_init @ v_init)
                - d0**2 / (2 * var_0)
                - d1**2 / (2 * var_1)
                - e0**2 / (2 * v0),
            ),
        )
        return cls(np.log(s_init), m_init, v_init, var_t)

    def __add__(self, other: State) -> State:
        return self.add(other)


def kalman_binary_compress(
    spec: np.ndarray,
    spec_std: np.ndarray,
    sig_t: float,
    e0: float,
    v0: float,
) -> State:
    """Kalman binary compression for a spectrum.

    Parameters
    ----------
    spec : np.ndarray
        1D array of spectrum values.
    spec_std : np.ndarray
        1D array of spectrum standard deviations.
    sig_t : float
        Standard deviation of the tranition between states.
    e0 : float
        Initial guess of the expected value of the first hidden state A0.
    v0 : float
        Initial guess of the variance of the first hidden state A0.

    Returns
    -------
    State
        Final state for the whole spectrum.
    """
    var_d = spec_std**2
    var_t = sig_t**2
    states = [
        State.init_from_data_f01(spec[0], spec[1], var_d[0], var_d[1], var_t, e0, v0),
    ] + [
        State.init_from_data(spec[i], spec[i + 1], var_d[i], var_d[i + 1], var_t)
        for i in range(2, len(spec), 2)
    ]
    while len(states) > 1:
        states = [states[i] + states[i + 1] for i in range(0, len(states), 2)]
    return states[0]
