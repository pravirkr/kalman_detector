from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

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

    Legacy seven-scalar moment form of Kumar, Zackay & Law (2024), Appendix B,
    kept for reproducibility. :class:`BinaryTreeBank` and
    :func:`kalman_binary_compress_diffuse` implement the canonical
    information-form tree used for new work.

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
        M000, _M001, M010, M011 = self.m.flatten()
        M100, _M101, M110, M111 = other.m.flatten()
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
        M000, _M001, M010, M011 = self.m.flatten()
        M100, _M101, M110, M111 = other.m.flatten()
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

    Legacy moment-form tree of Kumar, Zackay & Law (2024), Appendix B, kept to
    reproduce that paper (see :func:`~kalman_detector.core.kalman_filter_binary`).
    Requires a power-of-two number of channels; no channel mask. For new work
    use :func:`kalman_binary_compress_diffuse`.

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


class BinaryTreeBank(NamedTuple):
    """Data-independent coefficients of the diffuse canonical binary tree.

    The spectrum is modelled as a gaussian random walk ``A_f`` (step variance
    ``sig_eta**2``, flat prior on the level) observed with independent noise of
    precision ``w_f = 1 / spec_std[f]**2`` (``w_f = 0`` for masked channels).
    Every node ``[lo, hi)`` of the binary tree carries the 2x2 precision of its
    end-point states; merging two siblings integrates out the two junction
    states. Only the five folded coefficients of each merge are stored.

    Attributes
    ----------
    weights : np.ndarray
        Channel precisions ``w_f`` padded with zeros to ``npad`` (a power of two).
    levels : dict[int, np.ndarray]
        ``levels[size][lo // size] = (p00, p01, p11, m0, m1)`` for the merge
        producing node ``[lo, lo + size)``. ``p = G / 2`` with ``G`` the inverse
        of the junction Schur complement, ``m0 = -2 l`` and ``m1 = -2 r`` with
        ``l`` and ``r`` the end-point couplings of the left and right child.
    root : np.ndarray
        ``(G00 / 2, G01, G11 / 2)`` of the inverse root end-point precision.
    tree_const : float
        Data-independent part of the full (diffuse) log likelihood ratio.
    inv_var_sum : float
        ``V_tot = sum(w_f)``.
    nchans : int
        Number of channels before padding.
    """

    weights: np.ndarray
    levels: dict[int, np.ndarray]
    root: np.ndarray
    tree_const: float
    inv_var_sum: float
    nchans: int

    @property
    def contrast_const(self) -> float:
        """Constant ``c_det`` of the mean-free contrast ``raw - F**2/(2 V) + c_det``."""
        return self.tree_const + 0.5 * np.log(self.inv_var_sum / (2 * np.pi))


def binary_tree_bank(
    spec_std: np.ndarray,
    sig_eta: float,
    chan_mask: np.ndarray | None = None,
) -> BinaryTreeBank:
    """Per-node merge coefficients for heterogeneous noise and channel masks.

    Parameters
    ----------
    spec_std : np.ndarray
        1D array of per-channel noise standard deviations. Channels with
        non-positive or non-finite std are treated as masked.
    sig_eta : float
        State transition std (same units as ``spec_std``).
    chan_mask : np.ndarray, optional
        Boolean mask of channels to ignore, by default None.

    Returns
    -------
    BinaryTreeBank
        Coefficients for a tree over ``npad = 2**ceil(log2(nchans))`` channels.
        Padding channels are masked; masked channels at any position (including
        the band edges or whole subtrees) leave the score unchanged, because an
        unobserved random-walk segment integrates to one.

    Raises
    ------
    ValueError
        If ``sig_eta`` is not positive or all channels are masked.
    """
    spec_std = np.asarray(spec_std, dtype=np.float64)
    nchans = len(spec_std)
    if not sig_eta > 0:
        msg = "sig_eta must be positive."
        raise ValueError(msg)
    valid = np.isfinite(spec_std) & (spec_std > 0)
    if chan_mask is not None:
        valid &= ~np.asarray(chan_mask, dtype=bool)
    if not np.any(valid):
        msg = "All channels are masked."
        raise ValueError(msg)
    npad = max(2, 1 << (nchans - 1).bit_length())
    weights = np.zeros(npad)
    weights[:nchans][valid] = 1 / spec_std[valid] ** 2
    q = float(sig_eta) ** 2
    # Node end-point precision J = [[s0 + k, -k], [-k, s1 + k]]: coupling k > 0
    # and row sums s0, s1 >= 0 (observed information). Tracking (s0, s1, k)
    # instead of J avoids the cancellation j00 + j01 for stiff chains, and every
    # update below is a sum of non-negative terms.
    # Leaf pairs [2i, 2i+2): one transition between the two channels.
    s0 = weights[0::2].copy()
    s1 = weights[1::2].copy()
    k = np.full(npad // 2, 1 / q)
    tree_const = -0.5 * (npad // 2) * np.log(2 * np.pi * q)
    levels = {}
    size = 4
    while size <= npad:
        kl, kr = k[0::2], k[1::2]
        sl0, sl1, sr0, sr1 = s0[0::2], s1[0::2], s0[1::2], s1[1::2]
        a, b = sl1 + kl, sr0 + kr
        den = a + b + q * a * b
        g00, g01, g11 = (1 + q * b) / den, 1 / den, (1 + q * a) / den
        # Couplings l = -kl, r = -kr, so m0 = -2 l = 2 kl and m1 = 2 kr.
        levels[size] = np.column_stack(
            (0.5 * g00, 0.5 * g01, 0.5 * g11, 2 * kl, 2 * kr),
        )
        # Integrating out the two junction states and their transition.
        tree_const += float(np.sum(0.5 * np.log(2 * np.pi / den)))
        s0 = sl0 + kl * (sl1 * (1 + q * b) + sr0) / den
        s1 = sr1 + kr * (sr0 * (1 + q * a) + sl1) / den
        k = kl * kr / den
        size *= 2
    s0, s1, k = s0[0], s1[0], k[0]
    det = s0 * s1 + k * (s0 + s1)
    root = np.array([0.5 * (s1 + k) / det, k / det, 0.5 * (s0 + k) / det])
    # Flat (diffuse) prior on the two end-point states of the whole band.
    tree_const += np.log(2 * np.pi) - 0.5 * np.log(det)
    return BinaryTreeBank(
        weights=weights,
        levels=levels,
        root=root,
        tree_const=float(tree_const),
        inv_var_sum=float(np.sum(weights)),
        nchans=nchans,
    )


def kalman_binary_compress_diffuse(
    spec: np.ndarray,
    spec_std: np.ndarray,
    sig_t: float,
    chan_mask: np.ndarray | None = None,
    bank: BinaryTreeBank | None = None,
) -> float:
    """Diffuse Kalman contrast score by canonical binary-tree reduction.

    Each node holds the float4 state ``(F, h0, h1, c)``: the inverse-variance
    weighted sum, the two end-point information components and the quadratic
    accumulator. Each merge costs 16 FLOPs per state (see ``binary_fdmt``):

    ``u = p00 a + p01 b``, ``v = p01 a + p11 b``, ``F' = FL + FR``,
    ``h0' = hL0 + m0 u``, ``h1' = hR1 + m1 v``, ``c' = cL + cR + a u + b v``
    with ``a = hL1`` and ``b = hR0``.

    Parameters
    ----------
    spec : np.ndarray
        1D array of spectrum values (any length).
    spec_std : np.ndarray
        1D array of spectrum (noise) standard deviations.
    sig_t : float
        Standard deviation of the transition between states.
    chan_mask : np.ndarray, optional
        Boolean mask of channels to ignore, by default None.
    bank : BinaryTreeBank, optional
        Precomputed :func:`binary_tree_bank` for ``(spec_std, sig_t, chan_mask)``.

    Returns
    -------
    float
        The diffuse contrast score, equal to
        ``kalman_filter(spec, spec_std, sig_t, chan_mask=chan_mask, diffuse=True)``.
    """
    if bank is None:
        bank = binary_tree_bank(spec_std, sig_t, chan_mask)
    spec = np.asarray(spec, dtype=np.float64)
    if len(spec) != bank.nchans:
        msg = f"spec has {len(spec)} channels, bank has {bank.nchans}."
        raise ValueError(msg)
    weights = bank.weights
    info = np.zeros(len(weights))
    obs = weights[: bank.nchans] > 0
    info[: bank.nchans][obs] = weights[: bank.nchans][obs] * spec[obs]
    h0, h1 = info[0::2], info[1::2]
    flux = h0 + h1
    quad = np.zeros_like(h0)
    for size in sorted(bank.levels):
        p00, p01, p11, m0, m1 = bank.levels[size].T
        a, b = h1[0::2], h0[1::2]
        u = p00 * a + p01 * b
        v = p01 * a + p11 * b
        flux = flux[0::2] + flux[1::2]
        h0 = h0[0::2] + m0 * u
        h1 = h1[1::2] + m1 * v
        quad = quad[0::2] + quad[1::2] + a * u + b * v
    r0, r1, r2 = bank.root
    raw = quad[0] + r0 * h0[0] ** 2 + r1 * h0[0] * h1[0] + r2 * h1[0] ** 2
    return float(raw - 0.5 * flux[0] ** 2 / bank.inv_var_sum + bank.contrast_const)
