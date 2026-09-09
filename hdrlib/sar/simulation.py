# SAR-specific data generation utilities for Monte-Carlo simulations.
#
# Data is always generated in numpy so generation is decoupled from the
# compute backend.  Callers move the result to the desired device afterwards
# via get_data_on_device().

from __future__ import annotations

import numpy as np

from ..core.simulation import make_sigma_true


def generate_gaussian_data_h1(
    n_trials: int,
    T_max: int,
    n_samples: int,
    n_features: int,
    Sigma_1: np.ndarray,
    Sigma_2: np.ndarray,
    seed: int = 0,
    n_change_dates: int = 2,
) -> np.ndarray:
    """Gaussian data under H1: change point at date n_change_dates.

    Dates 0..n_change_dates-1 drawn from CN(0, Sigma_1),
    dates n_change_dates..T_max-1 drawn from CN(0, Sigma_2).

    Returns
    -------
    np.ndarray of shape (n_trials, T_max, n_samples, n_features), complex128
    """
    assert n_change_dates < T_max, "n_change_dates must be < T_max"
    rng = np.random.default_rng(seed)
    L1 = np.linalg.cholesky(Sigma_1)
    L2 = np.linalg.cholesky(Sigma_2)
    n_h1 = T_max - n_change_dates
    g0 = (rng.standard_normal((n_trials, n_change_dates, n_samples, n_features)) +
          1j * rng.standard_normal((n_trials, n_change_dates, n_samples, n_features))) / np.sqrt(2)
    g1 = (rng.standard_normal((n_trials, n_h1, n_samples, n_features)) +
          1j * rng.standard_normal((n_trials, n_h1, n_samples, n_features))) / np.sqrt(2)
    part0 = (g0 @ L1.T.conj()).astype(np.complex128)
    part1 = (g1 @ L2.T.conj()).astype(np.complex128)
    return np.concatenate([part0, part1], axis=1)


def generate_dcg_data_h1(
    n_trials: int,
    T_max: int,
    n_samples: int,
    n_features: int,
    Sigma_1: np.ndarray,
    Sigma_2: np.ndarray,
    seed: int = 0,
    n_change_dates: int = 2,
    tau_shape: float = 1.0,
    tau_scale: float = 1.0,
) -> np.ndarray:
    """DCG data under H1: change point at date n_change_dates.

    Same structure as generate_gaussian_data_h1 but with Gamma textures.

    Returns
    -------
    np.ndarray of shape (n_trials, T_max, n_samples, n_features), complex128
    """
    assert n_change_dates < T_max, "n_change_dates must be < T_max"
    rng = np.random.default_rng(seed)
    L1 = np.linalg.cholesky(Sigma_1)
    L2 = np.linalg.cholesky(Sigma_2)
    n_h1 = T_max - n_change_dates
    g0 = (rng.standard_normal((n_trials, n_change_dates, n_samples, n_features)) +
          1j * rng.standard_normal((n_trials, n_change_dates, n_samples, n_features))) / np.sqrt(2)
    g1 = (rng.standard_normal((n_trials, n_h1, n_samples, n_features)) +
          1j * rng.standard_normal((n_trials, n_h1, n_samples, n_features))) / np.sqrt(2)
    # tau fixed per sample within each segment, independent draws before/after change point
    tau0 = rng.gamma(tau_shape, tau_scale, size=(n_trials, 1, n_samples, 1))
    tau1 = rng.gamma(tau_shape, tau_scale, size=(n_trials, 1, n_samples, 1))
    part0 = (np.sqrt(tau0) * (g0 @ L1.T.conj())).astype(np.complex128)
    part1 = (np.sqrt(tau1) * (g1 @ L2.T.conj())).astype(np.complex128)
    return np.concatenate([part0, part1], axis=1)


def make_ab_true(
    a: int,
    b: int,
    seed_a: int = 0,
    seed_b: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Random SHPD(a) and SHPD(b) matrices (det=1) for Kronecker simulations.

    Parameters
    ----------
    a, b : int
        Sizes of the two Kronecker factors.
    seed_a, seed_b : int
        Independent seeds for A and B.

    Returns
    -------
    A : np.ndarray of shape (a, a), complex128, det ≈ 1
    B : np.ndarray of shape (b, b), complex128, det ≈ 1
    """
    return (
        make_sigma_true(a, seed_a, normalize="det"),
        make_sigma_true(b, seed_b, normalize="det"),
    )


def make_ab_toeplitz(
    a: int,
    b: int,
    rho_a: complex = 0.3 + 0.7j,
    rho_b: complex = 0.3 + 0.6j,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-determinant Toeplitz Kronecker factors, [A]_ij = rho^|i-j|.

    This is the covariance model actually used to produce the figures of the
    reference below: its Section 5.1 describes random orthogonal factors with
    condition number 10 in the body text, but the published figure captions and
    the released code both use Toeplitz factors. Kept as the default here so
    the figures of this repository match the published ones.

    **Reference**

    A. Mian, G. Ginolhac, F. Bouchard and A. Breloy, "Online change detection in
    SAR time-series with Kronecker product structured scaled Gaussian models",
    *Signal Processing*, 224:109589, 2024.
    [doi:10.1016/j.sigpro.2024.109589](https://doi.org/10.1016/j.sigpro.2024.109589)

    Parameters
    ----------
    a, b : int
        Sizes of the two Kronecker factors.
    rho_a, rho_b : complex
        Toeplitz correlation coefficients, |rho| < 1.

    Returns
    -------
    A : np.ndarray of shape (a, a), complex128, det = 1
    B : np.ndarray of shape (b, b), complex128, det = 1
    """
    def _toeplitz_det1(n: int, rho: complex) -> np.ndarray:
        idx = np.arange(n)
        d = idx[:, None] - idx[None, :]
        M = np.where(d >= 0, np.power(rho, np.abs(d)), np.conj(np.power(rho, np.abs(d))))
        M = M.astype(np.complex128)
        return M / np.abs(np.linalg.det(M)) ** (1.0 / n)

    return _toeplitz_det1(a, rho_a), _toeplitz_det1(b, rho_b)


def generate_kronecker_data(
    n_trials: int,
    T_max: int,
    n_samples: int,
    a: int,
    b: int,
    A_true: np.ndarray,
    B_true: np.ndarray,
    seed: int = 0,
    tau_shape: "float | None" = 1.0,
    tau_scale: float = 1.0,
    tau_per_date: bool = False,
    return_tau: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    r"""Kronecker SIRV data under $\mathcal{H}_0$:

    $$
    \boldsymbol{x}_{t,k} \sim \mathcal{CN}\!\left(\boldsymbol{0},
    \tau_k \, \boldsymbol{A} \otimes \boldsymbol{B}\right),
    \qquad \boldsymbol{A} \in \mathbb{C}^{a \times a}, \;
           \boldsymbol{B} \in \mathbb{C}^{b \times b}, \; d = ab .
    $$

    Uses the identity $\operatorname{vec}(\boldsymbol{M}) \sim
    \mathcal{CN}(\boldsymbol{0}, \boldsymbol{A} \otimes \boldsymbol{B})$ when
    $\boldsymbol{M} = \boldsymbol{L}_B \boldsymbol{G}
    \boldsymbol{L}_A^{\mathrm{T}}$ with $\boldsymbol{G} \sim
    \mathcal{CN}(\boldsymbol{0}, \boldsymbol{I}_{b \times a})$. Note the
    transpose, not the conjugate transpose: $\boldsymbol{M} = \boldsymbol{L}_B
    \boldsymbol{G} \boldsymbol{L}_A^{\mathrm{H}}$ would give
    $\overline{\boldsymbol{A}} \otimes \boldsymbol{B}$ instead, which is
    indistinguishable in online-vs-offline comparisons but wrong as soon as
    an estimate is compared to the ground truth A. Texture tau_n ~ Gamma(tau_shape, tau_scale), drawn once
    per sample and held FIXED across dates.

    Holding tau fixed across dates is what H0 means for this model: the null
    hypothesis is theta^(t) = theta^(0) for every t, and theta = {A, B, tau}
    includes the textures. Re-drawing tau at each date leaves nothing for the
    estimator to converge to, so the ICRB on tau cannot be approached and the
    H0 statistic no longer telescopes. Same convention as generate_dcg_data().

    Parameters
    ----------
    n_trials, T_max, n_samples : int
    a, b : int   — Kronecker factor sizes, p = a*b
    A_true, B_true : np.ndarray  — SHPD ground truth
    seed : int
    tau_shape, tau_scale : float  — Gamma texture parameters
    tau_per_date : bool
        Re-draw the textures at every date. NOT the H0 model — kept only to
        reproduce earlier runs. Default False.
    return_tau : bool
        Also return the ground-truth textures, needed for MSE/ICRB studies.

    Returns
    -------
    X : np.ndarray of shape (n_trials, T_max, n_samples, p), complex128
    tau : np.ndarray of shape (n_trials, n_samples), float64
        Only when return_tau is True (and tau_per_date is False).
    """
    p = a * b
    rng = np.random.default_rng(seed)
    L_A = np.linalg.cholesky(A_true)   # (a, a)
    L_B = np.linalg.cholesky(B_true)   # (b, b)
    G = (
        rng.standard_normal((n_trials, T_max, n_samples, b, a)) +
        1j * rng.standard_normal((n_trials, T_max, n_samples, b, a))
    ) / np.sqrt(2)
    tau_size = (n_trials, T_max, n_samples, 1, 1) if tau_per_date else (n_trials, 1, n_samples, 1, 1)
    # tau_shape None means deterministic unit texture, i.e. the Gaussian sub-case.
    tau = (np.ones(tau_size) if tau_shape is None
           else rng.gamma(tau_shape, tau_scale, size=tau_size))
    # M = sqrt(tau) * L_B @ G @ L_A^H, shape (..., b, a)
    M = np.sqrt(tau) * (L_B @ G @ L_A.T)
    # Fortran-order flatten M (b×a) → x (p,): swapaxes then C-reshape = vec_F
    X = M.swapaxes(-1, -2).reshape(n_trials, T_max, n_samples, p).astype(np.complex128)
    if not return_tau:
        return X
    if tau_per_date:
        raise ValueError("return_tau is meaningless when tau_per_date is True: "
                         "there is no single ground-truth texture vector to compare to.")
    return X, tau[:, 0, :, 0, 0]


def generate_kronecker_data_h1(
    n_trials: int,
    T_max: int,
    n_samples: int,
    a: int,
    b: int,
    A1: np.ndarray,
    B1: np.ndarray,
    A2: np.ndarray,
    B2: np.ndarray,
    seed: int = 0,
    n_change_dates: int = 2,
    tau_shape: "float | None" = 1.0,
    tau_scale: float = 1.0,
    tau_per_date: bool = False,
) -> np.ndarray:
    """Kronecker SIRV data under H1: change point at date n_change_dates.

    Dates 0..n_change_dates-1: kron(A1, B1) with textures tau^(0).
    Dates n_change_dates..T_max-1: kron(A2, B2) with fresh textures tau^(1).

    Each segment holds its texture vector fixed across its dates, so that each
    segment is a valid H0 stretch and the change is exactly a change of
    theta = {A, B, tau}. Set tau_per_date to re-draw at every date (not the
    model; kept only to reproduce earlier runs).

    Returns
    -------
    np.ndarray of shape (n_trials, T_max, n_samples, p), complex128
    """
    assert n_change_dates < T_max, "n_change_dates must be < T_max"
    p = a * b
    n_h1 = T_max - n_change_dates
    rng = np.random.default_rng(seed)

    L_A1, L_B1 = np.linalg.cholesky(A1), np.linalg.cholesky(B1)
    L_A2, L_B2 = np.linalg.cholesky(A2), np.linalg.cholesky(B2)

    G0 = (
        rng.standard_normal((n_trials, n_change_dates, n_samples, b, a)) +
        1j * rng.standard_normal((n_trials, n_change_dates, n_samples, b, a))
    ) / np.sqrt(2)
    G1 = (
        rng.standard_normal((n_trials, n_h1, n_samples, b, a)) +
        1j * rng.standard_normal((n_trials, n_h1, n_samples, b, a))
    ) / np.sqrt(2)

    size0 = (n_trials, n_change_dates, n_samples, 1, 1) if tau_per_date else (n_trials, 1, n_samples, 1, 1)
    size1 = (n_trials, n_h1, n_samples, 1, 1) if tau_per_date else (n_trials, 1, n_samples, 1, 1)
    if tau_shape is None:
        tau0, tau1 = np.ones(size0), np.ones(size1)
    else:
        tau0 = rng.gamma(tau_shape, tau_scale, size=size0)
        tau1 = rng.gamma(tau_shape, tau_scale, size=size1)

    M0 = np.sqrt(tau0) * (L_B1 @ G0 @ L_A1.T)
    M1 = np.sqrt(tau1) * (L_B2 @ G1 @ L_A2.T)

    X0 = M0.swapaxes(-1, -2).reshape(n_trials, n_change_dates, n_samples, p)
    X1 = M1.swapaxes(-1, -2).reshape(n_trials, n_h1, n_samples, p)
    return np.concatenate([X0, X1], axis=1).astype(np.complex128)
