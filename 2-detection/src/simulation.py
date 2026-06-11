# Backend-agnostic data generation utilities for Monte-Carlo simulations.
#
# Data is always generated in numpy so generation is decoupled from the
# compute backend.  Callers move the result to the desired device afterwards
# via get_data_on_device().

from __future__ import annotations

import numpy as np


def make_sigma_true(n_features: int, seed: int, normalize: str = "trace") -> np.ndarray:
    """Random complex HPD shape matrix.

    Parameters
    ----------
    n_features : int
    seed : int
        RNG seed, independent from the data seed.
    normalize : "trace" or "det"
        "trace" — Tr(Sigma) = p, matches Tyler / DCG trace normalization.
        "det"   — det(Sigma) = 1, matches scaled-Gaussian MLE convention.

    Returns
    -------
    np.ndarray of shape (n_features, n_features), complex128
    """
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n_features, n_features)) +
         1j * rng.standard_normal((n_features, n_features)))
    Sigma = A @ A.conj().T / n_features + np.eye(n_features)
    if normalize == "trace":
        Sigma *= n_features / np.trace(Sigma).real
    elif normalize == "det":
        Sigma /= np.abs(np.linalg.det(Sigma)) ** (1.0 / n_features)
    return Sigma.astype(np.complex128)


def generate_gaussian_data(
    n_trials: int,
    T_max: int,
    n_samples: int,
    n_features: int,
    Sigma_true: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Complex Gaussian data under H0: X_{t,n} ~ CN(0, Sigma_true) i.i.d.

    Returns
    -------
    np.ndarray of shape (n_trials, T_max, n_samples, n_features), complex128
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma_true)
    g = (rng.standard_normal((n_trials, T_max, n_samples, n_features)) +
         1j * rng.standard_normal((n_trials, T_max, n_samples, n_features))) / np.sqrt(2)
    return (g @ L.T.conj()).astype(np.complex128)


def generate_dcg_data(
    n_trials: int,
    T_max: int,
    n_samples: int,
    n_features: int,
    Sigma_true: np.ndarray,
    seed: int = 0,
    tau_shape: float = 1.0,
    tau_scale: float = 1.0,
) -> np.ndarray:
    """Complex DCG (SIRV) data under H0: X_{t,n} = sqrt(tau_n) * z, z ~ CN(0, Sigma_true).

    tau_n ~ Gamma(tau_shape, tau_scale) per sample, constant across dates — matching the
    MatAndText null hypothesis tested by DeterministicCompoundGaussianGLRT.

    Returns
    -------
    np.ndarray of shape (n_trials, T_max, n_samples, n_features), complex128
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma_true)
    g = (rng.standard_normal((n_trials, T_max, n_samples, n_features)) +
         1j * rng.standard_normal((n_trials, T_max, n_samples, n_features))) / np.sqrt(2)
    # tau_n fixed per sample across all dates: shape (n_trials, 1, n_samples, 1)
    tau = rng.gamma(tau_shape, tau_scale, size=(n_trials, 1, n_samples, 1))
    return (np.sqrt(tau) * (g @ L.T.conj())).astype(np.complex128)


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


def T_vec_logspace(T_min: int, T_max: int, n_T: int) -> list[int]:
    """Logarithmically spaced unique integer T values in [T_min, T_max]."""
    return sorted({int(v) for v in np.unique(
        np.logspace(np.log10(T_min), np.log10(T_max), n_T).astype(int)
    )})


# ---------------------------------------------------------------------------
# Kronecker structured scaled Gaussian (SIRV) data generation
# ---------------------------------------------------------------------------

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


def generate_kronecker_data(
    n_trials: int,
    T_max: int,
    n_samples: int,
    a: int,
    b: int,
    A_true: np.ndarray,
    B_true: np.ndarray,
    seed: int = 0,
    tau_shape: float = 1.0,
    tau_scale: float = 1.0,
) -> np.ndarray:
    """Kronecker SIRV data under H0: x_{t,n} ~ CN(0, tau_{t,n} * kron(A, B)).

    Uses the identity vec_F(M) ~ CN(0, kron(A,B)) when M = L_B @ G @ L_A^H,
    G ~ CN(0, I_{b×a}). Texture tau_{t,n} ~ Gamma(tau_shape, tau_scale),
    drawn independently per sample and per date.

    Parameters
    ----------
    n_trials, T_max, n_samples : int
    a, b : int   — Kronecker factor sizes, p = a*b
    A_true, B_true : np.ndarray  — SHPD ground truth
    seed : int
    tau_shape, tau_scale : float  — Gamma texture parameters

    Returns
    -------
    np.ndarray of shape (n_trials, T_max, n_samples, p), complex128
    """
    p = a * b
    rng = np.random.default_rng(seed)
    L_A = np.linalg.cholesky(A_true)   # (a, a)
    L_B = np.linalg.cholesky(B_true)   # (b, b)
    G = (
        rng.standard_normal((n_trials, T_max, n_samples, b, a)) +
        1j * rng.standard_normal((n_trials, T_max, n_samples, b, a))
    ) / np.sqrt(2)
    tau = rng.gamma(tau_shape, tau_scale, size=(n_trials, T_max, n_samples, 1, 1))
    # M = sqrt(tau) * L_B @ G @ L_A^H, shape (..., b, a)
    M = np.sqrt(tau) * (L_B @ G @ L_A.conj().T)
    # Fortran-order flatten M (b×a) → x (p,): swapaxes then C-reshape = vec_F
    return M.swapaxes(-1, -2).reshape(n_trials, T_max, n_samples, p).astype(np.complex128)


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
    tau_shape: float = 1.0,
    tau_scale: float = 1.0,
) -> np.ndarray:
    """Kronecker SIRV data under H1: change point at date n_change_dates.

    Dates 0..n_change_dates-1: kron(A1, B1) with Gamma textures.
    Dates n_change_dates..T_max-1: kron(A2, B2) with fresh Gamma textures.

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

    tau0 = rng.gamma(tau_shape, tau_scale, size=(n_trials, n_change_dates, n_samples, 1, 1))
    tau1 = rng.gamma(tau_shape, tau_scale, size=(n_trials, n_h1, n_samples, 1, 1))

    M0 = np.sqrt(tau0) * (L_B1 @ G0 @ L_A1.conj().T)
    M1 = np.sqrt(tau1) * (L_B2 @ G1 @ L_A2.conj().T)

    X0 = M0.swapaxes(-1, -2).reshape(n_trials, n_change_dates, n_samples, p)
    X1 = M1.swapaxes(-1, -2).reshape(n_trials, n_h1, n_samples, p)
    return np.concatenate([X0, X1], axis=1).astype(np.complex128)
