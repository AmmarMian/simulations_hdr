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
