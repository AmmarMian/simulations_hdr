# Backend-agnostic data generation utilities for Monte-Carlo simulations.
# Generic functions shared across modalities.
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
    r"""Complex Gaussian data under $\mathcal{H}_0$, i.i.d. across dates and samples:

    $$
    \boldsymbol{x}_{t,k} \sim
    \mathcal{CN}\!\left(\boldsymbol{0}, \boldsymbol{\Sigma}\right),
    \qquad t \in \{1,\dots,T\}, \; k \in \{1,\dots,N\}.
    $$

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
    r"""Complex deterministic compound-Gaussian (SIRV) data under $\mathcal{H}_0$:

    $$
    \boldsymbol{x}_{t,k} = \sqrt{\tau_k}\, \boldsymbol{z}_{t,k},
    \qquad \boldsymbol{z}_{t,k} \sim
      \mathcal{CN}\!\left(\boldsymbol{0}, \boldsymbol{\Sigma}\right),
    \qquad \tau_k \sim \mathcal{G}(\text{shape}, \text{scale}).
    $$

    The texture is drawn once per sample and held constant across the dates,
    which is what the MatAndText null hypothesis of
    :class:`~hdrlib.sar.detectors.ScaleAndShapeGLRT` asserts.

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


def T_vec_logspace(T_min: int, T_max: int, n_T: int) -> list[int]:
    """Logarithmically spaced unique integer T values in [T_min, T_max]."""
    return sorted({int(v) for v in np.unique(
        np.logspace(np.log10(T_min), np.log10(T_max), n_T).astype(int)
    )})
