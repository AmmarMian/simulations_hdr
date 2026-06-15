"""Sonar two-array MSG data model — simulation helpers.

All functions return numpy arrays; callers move to device via
``hdrlib.core.backend.get_data_on_device`` when needed.

Model
-----
Two co-located ULAs (horizontal H, vertical V) crossing at their centres.
Primary observation (one range bin, H0 / H1)::

    H0: x = z             z ~ Sigma_tilde M Sigma_tilde
    H1: x = P alpha + z   P = blkdiag(p1, p2), alpha = [a1, a2]^T

where Sigma_tilde = diag(sigma_tilde_1, sigma_tilde_2) ⊗ I_m and
sigma_tilde_i = sigma_i * sqrt(tau_i).

For simulations sigma_i = 1 and tau_i drawn from the chosen distribution
(tau=1 ↔ Gaussian, tau ~ Gamma(nu, 1/nu) ↔ K-distributed).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Covariance model
# ---------------------------------------------------------------------------

def make_sonar_covariance(
    m: int,
    beta: float = 3e-4,
    rho1: float = 0.4,
    rho2: float = 0.9,
) -> np.ndarray:
    """Two-array Toeplitz cross-covariance M of shape (2m, 2m).

    Sensor j ∈ {0,...,2m-1} has 2-D coordinates (j_x, j_y) determined by
    the cross geometry: array 1 runs horizontally through the centre of the
    field and array 2 runs vertically through the same centre.

    M[j, l] = beta * rho1^|j_x - l_x| * rho2^|j_y - l_y|

    With the nominal parameters (beta=3e-4, rho1=0.4, rho2=0.9):
      - M_11 is weakly correlated (Toeplitz with rho1=0.4)
      - M_22 is strongly correlated (Toeplitz with rho2=0.9)
      - M_12 is non-zero only near the crossing centre
    """
    # The crossing point is placed at a half-integer coordinate so that no
    # sensor from either array occupies the exact same 2-D position.
    # Array 1 (horizontal): element j at position (j,   c_half), y = c_half
    # Array 2 (vertical):   element j at position (c_half, j), x = c_half
    # Using c_half = m/2 - 0.5 (= (m-1)/2 as a float) ensures the crossing
    # lies between sensors for any m — avoiding the perfectly correlated
    # (singular) case that arises when m is even and c = m//2 is integer.
    c_half = m / 2.0 - 0.5   # e.g. 3.5 for m=8, 31.5 for m=64

    j_idx = np.arange(m, dtype=float)
    # Array 1 positions: (j, c_half)  — x varies, y is fixed
    x1 = j_idx
    y1 = np.full(m, c_half)
    # Array 2 positions: (c_half, j) — x is fixed, y varies
    x2 = np.full(m, c_half)
    y2 = j_idx

    xs = np.concatenate([x1, x2])   # (2m,)
    ys = np.concatenate([y1, y2])   # (2m,)

    dx = np.abs(xs[:, None] - xs[None, :])   # (2m, 2m)
    dy = np.abs(ys[:, None] - ys[None, :])   # (2m, 2m)
    M = beta * (rho1 ** dx) * (rho2 ** dy)
    return M.astype(np.complex128)


# ---------------------------------------------------------------------------
# Steering matrix
# ---------------------------------------------------------------------------

def make_steering_matrix(
    m: int,
    theta1_deg: float,
    theta2_deg: float,
    d_over_lambda: float = 0.5,
) -> np.ndarray:
    """Steering matrix P = blkdiag(p1, p2) of shape (2m, 2).

    Array 1 steers to azimuth theta1, array 2 to elevation theta2.
    Half-wavelength spacing (d/lambda = 0.5) is the default.
    """
    idx = np.arange(m)
    p1 = np.exp(1j * 2 * np.pi * d_over_lambda * np.sin(np.deg2rad(theta1_deg)) * idx)
    p2 = np.exp(1j * 2 * np.pi * d_over_lambda * np.sin(np.deg2rad(theta2_deg)) * idx)

    P = np.zeros((2 * m, 2), dtype=np.complex128)
    P[:m, 0] = p1
    P[m:, 1] = p2
    return P


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_sonar_data_h0(
    n_trials: int,
    m: int,
    M: np.ndarray,
    tau_shape: float = 1.0,
    tau_scale: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Primary data under H0 (no target): x = sqrt(tau) * CN(0, M).

    Gaussian clutter: tau_shape=1, tau_scale=1 (tau=1 deterministically).
    K-distributed:   tau ~ Gamma(nu, 1/nu) → tau_shape=nu, tau_scale=1/nu.

    Returns
    -------
    np.ndarray of shape (n_trials, 2m), complex128
    """
    p = 2 * m
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(M)
    g = (rng.standard_normal((n_trials, p)) +
         1j * rng.standard_normal((n_trials, p))) / np.sqrt(2)
    c = (g @ L.T.conj()).astype(np.complex128)
    if tau_shape == 1.0 and tau_scale == 1.0:
        return c
    tau = rng.gamma(tau_shape, tau_scale, size=(n_trials, 1))
    return np.sqrt(tau) * c


def generate_sonar_data_h1(
    n_trials: int,
    m: int,
    M: np.ndarray,
    P: np.ndarray,
    alpha: float,
    tau_shape: float = 1.0,
    tau_scale: float = 1.0,
    seed: int = 1,
) -> np.ndarray:
    """Primary data under H1: x = P * [alpha, alpha]^T + z.

    Equal amplitudes on both arrays: alpha_1 = alpha_2 = alpha.

    Returns
    -------
    np.ndarray of shape (n_trials, 2m), complex128
    """
    z = generate_sonar_data_h0(n_trials, m, M, tau_shape, tau_scale, seed)
    a = np.array([alpha, alpha], dtype=np.complex128)
    signal = P @ a  # (2m,)
    return z + signal[None, :]


def generate_secondary_data(
    n_trials: int,
    K: int,
    m: int,
    M: np.ndarray,
    tau_shape: float = 1.0,
    tau_scale: float = 1.0,
    seed: int = 100,
) -> np.ndarray:
    """K signal-free secondary samples per trial for covariance estimation.

    Returns
    -------
    np.ndarray of shape (n_trials, K, 2m), complex128
    """
    p = 2 * m
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(M)
    g = (rng.standard_normal((n_trials, K, p)) +
         1j * rng.standard_normal((n_trials, K, p))) / np.sqrt(2)
    c = (g @ L.T.conj()).astype(np.complex128)
    if tau_shape == 1.0 and tau_scale == 1.0:
        return c
    tau = rng.gamma(tau_shape, tau_scale, size=(n_trials, K, 1))
    return np.sqrt(tau) * c


# ---------------------------------------------------------------------------
# SNR sweep
# ---------------------------------------------------------------------------

def snr_alpha_sweep(
    m: int,
    M: np.ndarray,
    P: np.ndarray,
    snr_min_db: float = -25.0,
    snr_max_db: float = 5.0,
    n_snr: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (snr_db, alphas) for a logarithmic SNR sweep.

    SNR is defined as |alpha|^2 * (p1^H M11^{-1} p1 + p2^H M22^{-1} p2) / (2m),
    which normalises by the beamformed signal-to-noise ratio per array element.

    Returns
    -------
    snr_db : np.ndarray (n_snr,)
    alphas : np.ndarray (n_snr,), real, such that SNR_i = |alpha_i|^2 * norm_factor
    """
    M_inv = np.linalg.inv(M)
    p1 = P[:m, 0]
    p2 = P[m:, 1]
    M_inv_11 = M_inv[:m, :m]
    M_inv_22 = M_inv[m:, m:]
    norm_factor = float(np.real(p1.conj() @ M_inv_11 @ p1 + p2.conj() @ M_inv_22 @ p2)) / (2 * m)

    snr_db = np.linspace(snr_min_db, snr_max_db, n_snr)
    snr_lin = 10.0 ** (snr_db / 10.0)
    alphas = np.sqrt(snr_lin / norm_factor)
    return snr_db, alphas
