# Intrinsic Cramer-Rao bounds and Riemannian error measures for the Kronecker
# structured scaled Gaussian model.
#
# Reference: A. Mian, G. Ginolhac, F. Bouchard, A. Breloy, "Online change
# detection in SAR time-series with Kronecker product structured scaled
# Gaussian models", Signal Processing 224 (2024), Propositions 2 and 4.
#
# Two conventions in that paper need care, and are resolved here in favour of
# the dimension count, which is what the released code also implements:
#
#   * Proposition 2 prints the weights a/p (on A) and b/p (on B) the wrong way
#     round.  Differentiating phi(theta) = tau_i A (x) B gives
#     tr(A^-1 xi_A A^-1 eta_A (x) I_b) = b tr(A^-1 xi_A A^-1 eta_A), so the
#     weight carried by the A term is b/p.  Equation (25) of the same paper is
#     consistent with b/p, and so is manifolds.KroneckerHermitianPositiveScaledGaussian
#     (weights (b/p, a/p, 1/n)).
#
#   * The third line of equation (25) reads E[d^2_tau] <= 1/(T p).  The
#     dimension count gives (1/n) E[d^2_tau] <= n/(T p n), hence
#     E[d^2_tau] <= n/(T p): a factor n is missing in the paper.

from __future__ import annotations

from typing import Union

import numpy as np

from ..core.backend import Array, Backend, get_backend_module
from ..core.manifolds import (
    KroneckerHermitianPositiveScaledGaussian,
    ScaledGaussianFIM,
)


def icrb_kronecker_scaled_gaussian(
    a: int,
    b: int,
    n_samples: int,
    T: "int | np.ndarray",
) -> dict:
    """Intrinsic Cramer-Rao bounds for (A, B, tau), per component and total.

    The bound on each component is the ratio between the dimension of that
    component and the amount of data, T * p * n, rescaled by the weight the
    component carries in the metric of Proposition 2.

    Parameters
    ----------
    a, b : int
        Kronecker factor sizes, p = a * b.
    n_samples : int
        Number of samples n in a patch.
    T : int or array of int
        Number of dates.

    Returns
    -------
    dict with keys "A", "B", "tau", "total", each a float or an array shaped
    like *T*.  "A", "B", "tau" bound the component squared geodesic distances
    d^2_{sH++}(.,.) and d^2_{R++^n}(.,.); "total" bounds d^2_M, the weighted
    sum of equation (18).
    """
    T_arr = np.asarray(T, dtype=float)
    p = a * b
    n = n_samples
    return {
        "A": (a**2 - 1) / (b * n * T_arr),
        "B": (b**2 - 1) / (a * n * T_arr),
        "tau": n / (p * T_arr),
        "total": ((a**2 - 1) + (b**2 - 1) + n) / (p * n * T_arr),
    }


def normalize_det1(
    A: Array,
    B: Array,
    tau: Array,
    a: int,
    b: int,
    backend_name: Union[str, Backend] = "numpy",
) -> tuple:
    """Move an (A, B, tau) triple to the unit-determinant representative.

    The estimators of sar.estimation_kronecker do not enforce |A| = |B| = 1;
    they return some representative of the equivalence class
    (A, B, tau) ~ (A/c_a, B/c_b, c_a c_b tau).  The parametrisation of the
    model, and the geometry of sH++, both require the unit-determinant one.
    Skipping this step shows up as a constant multiplicative offset on tau and
    a constant additive offset on the distances of A and B.

    Returns
    -------
    (A, B, tau) with |A| = |B| = 1 and the same kron(A, B) * tau product.
    """
    be = get_backend_module(backend_name)
    c_a = be.exp(be.real(be.linalg.slogdet(A)[1]) / a)
    c_b = be.exp(be.real(be.linalg.slogdet(B)[1]) / b)
    A_n = A / c_a[..., None, None]
    B_n = B / c_b[..., None, None]
    scale = c_a * c_b
    tau_n = tau * scale[..., None] if tau.ndim == A.ndim - 1 else tau * scale[..., None, None]
    return A_n, B_n, tau_n


def kronecker_component_errors(
    A: Array,
    B: Array,
    tau: Array,
    A_true: Array,
    B_true: Array,
    tau_true: Array,
    a: int,
    b: int,
    n_samples: int,
    backend_name: Union[str, Backend] = "numpy",
    normalize: bool = True,
    conjugate_A_true: bool = True,
) -> dict:
    """Squared Riemannian errors of an estimate, per component and total.

    Component distances are the geodesic distances of equation (18): the affine
    invariant distance on sH++ for A and B, and ||log(tau_true / tau)||_2 on
    R++^n for the textures.  The total is their weighted sum, i.e. d^2_M.

    Parameters
    ----------
    A : Array of shape (..., a, a)
    B : Array of shape (..., b, b)
    tau : Array of shape (..., n) or (..., n, 1)
    A_true, B_true, tau_true : Arrays, broadcastable against the estimates.
    a, b, n_samples : int
    backend_name : str or Backend
    normalize : bool
        Move the estimate to its unit-determinant representative first.
        Default True; turn it off only if the estimate is already normalised.
    conjugate_A_true : bool
        Conjugate the ground-truth A before comparing, to account for the
        Sigma = A^T (x) B convention of the estimators. Default True.

    Returns
    -------
    dict with keys "A", "B", "tau", "total", each an Array of the batch shape.
    """
    be = get_backend_module(backend_name)
    manifold = KroneckerHermitianPositiveScaledGaussian(a, b, n_samples, backend_name=backend_name)
    m_A, m_B, m_tau = manifold.manifolds
    w_A, w_B, w_tau = manifold.weights

    if normalize:
        A, B, tau = normalize_det1(A, B, tau, a, b, backend_name)
    if conjugate_A_true:
        # The estimators parametrise Sigma = A^T (x) B, so the A they return is
        # the conjugate of the A that generated the data. See the CONVENTION
        # note in sar.estimation_kronecker._kronecker_quadratic_forms.
        A_true = A_true.conj()

    if tau.shape[-1] == 1 and tau.ndim > 1:
        tau = tau[..., 0]
    if tau_true.shape[-1] == 1 and tau_true.ndim > 1:
        tau_true = tau_true[..., 0]

    d2_A = m_A.dist(A, A_true) ** 2
    d2_B = m_B.dist(B, B_true) ** 2
    d2_tau = m_tau.dist(tau, tau_true) ** 2
    total = w_A * d2_A + w_B * d2_B + w_tau * d2_tau
    return {"A": d2_A, "B": d2_B, "tau": d2_tau, "total": total}


def icrb_scaled_gaussian(
    n_features: int,
    n_samples: int,
    T: "int | np.ndarray",
) -> dict:
    """Intrinsic Cramer-Rao bound for the UNSTRUCTURED scaled Gaussian model.

    Same counting argument as icrb_kronecker_scaled_gaussian, with the shape
    matrix living in sH++(p) (dimension p^2 - 1) instead of the product
    sH++(a) x sH++(b) (dimension (a^2-1) + (b^2-1)).  The gap between the two
    totals is exactly what the Kronecker structure buys.

    Returns
    -------
    dict with keys "Sigma", "tau", "total".
    """
    T_arr = np.asarray(T, dtype=float)
    p, n = n_features, n_samples
    return {
        "Sigma": (p**2 - 1) / (n * T_arr),
        "tau": n / (p * T_arr),
        "total": ((p**2 - 1) + n) / (p * n * T_arr),
    }


def scaled_gaussian_component_errors(
    Sigma: Array,
    tau: Array,
    Sigma_true: Array,
    tau_true: Array,
    n_features: int,
    n_samples: int,
    backend_name: Union[str, Backend] = "numpy",
    normalize: bool = True,
) -> dict:
    """Squared Riemannian errors for the unstructured scaled Gaussian model.

    Mirrors kronecker_component_errors: affine invariant distance on sH++(p)
    for the shape matrix, log-ratio distance for the textures, and the total
    d^2_M weighted by (1/p, 1/n) as in manifolds.ScaledGaussianFIM.
    """
    be = get_backend_module(backend_name)
    manifold = ScaledGaussianFIM(n_features, n_samples, backend_name=backend_name)
    m_Sigma, m_tau = manifold.manifolds
    w_Sigma, w_tau = manifold.weights

    if normalize:
        c = be.exp(be.real(be.linalg.slogdet(Sigma)[1]) / n_features)
        Sigma = Sigma / c[..., None, None]
        tau = tau * c[..., None] if tau.ndim == Sigma.ndim - 1 else tau * c[..., None, None]

    if tau.shape[-1] == 1 and tau.ndim > 1:
        tau = tau[..., 0]
    if tau_true.shape[-1] == 1 and tau_true.ndim > 1:
        tau_true = tau_true[..., 0]

    d2_S = m_Sigma.dist(Sigma, Sigma_true) ** 2
    d2_tau = m_tau.dist(tau, tau_true) ** 2
    return {
        "Sigma": d2_S,
        "tau": d2_tau,
        "total": w_Sigma * d2_S + w_tau * d2_tau,
    }
