"""Two-array Tyler MLE (2TYL) covariance estimator.

Reference: Section 4 / eq. (cov_Tyler) of the sonar paper.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..core.backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
    concatenate,
    batched_trace,
)
from ..core.estimation import Estimator


def two_array_tyler(
    X: Array,
    m: int,
    tol: float = 1e-6,
    iter_max: int = 500,
    backend_name: Union[str, Backend] = "numpy",
) -> Array:
    """Two-array Tyler MLE fixed-point for MSG covariance estimation.

    Solves:
        M̂ = (1/K) Σ_k T̂_k⁻¹ x_k x_k^H T̂_k⁻¹

    where T̂_k = diag(√τ̂_{1k}, √τ̂_{2k}) ⊗ I_m and the textures are:
        τ̂_{1k} = t1 + √(t1/t2) · t12
        τ̂_{2k} = t2 + √(t2/t1) · t12
        t1 = x_{1k}^H M̂_{11}⁻¹ x_{1k} / m
        t2 = x_{2k}^H M̂_{22}⁻¹ x_{2k} / m
        t12 = Re(x_{1k}^H M̂_{12}⁻¹ x_{2k}) / m

    The estimate is trace-normalised to trace(M̂) = 2m at each step.

    Parameters
    ----------
    X : Array of shape (..., K, 2m)
        Secondary (signal-free) data.
    m : int
        Per-array dimension; total = 2m.
    tol : float
        Convergence threshold on relative Frobenius norm.
    iter_max : int
        Maximum number of iterations.
    backend_name : str or Backend

    Returns
    -------
    Array of shape (..., 2m, 2m)
    """
    be = get_backend_module(backend_name)
    X = get_data_on_device(X, backend_name)

    p = 2 * m
    K = X.shape[-2]
    x1 = X[..., :m]   # (..., K, m)
    x2 = X[..., m:]   # (..., K, m)

    # Initialise M_hat with the same batch shape as X so that diff = M_new - M_hat
    # always has a consistent shape from the very first iteration.
    batch_shape = X.shape[:-2]   # e.g. (n_trials,) or ()
    M_eye = np.eye(p, dtype=np.complex128)
    if batch_shape:
        M_eye_batched = np.broadcast_to(M_eye, (*batch_shape, p, p)).copy()
    else:
        M_eye_batched = M_eye
    M_hat = get_data_on_device(M_eye_batched, backend_name)

    eps = 1e-30  # numerical floor

    for _ in range(iter_max):
        M_inv = be.linalg.inv(M_hat)   # (..., 2m, 2m) or (2m, 2m)
        iM11 = M_inv[..., :m, :m]
        iM12 = M_inv[..., :m, m:]
        iM22 = M_inv[..., m:, m:]

        # Apply M_inv blocks to x1, x2 across K samples.
        # iMij @ xi for each k:  (2m,2m) @ (..., m, K) → (..., m, K)  → (..., K, m)
        vx1  = be.swapaxes(iM11 @ be.swapaxes(x1, -1, -2), -1, -2)   # (..., K, m)
        vx2  = be.swapaxes(iM22 @ be.swapaxes(x2, -1, -2), -1, -2)
        vx12 = be.swapaxes(iM12 @ be.swapaxes(x2, -1, -2), -1, -2)

        t1  = be.real((x1.conj() * vx1).sum(axis=-1)) / m    # (..., K)
        t2  = be.real((x2.conj() * vx2).sum(axis=-1)) / m
        t12 = be.real((x1.conj() * vx12).sum(axis=-1)) / m

        # Texture estimates — t12 can be negative (Re of complex quadratic
        # form), so abs() before sqrt to guarantee positivity.
        tau1 = be.abs(t1 + be.sqrt(t1 / (t2 + eps)) * t12) + eps   # (..., K)
        tau2 = be.abs(t2 + be.sqrt(t2 / (t1 + eps)) * t12) + eps

        # T̂_k⁻¹ x_k = [x1 / √τ1 ; x2 / √τ2]
        x1s = x1 / be.sqrt(tau1[..., None])   # (..., K, m)
        x2s = x2 / be.sqrt(tau2[..., None])
        xs  = concatenate(backend_name, [x1s, x2s], axis=-1)  # (..., K, 2m)

        # M̂_new = (1/K) xs^H xs  (outer product summed over K)
        M_new = be.swapaxes(xs, -1, -2).conj() @ xs / K    # (..., 2m, 2m)

        # Trace-normalise: tr(M̂) = 2m
        tr = be.real(batched_trace(backend_name, M_new))    # (...,) or scalar
        M_new = M_new * (p / tr[..., None, None])

        # Relative Frobenius convergence check
        diff = M_new - M_hat
        batch = M_new.shape[:-2]
        frob_d = be.sqrt(be.sum(be.abs(diff.reshape(*batch, -1)) ** 2, axis=-1))
        frob_M = be.sqrt(be.sum(be.abs(M_hat.reshape(*batch, -1)) ** 2, axis=-1))
        rel = frob_d / (frob_M + eps)

        M_hat = M_new

        if float(be.max(rel)) < tol:
            break

    return M_hat


class TwoArrayTylerEstimator(Estimator):
    """Wrapper around :func:`two_array_tyler` following the Estimator ABC.

    Parameters
    ----------
    m : int
        Per-array sensor count.
    tol : float
        Fixed-point convergence tolerance.
    iter_max : int
        Maximum iterations.
    backend_name : str or Backend
    """

    def __init__(
        self,
        m: int,
        tol: float = 1e-6,
        iter_max: int = 500,
        backend_name: Union[str, Backend] = "numpy",
    ) -> None:
        self.m = m
        self.tol = tol
        self.iter_max = iter_max
        self.backend_name = backend_name

    def compute(self, X: Array) -> Array:
        """Compute M̂_2TYL from secondary data.

        Parameters
        ----------
        X : Array of shape (..., K, 2m)
        Returns
        -------
        Array of shape (..., 2m, 2m)
        """
        X = get_data_on_device(X, self.backend_name)
        return two_array_tyler(X, self.m, self.tol, self.iter_max, self.backend_name)
