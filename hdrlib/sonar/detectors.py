"""Sonar two-array detectors: known-M and adaptive variants.

All known-M detectors accept an optional ``M_override`` kwarg in ``compute``;
pass a batched covariance estimate (..., 2m, 2m) to run adaptively without
the overhead of instantiating a new object per trial.  The
:class:`AdaptiveSonarDetector` wrapper handles this pattern automatically.

Detector glossary
-----------------
M-NMF-G   : two-array MSG GLRT (proposed, known M)
M-NMF-R   : two-array MSG Rao test (proposed, known M)
M-NMF-I   : uncorrelated-array GLRT (benchmark, known M per array)
NMF-i     : single-array NMF, i ∈ {1,2} (benchmark, known M_ii)
MIMO-MF   : optimal Gaussian detector (benchmark, known C=M)

Adaptive wrappers are built by passing any known-M detector and an
estimator to :class:`AdaptiveSonarDetector`.
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
)
from ..core.detection import Detector
from ..core.estimation import Estimator


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _quad(x1: Array, A: Array, x2: Array, be) -> Array:
    """Batched bilinear form x1^H A x2.

    A may be fixed (m,m) or batched (...,m,m); x1, x2 are (...,m).
    """
    if A.ndim == 2:
        return be.einsum("...i,ij,...j->...", x1.conj(), A, x2)
    return be.einsum("...i,...ij,...j->...", x1.conj(), A, x2)


def _scale_product(x1: Array, x2: Array, iA11, iA12, iA22, m: int, be) -> Array:
    """Return ŝ_{1_0} · ŝ_{2_0} = sqrt(a1·a2) + a12.

    Used for both H0 (iA = M⁻¹ blocks) and H1 (iA = R⁻¹ = M⁻¹-D⁻¹ blocks).
    """
    a1  = be.real(_quad(x1, iA11, x1, be)) / m   # (...,)
    a12 = be.real(_quad(x1, iA12, x2, be)) / m
    a2  = be.real(_quad(x2, iA22, x2, be)) / m
    return be.sqrt(be.abs(a1) * be.abs(a2)) + a12  # (...,)


def _h0_sigmas(x1: Array, x2: Array, iM11, iM12, iM22, m: int, be):
    """Individual H0 scale estimates ŝ_1, ŝ_2 (eq. in Appendix A).

    Returns (s1, s2) each of shape (...,).
    """
    a1  = be.real(_quad(x1, iM11, x1, be)) / m
    a12 = be.real(_quad(x1, iM12, x2, be)) / m
    a2  = be.real(_quad(x2, iM22, x2, be)) / m
    eps = 1e-30
    s1_sq = be.abs(a1 + be.sqrt(a1 / (a2 + eps)) * a12)
    s2_sq = be.abs(a2 + be.sqrt(a2 / (a1 + eps)) * a12)
    return be.sqrt(s1_sq), be.sqrt(s2_sq)


def _compute_R_inv(M_inv: Array, P: Array, be) -> Array:
    """R⁻¹ = M⁻¹ - D⁻¹ where D⁻¹ = M⁻¹P(P^H M⁻¹P)⁻¹P^H M⁻¹.

    Works for both fixed M_inv (2m,2m) and batched (...,2m,2m).
    """
    Mi_P  = M_inv @ P                                           # (...,2m,2)
    G     = be.swapaxes(P, -1, -2).conj() @ Mi_P               # (...,2,2) or (2,2)
    G_inv = be.linalg.inv(G)
    D_inv = Mi_P @ G_inv @ be.swapaxes(Mi_P, -1, -2).conj()    # (...,2m,2m)
    return M_inv - D_inv


def _mimo_mf_stat(x: Array, M_inv: Array, P: Array, be) -> Array:
    """x^H M⁻¹P (P^H M⁻¹P)⁻¹ P^H M⁻¹x  (MIMO matched-filter statistic)."""
    Mi_P  = M_inv @ P                                       # (...,2m,2) or (2m,2)
    G     = be.swapaxes(P, -1, -2).conj() @ Mi_P           # (...,2,2) or (2,2)
    G_inv = be.linalg.inv(G)
    # v = P^H M⁻¹ x = Mi_P^H x
    if Mi_P.ndim == 2:
        v = be.einsum("ij,...i->...j", Mi_P.conj(), x)     # (...,2)
    else:
        v = be.einsum("...ij,...i->...j", Mi_P.conj(), x)  # (...,2)
    return be.real(_quad(v, G_inv, v, be))                  # (...,)


# ---------------------------------------------------------------------------
# M-NMF-G: two-array MSG GLRT
# ---------------------------------------------------------------------------

class MNMFGlrt(Detector):
    """Two-array MSG GLRT (M-NMF-G) for known covariance M.

    L_G = (ŝ_{1_0}·ŝ_{2_0}) / (ŝ_{1_1}·ŝ_{2_1})

    where H0/H1 scale products use M⁻¹ and R⁻¹ = M⁻¹ - D⁻¹ respectively.

    Parameters
    ----------
    m : int
        Per-array sensor count.
    M : array-like (2m, 2m)
        Known covariance.
    P : array-like (2m, 2)
        Steering matrix P = blkdiag(p1, p2).
    backend_name : str or Backend
    """

    def __init__(
        self,
        m: int,
        M: Array,
        P: Array,
        backend_name: Union[str, Backend] = "numpy",
    ) -> None:
        self.m = m
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        be = self.be

        M = get_data_on_device(np.asarray(M, dtype=np.complex128), backend_name)
        P = get_data_on_device(np.asarray(P, dtype=np.complex128), backend_name)
        self.P = P
        self.M_inv = be.linalg.inv(M)
        self.R_inv = _compute_R_inv(self.M_inv, P, be)

    def _stat(self, x: Array, M_inv: Array, R_inv: Array) -> Array:
        be, m = self.be, self.m
        x1, x2 = x[..., :m], x[..., m:]
        s0 = _scale_product(x1, x2, M_inv[..., :m, :m], M_inv[..., :m, m:], M_inv[..., m:, m:], m, be)
        s1 = _scale_product(x1, x2, R_inv[..., :m, :m], R_inv[..., :m, m:], R_inv[..., m:, m:], m, be)
        return s0 / (s1 + 1e-60)

    def compute(self, X: Array, *args, M_override=None, **kwargs) -> Array:
        """Compute M-NMF-G statistic.

        Parameters
        ----------
        X : Array (..., 2m)
        M_override : Array (..., 2m, 2m), optional
            If given, use as M̂ instead of the precomputed M.

        Returns
        -------
        Array (...,)
        """
        be = self.be
        x = get_data_on_device(X, self.backend_name)
        if M_override is None:
            return self._stat(x, self.M_inv, self.R_inv)
        M_inv = be.linalg.inv(M_override)
        R_inv = _compute_R_inv(M_inv, self.P, be)
        return self._stat(x, M_inv, R_inv)


# ---------------------------------------------------------------------------
# M-NMF-R: two-array MSG Rao test
# ---------------------------------------------------------------------------

class MNMFRao(Detector):
    """Two-array MSG Rao test (M-NMF-R) for known covariance M.

    L_R = 2 x^H C̃₀⁻¹ P (P^H C̃₀⁻¹ P)⁻¹ P^H C̃₀⁻¹ x
        = 2 u^H D⁻¹ u,   u = Σ̂₀⁻¹ x = [x1/ŝ₁ ; x2/ŝ₂]

    where D⁻¹ = M⁻¹P(P^H M⁻¹P)⁻¹P^H M⁻¹.

    Parameters
    ----------
    m : int
    M : array-like (2m, 2m)
    P : array-like (2m, 2)
    backend_name : str or Backend
    """

    def __init__(
        self,
        m: int,
        M: Array,
        P: Array,
        backend_name: Union[str, Backend] = "numpy",
    ) -> None:
        self.m = m
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        be = self.be

        M = get_data_on_device(np.asarray(M, dtype=np.complex128), backend_name)
        P = get_data_on_device(np.asarray(P, dtype=np.complex128), backend_name)
        self.P = P
        self.M_inv = be.linalg.inv(M)

    def _stat(self, x: Array, M_inv: Array) -> Array:
        be, m = self.be, self.m
        x1, x2 = x[..., :m], x[..., m:]

        # H0 scale estimates
        s1, s2 = _h0_sigmas(x1, x2, M_inv[..., :m, :m], M_inv[..., :m, m:],
                             M_inv[..., m:, m:], m, be)

        # u = [x1/s1; x2/s2]
        u = concatenate(self.backend_name,
                        [x1 / (s1[..., None] + 1e-60),
                         x2 / (s2[..., None] + 1e-60)], axis=-1)  # (..., 2m)

        return 2.0 * _mimo_mf_stat(u, M_inv, self.P, be)

    def compute(self, X: Array, *args, M_override=None, **kwargs) -> Array:
        x = get_data_on_device(X, self.backend_name)
        if M_override is None:
            return self._stat(x, self.M_inv)
        return self._stat(x, self.be.linalg.inv(M_override))


# ---------------------------------------------------------------------------
# M-NMF-I: uncorrelated-arrays GLRT
# ---------------------------------------------------------------------------

class MNMFIndependent(Detector):
    """GLRT assuming independent arrays (M-NMF-I / MIMO ANMF).

    L_G = ∏_i (1 - NMF_i)^{-m}
    Returned as the log-statistic: -m Σ_i log(1 - NMF_i).

    Parameters
    ----------
    m : int
    M : array-like (2m, 2m)  — only diagonal blocks M_11, M_22 are used
    P : array-like (2m, 2)
    backend_name : str or Backend
    """

    def __init__(
        self,
        m: int,
        M: Array,
        P: Array,
        backend_name: Union[str, Backend] = "numpy",
    ) -> None:
        self.m = m
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        be = self.be

        M = get_data_on_device(np.asarray(M, dtype=np.complex128), backend_name)
        P = get_data_on_device(np.asarray(P, dtype=np.complex128), backend_name)
        # Only diagonal blocks needed
        M11 = M[:m, :m]
        M22 = M[m:, m:]
        self.iM11 = be.linalg.inv(M11)   # (m, m)
        self.iM22 = be.linalg.inv(M22)
        self.p1 = P[:m, 0]               # (m,)
        self.p2 = P[m:, 1]               # (m,)
        # Precompute q_i = p_i^H M_{ii}^{-1} p_i  (real scalar)
        self.q1 = float(be.real(_quad(self.p1, self.iM11, self.p1, be)))
        self.q2 = float(be.real(_quad(self.p2, self.iM22, self.p2, be)))

    def _stat(self, x: Array, iM11, iM22) -> Array:
        be, m = self.be, self.m
        x1, x2 = x[..., :m], x[..., m:]

        a1 = be.real(_quad(x1, iM11, x1, be)) / m   # (...,)
        a2 = be.real(_quad(x2, iM22, x2, be)) / m

        # p_i^H M_{ii}^{-1} x_i  (complex scalar per batch)
        if iM11.ndim == 2:
            ph1x1 = be.einsum("i,...i->...", (iM11 @ self.p1).conj(), x1)
            ph2x2 = be.einsum("i,...i->...", (iM22 @ self.p2).conj(), x2)
            q1, q2 = self.q1, self.q2
        else:
            # Batched M
            v1 = iM11 @ self.p1   # (..., m)  — matmul broadcasts p1 (m,)
            v2 = iM22 @ self.p2
            ph1x1 = be.einsum("...i,...i->...", v1.conj(), x1)
            ph2x2 = be.einsum("...i,...i->...", v2.conj(), x2)
            q1 = be.real(be.einsum("...i,...i->...", v1.conj(), self.p1 + 0 * v1))
            q2 = be.real(be.einsum("...i,...i->...", v2.conj(), self.p2 + 0 * v2))

        nmf1 = be.abs(ph1x1) ** 2 / (q1 * m * a1 + 1e-60)   # (...,)
        nmf2 = be.abs(ph2x2) ** 2 / (q2 * m * a2 + 1e-60)

        # Clip to [0, 1) to keep log well-defined
        nmf1 = be.abs(nmf1)
        nmf2 = be.abs(nmf2)
        return -m * (be.log(1.0 - nmf1 + 1e-15) + be.log(1.0 - nmf2 + 1e-15))

    def compute(self, X: Array, *args, M_override=None, **kwargs) -> Array:
        x = get_data_on_device(X, self.backend_name)
        be = self.be
        if M_override is None:
            return self._stat(x, self.iM11, self.iM22)
        iM = be.linalg.inv(M_override)
        return self._stat(x, iM[..., :self.m, :self.m], iM[..., self.m:, self.m:])


# ---------------------------------------------------------------------------
# NMFSingleArray: single-array NMF  (GLRT for one array in PH env.)
# ---------------------------------------------------------------------------

class NMFSingleArray(Detector):
    """Single-array Normalised Matched Filter for array *array_idx* ∈ {0,1}.

    NMF_i = |p_i^H M_{ii}^{-1} x_i|² / (q_i · x_i^H M_{ii}^{-1} x_i)
    where q_i = p_i^H M_{ii}^{-1} p_i.

    Parameters
    ----------
    m : int
    M : array-like (2m, 2m)
    P : array-like (2m, 2)
    array_idx : int  — 0 for array 1, 1 for array 2
    backend_name : str or Backend
    """

    def __init__(
        self,
        m: int,
        M: Array,
        P: Array,
        array_idx: int = 0,
        backend_name: Union[str, Backend] = "numpy",
    ) -> None:
        if array_idx not in (0, 1):
            raise ValueError("array_idx must be 0 (array 1) or 1 (array 2)")
        self.array_idx = array_idx
        self.m = m
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        be = self.be

        M = get_data_on_device(np.asarray(M, dtype=np.complex128), backend_name)
        P = get_data_on_device(np.asarray(P, dtype=np.complex128), backend_name)

        sl = slice(0, m) if array_idx == 0 else slice(m, 2 * m)
        Mii = M[sl, sl]
        self.iMii = be.linalg.inv(Mii)   # (m, m)
        self.pi   = P[sl, array_idx]      # (m,)
        self.q    = float(be.real(_quad(self.pi, self.iMii, self.pi, be)))
        self._sl  = sl

    def _stat(self, xi: Array, iMii) -> Array:
        be = self.be
        # p_i^H M_{ii}^{-1} x_i
        if iMii.ndim == 2:
            phx = be.einsum("i,...i->...", (iMii @ self.pi).conj(), xi)
            q   = self.q
        else:
            v   = iMii @ self.pi   # (..., m)
            phx = be.einsum("...i,...i->...", v.conj(), xi)
            q   = be.real(be.einsum("...i,...i->...", v.conj(), self.pi + 0 * v))
        denom = q * be.real(_quad(xi, iMii, xi, be)) + 1e-60
        return be.abs(phx) ** 2 / denom

    def compute(self, X: Array, *args, M_override=None, **kwargs) -> Array:
        x = get_data_on_device(X, self.backend_name)
        xi = x[..., self._sl]
        be = self.be
        if M_override is None:
            return self._stat(xi, self.iMii)
        iMii = be.linalg.inv(M_override[..., self._sl, self._sl])
        return self._stat(xi, iMii)


# ---------------------------------------------------------------------------
# MimoMatchedFilter: optimal Gaussian detector with known C = M
# ---------------------------------------------------------------------------

class MimoMatchedFilter(Detector):
    """MIMO Matched Filter / R-MIMO OGD (Gaussian, known C).

    L = x^H C⁻¹P (P^H C⁻¹P)⁻¹ P^H C⁻¹x

    Note: this detector is not scale-invariant; the adaptive 2TYL version is
    not meaningful (see paper Section 5.2).  Adaptive SCM version is valid in
    Gaussian environments.

    Parameters
    ----------
    m : int
    M : array-like (2m, 2m)  — plays the role of C
    P : array-like (2m, 2)
    backend_name : str or Backend
    """

    def __init__(
        self,
        m: int,
        M: Array,
        P: Array,
        backend_name: Union[str, Backend] = "numpy",
    ) -> None:
        self.m = m
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        be = self.be

        M = get_data_on_device(np.asarray(M, dtype=np.complex128), backend_name)
        P = get_data_on_device(np.asarray(P, dtype=np.complex128), backend_name)
        self.P = P
        self.M_inv = be.linalg.inv(M)

    def compute(self, X: Array, *args, M_override=None, **kwargs) -> Array:
        x   = get_data_on_device(X, self.backend_name)
        be  = self.be
        M_inv = self.M_inv if M_override is None else be.linalg.inv(M_override)
        return _mimo_mf_stat(x, M_inv, self.P, be)


# ---------------------------------------------------------------------------
# AdaptiveSonarDetector: wraps any known-M detector + a covariance estimator
# ---------------------------------------------------------------------------

class AdaptiveSonarDetector(Detector):
    """Adaptive wrapper: estimate M from secondary data, then run detector.

    Usage
    -----
    >>> from hdrlib.core.estimation import SCMEstimator
    >>> base = MNMFGlrt(m, M_nominal, P)
    >>> det  = AdaptiveSonarDetector(base, SCMEstimator(backend_name="numpy"))
    >>> stat = det.compute(x_primary, X_secondary=x_secondary)

    Parameters
    ----------
    base_detector : Detector
        Any sonar known-M detector that accepts ``M_override`` in ``compute``.
    cov_estimator : Estimator
        Estimator with ``compute(X_secondary) → M_hat``.
        X_secondary shape: (..., K, 2m); output: (..., 2m, 2m).
    """

    def __init__(self, base_detector: Detector, cov_estimator: Estimator) -> None:
        self.base      = base_detector
        self.estimator = cov_estimator
        self.backend_name = base_detector.backend_name

    def compute(self, X: Array, *args, X_secondary: Array = None, **kwargs) -> Array:
        """
        Parameters
        ----------
        X : Array (..., 2m)
            Primary data.
        X_secondary : Array (..., K, 2m)
            Secondary (signal-free) data for covariance estimation.
        """
        if X_secondary is None:
            raise ValueError("X_secondary is required for adaptive detection.")
        M_hat = self.estimator.compute(
            get_data_on_device(X_secondary, self.backend_name)
        )
        return self.base.compute(X, M_override=M_hat)
