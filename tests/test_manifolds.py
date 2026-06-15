"""Tests for Riemannian manifolds, focusing on batch dimension support."""

import numpy as np

from hdrlib.core.backend import get_backend_module
from hdrlib.core.manifolds import (
    HermitianPositiveDefinite,
    SpecialHermitianPositiveDefinite,
    StrictlyPositiveVectors,
    ScaledGaussianFIM,
)


# ============================================================================
# Test HermitianPositiveDefinite (HPD)
# ============================================================================
class TestHPD:
    """Tests for HPD manifold with batch dimension support."""

    def test_exp_log_roundtrip_scalar(self):
        """Exponential-logarithm map should roundtrip for scalar input."""
        be = get_backend_module("numpy")
        manifold = HermitianPositiveDefinite(3, backend_name="numpy")

        x = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        x = 0.5 * (x + x.conj().T)  # hermitianize
        x = x + 4 * np.eye(3)  # make PD

        y = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        y = 0.5 * (y + y.conj().T)
        y = y + 4 * np.eye(3)

        # log from x to y, then exp back
        u = manifold.log(x, y)
        y_recovered = manifold.exp(x, u)

        dist_error = float(manifold.dist(y, y_recovered))
        assert dist_error < 1e-10, f"Roundtrip error {dist_error} exceeds tolerance"

    def test_exp_log_roundtrip_batched(self):
        """Exponential-logarithm should roundtrip for batched input."""
        be = get_backend_module("numpy")
        manifold = HermitianPositiveDefinite(3, backend_name="numpy")

        # Batch of 4 HPD matrices
        batch_shape = (2, 2)
        X = np.random.randn(*batch_shape, 3, 3) + 1j * np.random.randn(*batch_shape, 3, 3)
        X = 0.5 * (X + np.swapaxes(X.conj(), -2, -1))  # hermitianize
        X = X + 4 * np.eye(3)[None, None, :, :]  # make PD

        Y = np.random.randn(*batch_shape, 3, 3) + 1j * np.random.randn(*batch_shape, 3, 3)
        Y = 0.5 * (Y + np.swapaxes(Y.conj(), -2, -1))
        Y = Y + 4 * np.eye(3)[None, None, :, :]

        # log and exp
        U = manifold.log(X, Y)
        Y_recovered = manifold.exp(X, U)

        dists = manifold.dist(Y, Y_recovered)  # shape (2, 2)
        assert dists.shape == batch_shape, f"Expected shape {batch_shape}, got {dists.shape}"
        assert np.all(dists < 1e-10), f"Max roundtrip error {dists.max()} exceeds tolerance"

    def test_inner_product_positive(self):
        """Inner product with u, u should be positive for nonzero u."""
        manifold = HermitianPositiveDefinite(3, backend_name="numpy")

        x = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        x = 0.5 * (x + x.conj().T) + 4 * np.eye(3)

        u = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        u = 0.5 * (u + u.conj().T)  # tangent vector

        inner = float(manifold.inner(x, u, u))
        assert inner > 1e-10, f"Inner product should be positive, got {inner}"

    def test_batched_inner_product_shape(self):
        """Batched inner product should return shape (...)."""
        manifold = HermitianPositiveDefinite(3, backend_name="numpy")

        batch_shape = (2, 3)
        X = np.random.randn(*batch_shape, 3, 3) + 1j * np.random.randn(*batch_shape, 3, 3)
        X = 0.5 * (X + np.swapaxes(X.conj(), -2, -1)) + 4 * np.eye(3)[None, None, :, :]

        U = np.random.randn(*batch_shape, 3, 3) + 1j * np.random.randn(*batch_shape, 3, 3)
        U = 0.5 * (U + np.swapaxes(U.conj(), -2, -1))

        inner = manifold.inner(X, U, U)
        assert inner.shape == batch_shape, f"Expected shape {batch_shape}, got {inner.shape}"


# ============================================================================
# Test SpecialHermitianPositiveDefinite (SHPD)
# ============================================================================
class TestSHPD:
    """Tests for SHPD manifold (det=1)."""

    def test_det_preserved_after_exp(self):
        """det(exp_x(u)) should equal 1 for SHPD."""
        manifold = SpecialHermitianPositiveDefinite(3, backend_name="numpy")

        x = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        x = 0.5 * (x + x.conj().T) + 4 * np.eye(3)
        x = x / (np.linalg.det(x) ** (1 / 3))  # normalize det to 1

        u = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        u = 0.5 * (u + u.conj().T)
        u = u - np.trace(u) / 3 * np.eye(3)  # make traceless

        x_new = manifold.exp(x, u)
        det_new = float(np.linalg.det(x_new))
        assert abs(det_new - 1.0) < 1e-10, f"det should be 1, got {det_new}"


# ============================================================================
# Test StrictlyPositiveVectors
# ============================================================================
class TestStrictlyPositiveVectors:
    """Tests for strictly positive vector manifold."""

    def test_exp_preserves_positivity(self):
        """exp_x(u) should remain strictly positive."""
        manifold = StrictlyPositiveVectors(5, backend_name="numpy")

        x = np.abs(np.random.randn(5)) + 0.1
        u = np.random.randn(5)

        x_new = manifold.exp(x, u)
        assert np.all(x_new > 0), f"exp result should be positive, got {x_new}"

    def test_inner_product_positive_batched(self):
        """Inner product inner(x, u, u) should be positive for batched input."""
        manifold = StrictlyPositiveVectors(5, backend_name="numpy")

        batch_shape = (3, 2)
        x = np.abs(np.random.randn(*batch_shape, 5)) + 0.1
        u = np.random.randn(*batch_shape, 5)

        inner = manifold.inner(x, u, u)
        assert inner.shape == batch_shape, f"Expected shape {batch_shape}, got {inner.shape}"
        assert np.all(inner > 0), f"All inner products should be positive"

    def test_log_exp_roundtrip(self):
        """log and exp should roundtrip."""
        manifold = StrictlyPositiveVectors(5, backend_name="numpy")

        x = np.abs(np.random.randn(5)) + 0.1
        y = np.abs(np.random.randn(5)) + 0.1

        u = manifold.log(x, y)
        y_recovered = manifold.exp(x, u)

        error = np.linalg.norm(y - y_recovered)
        assert error < 1e-10, f"Roundtrip error {error} exceeds tolerance"


# ============================================================================
# Test ScaledGaussianFIM (product manifold)
# ============================================================================
class TestScaledGaussianFIM:
    """Tests for ScaledGaussianFIM product manifold."""

    def test_inner_product_scalar_shape(self):
        """Inner product for single input should return scalar."""
        manifold = ScaledGaussianFIM(3, 5, backend_name="numpy")

        Sigma = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        Sigma = 0.5 * (Sigma + Sigma.conj().T) + 4 * np.eye(3)
        Sigma = Sigma / (np.linalg.det(Sigma) ** (1 / 3))

        tau = np.abs(np.random.randn(5)) + 0.1
        tau_reshaped = tau.reshape(5, 1)  # (..., n, 1) convention

        r_Sigma = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        r_Sigma = 0.5 * (r_Sigma + r_Sigma.conj().T)

        r_tau = np.random.randn(5)

        inner = manifold.inner([Sigma, tau], [r_Sigma, r_tau], [r_Sigma, r_tau])
        assert np.isscalar(inner) or inner.shape == (), f"Expected scalar, got shape {inner.shape}"

    def test_inner_product_batched_shape(self):
        """Inner product for batched input should return shape (...)."""
        manifold = ScaledGaussianFIM(3, 5, backend_name="numpy")

        batch_shape = (2, 2)

        # Batched Sigma: (2, 2, 3, 3)
        Sigma = np.random.randn(*batch_shape, 3, 3) + 1j * np.random.randn(*batch_shape, 3, 3)
        Sigma = 0.5 * (Sigma + np.swapaxes(Sigma.conj(), -2, -1)) + 4 * np.eye(3)[None, None, :, :]

        # Batched tau: (2, 2, 5)
        tau = np.abs(np.random.randn(*batch_shape, 5)) + 0.1

        r_Sigma = np.random.randn(*batch_shape, 3, 3) + 1j * np.random.randn(*batch_shape, 3, 3)
        r_Sigma = 0.5 * (r_Sigma + np.swapaxes(r_Sigma.conj(), -2, -1))

        r_tau = np.random.randn(*batch_shape, 5)

        inner = manifold.inner([Sigma, tau], [r_Sigma, r_tau], [r_Sigma, r_tau])
        assert inner.shape == batch_shape, f"Expected shape {batch_shape}, got {inner.shape}"
