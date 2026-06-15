# Tests for M-estimator classes
# Author: Ammar Mian
# Date: 22/10/2025

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

from hdrlib.core.estimation import (
    TylerEstimator,
    StudentTEstimator,
    HuberEstimator,
    fixed_point_m_estimation_centered,
    _tyler_m_estimator_function,
    _student_t_m_estimator_function,
    _huber_m_estimator_function,
)
from hdrlib.sar.estimation_online import OnlineScaledGaussianEstimator
from hdrlib.core.backend import get_data_on_device


@pytest.fixture
def sample_data_numpy():
    """Generate centered sample data for numpy."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))  # 50 samples, 5 features
    X = X - X.mean(axis=0)  # Center the data
    return X


@pytest.fixture
def sample_data_torch():
    """Generate centered sample data for torch."""
    torch.manual_seed(42)
    X = torch.randn((50, 5), dtype=torch.float64)  # 50 samples, 5 features
    X = X - X.mean(dim=0)  # Center the data
    return X


@pytest.fixture
def batched_sample_data_numpy():
    """Generate centered batched sample data for numpy."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2, 3, 50, 5))  # 2x3 batches, 50 samples, 5 features
    X = X - X.mean(axis=-2, keepdims=True)  # Center along samples axis
    return X


@pytest.fixture
def batched_sample_data_torch():
    """Generate centered batched sample data for torch."""
    torch.manual_seed(42)
    X = torch.randn((2, 3, 50, 5), dtype=torch.float64)
    X = X - X.mean(dim=-2, keepdim=True)  # Center along samples axis
    return X


class TestTylerEstimator:
    """Tests for TylerEstimator class."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_initialization(self, backend_name):
        """Test TylerEstimator initialization."""
        estimator = TylerEstimator(
            normalization="trace",
            backend_name=backend_name,
            tol=1e-4,
            iter_max=50,
        )
        assert estimator.backend_name == backend_name
        assert estimator.normalization == "trace"
        assert estimator.tol == 1e-4
        assert estimator.iter_max == 50

    def test_compute_numpy(self, sample_data_numpy):
        """Test TylerEstimator.compute() with numpy data."""
        estimator = TylerEstimator(normalization="trace", backend_name="numpy", iter_max=20)
        cov = estimator.compute(sample_data_numpy)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check symmetry
        assert np.allclose(cov, cov.T)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

        # Check trace normalization
        assert np.isclose(np.trace(cov), 5.0, atol=0.1)

    def test_compute_torch(self, sample_data_torch):
        """Test TylerEstimator.compute() with torch data."""
        estimator = TylerEstimator(
            normalization="trace", backend_name="torch-cpu", iter_max=20
        )
        cov = estimator.compute(sample_data_torch)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check symmetry
        assert torch.allclose(cov, cov.T, atol=1e-6)

        # Check positive definiteness
        eigvals = torch.linalg.eigvalsh(cov)
        assert torch.all(eigvals > 0)

        # Check trace normalization
        assert torch.isclose(torch.trace(cov), torch.tensor(5.0, dtype=cov.dtype), atol=0.1)

    @pytest.mark.parametrize("normalization", ["trace", "det", "diag"])
    def test_different_normalizations(self, sample_data_numpy, normalization):
        """Test TylerEstimator with different normalization methods."""
        estimator = TylerEstimator(normalization=normalization, backend_name="numpy", iter_max=20)
        cov = estimator.compute(sample_data_numpy)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check specific normalization
        if normalization == "trace":
            assert np.isclose(np.trace(cov), 5.0, atol=0.1)
        elif normalization == "det":
            assert np.isclose(np.linalg.det(cov), 1.0, atol=0.1)
        elif normalization == "diag":
            assert np.isclose(cov[0, 0], 1.0, atol=0.1)

    def test_batched_computation(self, batched_sample_data_numpy):
        """Test TylerEstimator with batched data."""
        estimator = TylerEstimator(normalization="trace", backend_name="numpy", iter_max=20)
        cov = estimator.compute(batched_sample_data_numpy)

        # Check output shape
        assert cov.shape == (2, 3, 5, 5)

        # Check all matrices are positive definite
        for i in range(2):
            for j in range(3):
                eigvals = np.linalg.eigvalsh(cov[i, j])
                assert np.all(eigvals > 0)


class TestStudentTEstimator:
    """Tests for StudentTEstimator class."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_initialization(self, backend_name):
        """Test StudentTEstimator initialization."""
        estimator = StudentTEstimator(
            df=3,
            backend_name=backend_name,
            tol=1e-4,
            iter_max=50,
        )
        assert estimator.backend_name == backend_name
        assert estimator.df == 3
        assert estimator.tol == 1e-4
        assert estimator.iter_max == 50

    def test_compute_numpy(self, sample_data_numpy):
        """Test StudentTEstimator.compute() with numpy data."""
        estimator = StudentTEstimator(df=3, backend_name="numpy", iter_max=20)
        cov = estimator.compute(sample_data_numpy)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check symmetry
        assert np.allclose(cov, cov.T)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_compute_torch(self, sample_data_torch):
        """Test StudentTEstimator.compute() with torch data."""
        estimator = StudentTEstimator(df=3, backend_name="torch-cpu", iter_max=20)
        cov = estimator.compute(sample_data_torch)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check symmetry
        assert torch.allclose(cov, cov.T, atol=1e-6)

        # Check positive definiteness
        eigvals = torch.linalg.eigvalsh(cov)
        assert torch.all(eigvals > 0)

    @pytest.mark.parametrize("df", [3, 10, 300])
    def test_different_df(self, sample_data_numpy, df):
        """Test StudentTEstimator with different degrees of freedom."""
        estimator = StudentTEstimator(df=df, backend_name="numpy", iter_max=20)
        cov = estimator.compute(sample_data_numpy)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_batched_computation(self, batched_sample_data_numpy):
        """Test StudentTEstimator with batched data."""
        estimator = StudentTEstimator(df=3, backend_name="numpy", iter_max=20)
        cov = estimator.compute(batched_sample_data_numpy)

        # Check output shape
        assert cov.shape == (2, 3, 5, 5)

        # Check all matrices are positive definite
        for i in range(2):
            for j in range(3):
                eigvals = np.linalg.eigvalsh(cov[i, j])
                assert np.all(eigvals > 0)

    def test_optional_normalization(self, sample_data_numpy):
        """Test StudentTEstimator with optional normalization."""
        # Without normalization
        estimator1 = StudentTEstimator(
            df=3, normalization=None, backend_name="numpy", iter_max=20
        )
        cov1 = estimator1.compute(sample_data_numpy)

        # With trace normalization
        estimator2 = StudentTEstimator(
            df=3, normalization="trace", backend_name="numpy", iter_max=20
        )
        cov2 = estimator2.compute(sample_data_numpy)

        # Both should be valid covariance matrices
        assert np.all(np.linalg.eigvalsh(cov1) > 0)
        assert np.all(np.linalg.eigvalsh(cov2) > 0)

        # With normalization, trace should be n_features
        assert np.isclose(np.trace(cov2), 5.0, atol=0.1)


class TestHuberEstimator:
    """Tests for HuberEstimator class."""

    @pytest.mark.parametrize("backend_name", ["numpy", "torch-cpu"])
    def test_initialization(self, backend_name):
        """Test HuberEstimator initialization."""
        estimator = HuberEstimator(
            lbda=2.0,
            beta=1.0,
            backend_name=backend_name,
            tol=1e-4,
            iter_max=50,
        )
        assert estimator.backend_name == backend_name
        assert estimator.lbda == 2.0
        assert estimator.beta == 1.0
        assert estimator.tol == 1e-4
        assert estimator.iter_max == 50

    def test_compute_numpy(self, sample_data_numpy):
        """Test HuberEstimator.compute() with numpy data."""
        estimator = HuberEstimator(lbda=2.0, beta=1.0, backend_name="numpy", iter_max=20)
        cov = estimator.compute(sample_data_numpy)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check symmetry
        assert np.allclose(cov, cov.T)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_compute_torch(self, sample_data_torch):
        """Test HuberEstimator.compute() with torch data."""
        estimator = HuberEstimator(lbda=2.0, beta=1.0, backend_name="torch-cpu", iter_max=20)
        cov = estimator.compute(sample_data_torch)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check symmetry
        assert torch.allclose(cov, cov.T, atol=1e-6)

        # Check positive definiteness
        eigvals = torch.linalg.eigvalsh(cov)
        assert torch.all(eigvals > 0)

    @pytest.mark.parametrize(
        "lbda,beta", [(2.0, 1.0), (1.5, 0.5), (float("inf"), 1.0)]
    )
    def test_different_parameters(self, sample_data_numpy, lbda, beta):
        """Test HuberEstimator with different lbda and beta parameters."""
        estimator = HuberEstimator(lbda=lbda, beta=beta, backend_name="numpy", iter_max=20)
        cov = estimator.compute(sample_data_numpy)

        # Check output shape
        assert cov.shape == (5, 5)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_batched_computation(self, batched_sample_data_numpy):
        """Test HuberEstimator with batched data."""
        estimator = HuberEstimator(lbda=2.0, beta=1.0, backend_name="numpy", iter_max=20)
        cov = estimator.compute(batched_sample_data_numpy)

        # Check output shape
        assert cov.shape == (2, 3, 5, 5)

        # Check all matrices are positive definite
        for i in range(2):
            for j in range(3):
                eigvals = np.linalg.eigvalsh(cov[i, j])
                assert np.all(eigvals > 0)


class TestEstimatorConsistency:
    """Tests for consistency between different estimators."""

    def test_tyler_vs_functional_api(self, sample_data_numpy):
        """Test that TylerEstimator gives same results as functional API."""
        # Class-based API
        estimator = TylerEstimator(
            normalization="trace", backend_name="numpy", iter_max=20, tol=1e-5
        )
        cov_class = estimator.compute(sample_data_numpy)

        # Functional API
        cov_func = fixed_point_m_estimation_centered(
            X=sample_data_numpy,
            m_estimator_function=_tyler_m_estimator_function,
            normalization="trace",
            backend_name="numpy",
            iter_max=20,
            tol=1e-5,
            n_features=5,
        )

        # Results should be very close
        assert np.allclose(cov_class, cov_func, atol=1e-6)

    def test_student_t_vs_functional_api(self, sample_data_numpy):
        """Test that StudentTEstimator gives same results as functional API."""
        # Class-based API
        estimator = StudentTEstimator(
            df=3, backend_name="numpy", iter_max=20, tol=1e-5
        )
        cov_class = estimator.compute(sample_data_numpy)

        # Functional API
        cov_func = fixed_point_m_estimation_centered(
            X=sample_data_numpy,
            m_estimator_function=_student_t_m_estimator_function,
            backend_name="numpy",
            iter_max=20,
            tol=1e-5,
            df=3,
            n_features=5,
        )

        # Results should be very close
        assert np.allclose(cov_class, cov_func, atol=1e-6)

    def test_huber_vs_functional_api(self, sample_data_numpy):
        """Test that HuberEstimator gives same results as functional API."""
        # Class-based API
        estimator = HuberEstimator(
            lbda=2.0, beta=1.0, backend_name="numpy", iter_max=20, tol=1e-5
        )
        cov_class = estimator.compute(sample_data_numpy)

        # Functional API
        cov_func = fixed_point_m_estimation_centered(
            X=sample_data_numpy,
            m_estimator_function=_huber_m_estimator_function,
            backend_name="numpy",
            iter_max=20,
            tol=1e-5,
            lbda=2.0,
            beta=1.0,
        )

        # Results should be very close
        assert np.allclose(cov_class, cov_func, atol=1e-6)


class TestEstimatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample_warning(self):
        """Test that estimator handles very small sample size."""
        # Very small sample (shouldn't crash, but may not converge well)
        X = np.random.randn(5, 3)  # Only 5 samples, 3 features
        X = X - X.mean(axis=0)

        estimator = TylerEstimator(normalization="trace", backend_name="numpy", iter_max=10)
        cov = estimator.compute(X)

        # Should still produce a valid shape
        assert cov.shape == (3, 3)

    def test_high_dimensional_data(self):
        """Test estimator with higher dimensional data."""
        X = np.random.randn(200, 50)  # 200 samples, 50 features
        X = X - X.mean(axis=0)

        estimator = TylerEstimator(normalization="trace", backend_name="numpy", iter_max=20)
        cov = estimator.compute(X)

        # Check shape
        assert cov.shape == (50, 50)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)


# ============================================================================
# Tests for Batched Natural Gradient Estimator
# ============================================================================
class TestNaturalGradientBatched:
    """Tests for scaled Gaussian natural gradient with batch dimensions."""

    def test_natural_gradient_scalar_shape(self):
        """Natural gradient without batch dims should return (p, p) Sigma, (n, 1) tau."""
        from hdrlib.core.estimation import natural_gradient_scaled_gaussian

        np.random.seed(42)
        X = np.random.randn(20, 6).astype(complex) / np.sqrt(2)

        Sigma, tau = natural_gradient_scaled_gaussian(X, iter_max=5)

        assert Sigma.shape == (6, 6), f"Expected (6, 6), got {Sigma.shape}"
        assert tau.shape == (20, 1), f"Expected (20, 1), got {tau.shape}"

    def test_natural_gradient_batched_shape(self):
        """Natural gradient with batch dims should return correct batch shape."""
        from hdrlib.core.estimation import natural_gradient_scaled_gaussian

        np.random.seed(42)
        X = np.random.randn(3, 2, 20, 6).astype(complex) / np.sqrt(2)

        Sigma, tau = natural_gradient_scaled_gaussian(X, iter_max=5)

        assert Sigma.shape == (3, 2, 6, 6), f"Expected (3, 2, 6, 6), got {Sigma.shape}"
        assert tau.shape == (3, 2, 20, 1), f"Expected (3, 2, 20, 1), got {tau.shape}"

    def test_natural_gradient_consistency(self):
        """Batched result should match serial per-element result."""
        from hdrlib.core.estimation import natural_gradient_scaled_gaussian

        np.random.seed(42)
        # Single batch element
        X_single = np.random.randn(1, 20, 6).astype(complex) / np.sqrt(2)
        Sigma_batch, tau_batch = natural_gradient_scaled_gaussian(X_single, iter_max=10)

        # The same data processed as a scalar
        X_scalar = X_single[0]
        Sigma_scalar, tau_scalar = natural_gradient_scaled_gaussian(X_scalar, iter_max=10)

        # Results should match
        sigma_error = np.linalg.norm(Sigma_batch[0] - Sigma_scalar) / np.linalg.norm(Sigma_scalar)
        tau_error = np.linalg.norm(tau_batch[0] - tau_scalar) / np.linalg.norm(tau_scalar)

        assert sigma_error < 1e-12, f"Sigma error: {sigma_error}"
        assert tau_error < 1e-12, f"Tau error: {tau_error}"


# ============================================================================
# Tests for Batched Gradient Functions
# ============================================================================
class TestRgradBatched:
    """Tests for _rgrad_scaled_gaussian batched operations."""

    def test_rgrad_shape_scalar(self):
        """Gradient at scalar point should have correct shape."""
        from hdrlib.core.estimation import _rgrad_scaled_gaussian
        from hdrlib.core.manifolds import ScaledGaussianFIM
        from hdrlib.core.backend import get_backend_module

        be = get_backend_module("numpy")
        manifold = ScaledGaussianFIM(6, 20, backend_name="numpy")

        X = np.random.randn(20, 6).astype(complex) / np.sqrt(2)
        Sigma = np.eye(6, dtype=complex)
        tau = np.ones((20, 1), dtype=float)

        r_Sigma, r_tau = _rgrad_scaled_gaussian(X, Sigma, tau, manifold, be)

        assert r_Sigma.shape == (6, 6), f"Expected (6, 6), got {r_Sigma.shape}"
        assert r_tau.shape == (20, 1), f"Expected (20, 1), got {r_tau.shape}"

    def test_rgrad_shape_batched(self):
        """Gradient at batched point should have correct shape."""
        from hdrlib.core.estimation import _rgrad_scaled_gaussian
        from hdrlib.core.manifolds import ScaledGaussianFIM
        from hdrlib.core.backend import get_backend_module

        be = get_backend_module("numpy")
        manifold = ScaledGaussianFIM(6, 20, backend_name="numpy")

        X = np.random.randn(3, 2, 20, 6).astype(complex) / np.sqrt(2)
        Sigma = np.tile(np.eye(6, dtype=complex)[None, None, :, :], (3, 2, 1, 1))
        tau = np.ones((3, 2, 20, 1), dtype=float)

        r_Sigma, r_tau = _rgrad_scaled_gaussian(X, Sigma, tau, manifold, be)

        assert r_Sigma.shape == (3, 2, 6, 6), f"Expected (3, 2, 6, 6), got {r_Sigma.shape}"
        assert r_tau.shape == (3, 2, 20, 1), f"Expected (3, 2, 20, 1), got {r_tau.shape}"

    def test_neg_loglik_shape(self):
        """Negative log-likelihood should return (...,) for batched input."""
        from hdrlib.core.estimation import _neg_log_likelihood_scaled_gaussian
        from hdrlib.core.backend import get_backend_module

        be = get_backend_module("numpy")

        # Scalar case
        X_scalar = np.random.randn(20, 6).astype(complex) / np.sqrt(2)
        Sigma_scalar = np.eye(6, dtype=complex)
        tau_scalar = np.ones((20, 1), dtype=float)

        f = _neg_log_likelihood_scaled_gaussian(X_scalar, Sigma_scalar, tau_scalar, be)
        assert np.isscalar(f) or f.shape == (), f"Expected scalar, got {f.shape}"

        # Batched case
        X_batched = np.random.randn(3, 2, 20, 6).astype(complex) / np.sqrt(2)
        Sigma_batched = np.tile(np.eye(6, dtype=complex)[None, None, :, :], (3, 2, 1, 1))
        tau_batched = np.ones((3, 2, 20, 1), dtype=float)

        f_batch = _neg_log_likelihood_scaled_gaussian(X_batched, Sigma_batched, tau_batched, be)
        assert f_batch.shape == (3, 2), f"Expected (3, 2), got {f_batch.shape}"


# ============================================================================
# Tests for Online Scaled Gaussian Estimator
# ============================================================================
class TestOnlineScaledGaussianBatched:
    """Tests for online estimator with batched spatial dimensions."""

    def test_online_estimator_scalar(self):
        """Online estimator with scalar (no batch) spatial dims."""
        np.random.seed(42)
        estimator = OnlineScaledGaussianEstimator(
            n_features=6, n_samples=20, backend_name="numpy"
        )

        # Three time steps of data
        for t in range(3):
            X = np.random.randn(20, 6).astype(complex) / np.sqrt(2)
            Sigma, tau = estimator.update(X)

            assert Sigma.shape == (6, 6), f"Step {t}: Expected (6, 6), got {Sigma.shape}"
            assert tau.shape == (20, 1), f"Step {t}: Expected (20, 1), got {tau.shape}"

    def test_online_estimator_batched(self):
        """Online estimator with spatial batch dimensions."""
        np.random.seed(42)
        estimator = OnlineScaledGaussianEstimator(
            n_features=6, n_samples=20, backend_name="numpy"
        )

        # Three time steps of batched data (spatial batch 2x2)
        for t in range(3):
            X = np.random.randn(2, 2, 20, 6).astype(complex) / np.sqrt(2)
            Sigma, tau = estimator.update(X)

            assert Sigma.shape == (2, 2, 6, 6), f"Step {t}: Expected (2, 2, 6, 6), got {Sigma.shape}"
            assert tau.shape == (2, 2, 20, 1), f"Step {t}: Expected (2, 2, 20, 1), got {tau.shape}"

    def test_online_estimator_reset(self):
        """Online estimator reset should prepare for warm-start."""
        np.random.seed(42)
        estimator = OnlineScaledGaussianEstimator(n_features=6, n_samples=20)

        X = np.random.randn(20, 6).astype(complex) / np.sqrt(2)
        estimator.update(X)

        assert hasattr(estimator, 'Sigma') and estimator.Sigma is not None
        assert hasattr(estimator, 'tau') and estimator.tau is not None

        # Reset
        estimator.reset()
        assert estimator._t == 0
        assert estimator.Sigma is None
        assert estimator.tau is None

        # Next update should warm-start
        estimator.update(X)
        assert estimator._t == 1
