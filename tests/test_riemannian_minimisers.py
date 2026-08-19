# Tests for the hand-written minimisers on the cone
# Author: Ammar Mian

import pytest
import numpy as np
import torch

from hdrlib.core.estimation import (
    TylerEstimator,
    tyler_cost,
    tyler_riemannian_gradient,
    minimize_tyler_fixed_point,
    minimize_tyler_riemannian,
    minimize_tyler_euclidean,
    frechet_mean_affine_invariant,
)
from hdrlib.core.backend import to_numpy
from hdrlib.core.manifolds import HermitianPositiveDefinite


@pytest.fixture
def sample_data_numpy():
    """Centered, ill-conditioned sample data for numpy."""
    rng = np.random.default_rng(42)
    scatter = np.diag(np.logspace(-1, 1, 5))
    X = rng.standard_normal((200, 5)) @ np.linalg.cholesky(scatter).T
    return X - X.mean(axis=0)


@pytest.fixture
def sample_data_torch(sample_data_numpy):
    """The same data, as a float64 torch tensor."""
    return torch.as_tensor(sample_data_numpy, dtype=torch.float64)


@pytest.fixture
def covariances_numpy():
    """A small set of well-conditioned covariance matrices."""
    rng = np.random.default_rng(0)
    factors = rng.standard_normal((6, 4, 4))
    return np.stack([factor @ factor.T + 4 * np.eye(4) for factor in factors])


def normalize_determinant(matrix):
    n_features = matrix.shape[-1]
    return matrix / np.linalg.det(matrix) ** (1.0 / n_features)


class TestTylerCost:
    def test_scale_invariance(self, sample_data_numpy):
        """The cost does not see the scale, exactly as the estimator does not."""
        Sigma = np.cov(sample_data_numpy, rowvar=False)
        reference = tyler_cost(sample_data_numpy, Sigma)
        for factor in (0.1, 3.0, 100.0):
            assert tyler_cost(sample_data_numpy, factor * Sigma) == pytest.approx(
                reference, rel=1e-10
            )

    def test_gradient_matches_finite_differences(self, sample_data_numpy):
        """The Riemannian gradient is the Euclidean one carried by Sigma . Sigma."""
        rng = np.random.default_rng(1)
        Sigma = np.cov(sample_data_numpy, rowvar=False)
        gradient = tyler_riemannian_gradient(sample_data_numpy, Sigma)
        euclidean = np.linalg.solve(Sigma, np.linalg.solve(Sigma, gradient).T).T

        direction = rng.standard_normal((5, 5))
        direction = 0.5 * (direction + direction.T)
        step = 1e-6
        numerical = (
            tyler_cost(sample_data_numpy, Sigma + step * direction)
            - tyler_cost(sample_data_numpy, Sigma - step * direction)
        ) / (2 * step)
        assert numerical == pytest.approx(np.trace(euclidean @ direction), rel=1e-5)

    def test_gradient_vanishes_at_the_fixed_point(self, sample_data_numpy):
        Sigma, _ = minimize_tyler_fixed_point(sample_data_numpy, iter_max=300)
        gradient = tyler_riemannian_gradient(sample_data_numpy, Sigma)
        assert np.linalg.norm(gradient) < 1e-9


class TestTylerMinimisers:
    # The Euclidean descent converges to the same point but far more slowly —
    # that is the whole point of the convergence figure — so it is granted a
    # looser tolerance rather than an unreasonable iteration budget.
    @pytest.mark.parametrize(
        "minimize, tolerance",
        [
            (minimize_tyler_fixed_point, 1e-6),
            (minimize_tyler_riemannian, 1e-6),
            (minimize_tyler_euclidean, 1e-1),
        ],
    )
    def test_reaches_the_same_point(self, sample_data_numpy, minimize, tolerance):
        """The criterion has one minimiser; the three ways of reaching it agree."""
        reference = normalize_determinant(
            np.asarray(
                TylerEstimator(normalization="det", tol=1e-12, iter_max=1000).compute(
                    sample_data_numpy
                )
            )
        )
        Sigma, history = minimize(sample_data_numpy, iter_max=2000, tol=1e-10)
        manifold = HermitianPositiveDefinite(5)
        assert float(manifold.dist(np.asarray(Sigma), reference)) < tolerance
        assert history["cost"][-1] <= history["cost"][0]

    def test_cost_decreases_monotonically(self, sample_data_numpy):
        """The line search guarantees it for the descents; check it holds."""
        for minimize in (minimize_tyler_riemannian, minimize_tyler_euclidean):
            _, history = minimize(sample_data_numpy, iter_max=50)
            costs = np.array(history["cost"])
            assert np.all(np.diff(costs) <= 1e-12)

    def test_fixed_point_is_a_unit_gradient_step(self, sample_data_numpy):
        """Sigma - grad L(Sigma) is Tyler's update, up to the normalisation."""
        Sigma = np.eye(5)
        gradient = tyler_riemannian_gradient(sample_data_numpy, Sigma)
        quadratic = np.einsum("ni,ij,nj->n", sample_data_numpy, np.eye(5), sample_data_numpy)
        update = 5 * (sample_data_numpy.T @ (sample_data_numpy / quadratic[:, None])) / 200
        assert np.allclose(Sigma - gradient, update)

    def test_torch_backend_agrees_with_numpy(self, sample_data_numpy, sample_data_torch):
        Sigma_numpy, _ = minimize_tyler_fixed_point(sample_data_numpy, iter_max=300)
        Sigma_torch, _ = minimize_tyler_fixed_point(
            sample_data_torch, iter_max=300, backend_name="torch-cpu"
        )
        assert np.allclose(np.asarray(Sigma_numpy), to_numpy(Sigma_torch), atol=1e-8)


class TestFrechetMean:
    def test_two_matrices_give_the_geometric_mean(self, covariances_numpy):
        """For two points the Fréchet mean is the midpoint of the geodesic."""
        pair = covariances_numpy[:2]
        mean, _ = frechet_mean_affine_invariant(pair, tol=1e-12)
        manifold = HermitianPositiveDefinite(4)
        midpoint = manifold.exp(pair[0], 0.5 * manifold.log(pair[0], pair[1]))
        assert np.allclose(mean, midpoint, atol=1e-8)

    def test_determinant_is_the_geometric_mean_of_determinants(self, covariances_numpy):
        """log det is linear along geodesics, so the mean inherits it."""
        mean, _ = frechet_mean_affine_invariant(covariances_numpy, tol=1e-12)
        determinants = np.linalg.det(covariances_numpy)
        assert np.linalg.det(mean) == pytest.approx(
            np.exp(np.log(determinants).mean()), rel=1e-8
        )

    def test_congruence_equivariance(self, covariances_numpy):
        """The mean of A Sigma A^T is A (mean of Sigma) A^T: it sees no basis."""
        rng = np.random.default_rng(3)
        transform = rng.standard_normal((4, 4))
        transformed = np.stack(
            [transform @ matrix @ transform.T for matrix in covariances_numpy]
        )
        mean, _ = frechet_mean_affine_invariant(covariances_numpy, tol=1e-12)
        mean_transformed, _ = frechet_mean_affine_invariant(transformed, tol=1e-12)
        assert np.allclose(
            mean_transformed, transform @ mean @ transform.T, rtol=1e-6, atol=1e-6
        )

    def test_gradient_norm_decreases(self, covariances_numpy):
        _, history = frechet_mean_affine_invariant(covariances_numpy, tol=1e-12)
        norms = np.array(history["gradient_norm"])
        assert norms[-1] < norms[0]
        assert norms[-1] < 1e-8
