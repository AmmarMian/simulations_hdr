# Tests for the estimators, the corrected distance and the two Fréchet means.

import numpy as np
import pytest
import torch

from hdrlib.core.estimation import frechet_mean_affine_invariant
from hdrlib.learning.rmt import (
    analytical_shrinkage,
    frechet_mean_cholesky,
    ledoit_wolf_linear,
    oas,
    rmt_corrected_squared_distance,
    rmt_frechet_mean,
    scm,
)

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

SHRINKAGES = {
    "ledoit_wolf_linear": lambda d, b: ledoit_wolf_linear(d, b),
    "oas": lambda d, b: oas(d, b),
    "analytical_shrinkage": lambda d, b: analytical_shrinkage(d, b, shrink=0),
}


def gaussian(seed=0, count=25, n_samples=40, n_features=4):
    return np.random.default_rng(seed).standard_normal((count, n_samples, n_features))


def spd(seed=0, count=25, n_features=4):
    return scm(gaussian(seed, count, n_features=n_features), "numpy")


# ── scm ───────────────────────────────────────────────────────────────────────


def test_scm_matches_the_definition():
    """Centred by assumption, so it is X^T X / n and not a centred estimate."""
    data = gaussian()
    expected = np.stack([x.T @ x / x.shape[0] for x in data])
    np.testing.assert_allclose(scm(data, "numpy"), expected, rtol=1e-13)


def test_scm_is_spd_when_oversampled():
    eigenvalues = np.linalg.eigvalsh(spd())
    assert eigenvalues.min() > 0


# ── the shrinkage estimators ──────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(SHRINKAGES))
def test_shrinkage_outputs_are_spd_and_symmetric(name):
    out = SHRINKAGES[name](gaussian(), "numpy")
    np.testing.assert_allclose(out, np.swapaxes(out, -1, -2), rtol=1e-12)
    assert np.linalg.eigvalsh(out).min() > 0


@pytest.mark.parametrize("name", list(SHRINKAGES))
def test_shrinkage_is_rotation_equivariant(name):
    """Shrinking towards a scaled identity commutes with rotations.

    Not affine equivariance — the target singles out a scale, so a general
    congruence is not expected to commute. Rotation is, and it is the property
    that catches an index transposed somewhere in the eigenbasis.
    """
    rng = np.random.default_rng(5)
    data = gaussian()
    rotation, _ = np.linalg.qr(rng.standard_normal((data.shape[-1],) * 2))
    plain = SHRINKAGES[name](data, "numpy")
    rotated = SHRINKAGES[name](data @ rotation.T, "numpy")
    np.testing.assert_allclose(
        rotated, rotation @ plain @ rotation.T, atol=1e-12
    )


@pytest.mark.parametrize("name", list(SHRINKAGES))
def test_shrinkage_is_better_conditioned_than_the_scm(name):
    """The point of shrinking: pull the spectrum in, at c = p/n = 0.5."""
    data = gaussian(n_samples=8, n_features=4)
    plain = np.linalg.cond(scm(data, "numpy"))
    shrunk = np.linalg.cond(SHRINKAGES[name](data, "numpy"))
    assert np.median(shrunk) < np.median(plain)


@pytest.mark.parametrize("name", list(SHRINKAGES))
@pytest.mark.parametrize("backend", ["torch-cpu", pytest.param("torch-cuda", marks=cuda)])
def test_shrinkage_agrees_across_backends(name, backend):
    """Extends the numpy/torch-cpu check to the backend the LW-NL bug was on."""
    data = gaussian()
    reference = SHRINKAGES[name](data, "numpy")
    other = SHRINKAGES[name](
        torch.as_tensor(data, device="cuda" if "cuda" in backend else "cpu"), backend
    )
    np.testing.assert_allclose(
        np.asarray(other.cpu()), reference, rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("name", list(SHRINKAGES))
def test_shrinkage_is_sign_flip_equivariant(name):
    """A signature change is a congruence by a diagonal of ±1."""
    rng = np.random.default_rng(1)
    data = gaussian()
    signs = rng.choice([-1.0, 1.0], size=data.shape[-1])
    plain = SHRINKAGES[name](data, "numpy")
    flipped = SHRINKAGES[name](data * signs, "numpy")
    np.testing.assert_allclose(flipped, plain * np.outer(signs, signs), atol=1e-12)


# ── the two Fréchet means ─────────────────────────────────────────────────────


def test_frechet_mean_cholesky_matches_the_other_optimiser():
    """The docstring says these compute the same object by a different route."""
    covariances = spd()
    by_cholesky, _ = frechet_mean_cholesky(
        covariances, tol=1e-12, max_iterations=200, backend="numpy"
    )
    by_gradient, _ = frechet_mean_affine_invariant(covariances, tol=1e-12)
    np.testing.assert_allclose(by_cholesky, by_gradient, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "mean_of",
    [
        pytest.param("cholesky", id="frechet_mean_cholesky"),
        pytest.param("rmt", id="rmt_frechet_mean"),
    ],
)
def test_frechet_means_are_congruence_equivariant(mean_of):
    """mean(A X Aᵀ) = A mean(X) Aᵀ — the defining symmetry of the geometry.

    Both costs depend on the iterate only through the eigenvalues of
    ``L⁻¹ S L⁻ᵀ``, which a congruence leaves alone, so both minimisers should be
    equivariant exactly. The plain one is, to 1e-9. The corrected one is only
    equivariant to between 7e-4 and 6e-3, measured over six transforms of
    condition number 9 to 1700 — and it is not a conditioning effect, nor an
    unconverged descent, since its own step size reaches 5e-16 before it stops.
    Its optimum is simply flat enough that the descent settles in different
    places. The two tolerances below record that gap rather than hide it: 1e-2
    still fails loudly if the corrected mean ever drifts an order of magnitude.
    """
    rng = np.random.default_rng(3)
    data = gaussian()
    transform = rng.standard_normal((data.shape[-1],) * 2)

    if mean_of == "cholesky":
        covariances = scm(data, "numpy")
        plain, _ = frechet_mean_cholesky(
            covariances, tol=1e-12, max_iterations=200, backend="numpy"
        )
        moved, _ = frechet_mean_cholesky(
            transform @ covariances @ transform.T,
            tol=1e-12, max_iterations=200, backend="numpy",
        )
        tolerance = 1e-6
    else:
        plain, _ = rmt_frechet_mean(
            data, tol=1e-10, max_iterations=200, backend="numpy"
        )
        moved, _ = rmt_frechet_mean(
            data @ transform.T, tol=1e-10, max_iterations=200, backend="numpy"
        )
        tolerance = 1e-2

    target = transform @ plain @ transform.T
    assert np.abs(moved - target).max() / np.abs(target).max() < tolerance


@pytest.mark.parametrize(
    "mean_fn",
    [
        pytest.param(lambda d: frechet_mean_cholesky(scm(d, "numpy"), backend="numpy"),
                     id="frechet_mean_cholesky"),
        pytest.param(lambda d: rmt_frechet_mean(d, backend="numpy"),
                     id="rmt_frechet_mean"),
    ],
)
def test_frechet_means_return_spd(mean_fn):
    mean, _ = mean_fn(gaussian())
    np.testing.assert_allclose(mean, mean.T, rtol=1e-10)
    assert np.linalg.eigvalsh(mean).min() > 0


def test_frechet_mean_cholesky_of_one_matrix_is_that_matrix():
    one = spd(count=1)
    mean, _ = frechet_mean_cholesky(
        one, tol=1e-12, max_iterations=200, backend="numpy"
    )
    np.testing.assert_allclose(mean, one[0], rtol=1e-6, atol=1e-8)


def test_frechet_means_reject_single_precision():
    with pytest.raises(TypeError, match="float64"):
        rmt_frechet_mean(gaussian().astype(np.float32), backend="numpy")


# ── the corrected distance ────────────────────────────────────────────────────


def test_corrected_distance_is_finite_on_estimated_covariances():
    covariances = spd(count=60)
    distances = rmt_corrected_squared_distance(
        covariances[0], covariances[1:], 40, "numpy"
    )
    assert np.all(np.isfinite(distances))


def test_corrected_distance_separates_populations():
    """It must still order things: far covariances beyond near ones."""
    rng = np.random.default_rng(2)
    near = scm(rng.standard_normal((30, 40, 4)), "numpy")
    stretch = np.diag([1.0, 3.0, 6.0, 10.0])
    far = scm(rng.standard_normal((30, 40, 4)) @ stretch, "numpy")
    reference, _ = frechet_mean_cholesky(near, backend="numpy")
    d_near = rmt_corrected_squared_distance(reference, near, 40, "numpy")
    d_far = rmt_corrected_squared_distance(reference, far, 40, "numpy")
    assert d_near.mean() < d_far.mean()


def test_corrected_distance_is_degenerate_on_a_degenerate_spectrum():
    """Documents a real limitation rather than pretending it isn't there.

    The correction divides by ``(λi − λj)²`` off the diagonal, so a whitened
    matrix whose eigenvalues coincide gives 0/0. Comparing a matrix with itself
    — or with any multiple of itself — whitens to a multiple of the identity and
    hits it exactly. Estimated covariances are never degenerate to that
    precision (the test above covers real ones), but the reflexivity a distance
    would normally be expected to satisfy cannot be asserted here, and pinning
    it stops someone "fixing" the NaN by accident and calling it converged.
    """
    one = spd(count=1)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        same = rmt_corrected_squared_distance(one, one[None], 40, "numpy")
        scaled = rmt_corrected_squared_distance(one, 2.0 * one[None], 40, "numpy")
    assert not np.isfinite(same[0]) or abs(same[0]) > 1.0
    assert not np.isfinite(scaled[0]) or abs(scaled[0]) > 1.0


@cuda
def test_corrected_distance_agrees_across_backends():
    covariances = spd(count=40)
    host = rmt_corrected_squared_distance(
        covariances[0], covariances[1:], 40, "numpy"
    )
    device = rmt_corrected_squared_distance(
        torch.as_tensor(covariances[0], device="cuda"),
        torch.as_tensor(covariances[1:], device="cuda"),
        40, "torch-cuda",
    )
    np.testing.assert_allclose(device.cpu().numpy(), host, rtol=1e-8, atol=1e-10)
