# Tests for the two K-means on the cone. The module had no tests at all until an
# audit said so; the cross-backend cases are the ones that matter, since two
# numpy/CUDA disagreements were found here by hand and neither would have shown
# up in a suite that only ran on the host.

import numpy as np
import pytest
import torch

from hdrlib.learning.clustering import (
    SPD_METRICS,
    clustering_accuracy,
    match_labels,
    mean_iou,
    riemannian_kmeans,
    spd_kmeans,
    squared_fisher_distance,
)
from hdrlib.learning.rmt import scm

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ── data ──────────────────────────────────────────────────────────────────────


def two_populations(seed=0, n_per=60, n_samples=25, n_features=5, spread=6.0):
    """Two well-separated SPD populations, and the partition that generated them.

    Separated by an anisotropic congruence rather than by scale: the
    affine-invariant metric is invariant to a common positive factor, so two
    populations differing only by scale would be a fair test of ``euclid`` and
    no test at all of ``riemann``.
    """
    rng = np.random.default_rng(seed)
    stretch = np.diag(np.linspace(1.0, spread, n_features))
    windows, truth = [], []
    for cluster, mixing in enumerate((np.eye(n_features), stretch)):
        draw = rng.standard_normal((n_per, n_samples, n_features)) @ mixing
        windows.append(draw)
        truth.append(np.full(n_per, cluster))
    order = rng.permutation(2 * n_per)
    return np.concatenate(windows)[order], np.concatenate(truth)[order]


def spd_batch(seed=0, count=40, n_features=4):
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((count, n_features, n_features))
    return factor @ np.swapaxes(factor, -1, -2) + n_features * np.eye(n_features)


def assert_same_partition(actual, expected):
    """Compare partitions, not label numbers.

    Which cluster gets called 0 and which 1 is not something K-means promises,
    and it is not stable across backends: numpy and CUDA can walk to the same
    partition under complementary names. Matching with the module's own
    Hungarian helper compares what the algorithm actually determines.
    """
    np.testing.assert_array_equal(match_labels(actual, expected), expected)


# ── spd_kmeans: does each metric recover an obvious partition? ────────────────


@pytest.mark.parametrize("metric", SPD_METRICS)
def test_spd_kmeans_recovers_two_populations(metric):
    windows, truth = two_populations()
    covariances = scm(windows, "numpy")
    labels, inertia, histories = spd_kmeans(
        covariances, 2, metric=metric, n_init=3, seed=0, backend="numpy"
    )
    accuracy = clustering_accuracy(match_labels(labels, truth), truth)
    assert accuracy > 0.95, f"{metric} recovered only {accuracy:.3f}"
    assert np.isfinite(inertia) and inertia > 0
    assert len(histories) == 3


@pytest.mark.parametrize("metric", SPD_METRICS)
def test_spd_kmeans_labels_are_a_partition(metric):
    """Every point labelled, both clusters used, nothing out of range."""
    windows, _ = two_populations()
    labels, _, _ = spd_kmeans(
        scm(windows, "numpy"), 2, metric=metric, n_init=2, seed=0, backend="numpy"
    )
    assert labels.shape == (windows.shape[0],)
    assert labels.dtype == np.int64
    assert set(np.unique(labels)) == {0, 1}


def test_spd_kmeans_is_deterministic_under_a_fixed_seed():
    windows, _ = two_populations()
    covariances = scm(windows, "numpy")
    first = spd_kmeans(covariances, 2, n_init=2, seed=7, backend="numpy")
    second = spd_kmeans(covariances, 2, n_init=2, seed=7, backend="numpy")
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_spd_kmeans_rejects_single_precision():
    """The guard exists because float32 loses the sign of the small eigenvalues."""
    windows, _ = two_populations()
    covariances = scm(windows, "numpy").astype(np.float32)
    with pytest.raises(TypeError, match="float64"):
        spd_kmeans(covariances, 2, backend="numpy")


def test_spd_kmeans_refuses_jax():
    """Documented refusal: jax would silently compute this in single precision."""
    windows, _ = two_populations()
    with pytest.raises(ValueError, match="jax"):
        spd_kmeans(scm(windows, "numpy"), 2, backend="jax-cpu")


def test_spd_kmeans_rejects_unknown_metric():
    windows, _ = two_populations()
    with pytest.raises(KeyError):
        spd_kmeans(scm(windows, "numpy"), 2, metric="euclidian", backend="numpy")


# ── the property the batch bound is allowed to have: none ─────────────────────


def test_spd_kmeans_inertia_is_independent_of_max_batch():
    """max_batch slices a reduction; it must not move a single bit of the answer.

    The affine-invariant assignment is sliced because cuSOLVER refuses a batch
    of n_clusters x n_points matrices. Every matrix in it is independent of the
    others, so where the slices fall cannot matter — and the docstring promises
    exactly that.
    """
    windows, _ = two_populations()
    covariances = scm(windows, "numpy")
    reference = None
    for max_batch in (16, 97, 1000, 16000):
        labels, inertia, _ = spd_kmeans(
            covariances, 2, metric="riemann", n_init=2, seed=3,
            max_batch=max_batch, backend="numpy",
        )
        if reference is None:
            reference = (labels, inertia)
            continue
        np.testing.assert_array_equal(labels, reference[0])
        assert inertia == reference[1], f"max_batch={max_batch} moved the inertia"


# ── cross-backend agreement ───────────────────────────────────────────────────


@cuda
@pytest.mark.parametrize("metric", SPD_METRICS)
def test_spd_kmeans_agrees_between_numpy_and_cuda(metric):
    """The failure mode this module has actually had, twice."""
    windows, _ = two_populations()
    covariances = scm(windows, "numpy")
    host = spd_kmeans(
        covariances, 2, metric=metric, n_init=2, seed=1, backend="numpy"
    )
    device = spd_kmeans(
        torch.as_tensor(covariances, device="cuda"), 2, metric=metric,
        n_init=2, seed=1, backend="torch-cuda",
    )
    assert_same_partition(device[0], host[0])
    # Inertia is a sum over the whole set, so it is naming-invariant and can be
    # compared directly; the flat metrics are closed-form and agree far tighter
    # than this, the affine-invariant one carries its eigensolver's noise.
    assert host[1] == pytest.approx(device[1], rel=1e-6)


@cuda
def test_riemannian_kmeans_agrees_between_numpy_and_cuda():
    """Covers the four estimator methods, including the one that disagreed."""
    windows, _ = two_populations(n_per=30)
    for method in ("SCM", "LW", "LW-NL", "RMT"):
        host = riemannian_kmeans(
            windows, 2, method=method, n_init=1, max_iter=4, seed=2,
            backend="numpy",
        )
        device = riemannian_kmeans(
            torch.as_tensor(windows, device="cuda"), 2, method=method,
            n_init=1, max_iter=4, seed=2, backend="torch-cuda",
        )
        assert_same_partition(device[0], host[0])
        # The partition is the assertion that matters and is exact. The inertia
        # is only the value of the objective at wherever the descent stopped,
        # and RMT gets a looser bound than the rest for a reason: its corrected
        # cost is flat near the optimum, so the iterate is not pinned to better
        # than ~1e-4 even between two runs of the same code on different
        # hardware. The plain methods are closed-form or steep and stay tight.
        tolerance = 1e-3 if method == "RMT" else 1e-4
        assert host[1] == pytest.approx(device[1], rel=tolerance), method


# ── riemannian_kmeans ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["SCM", "LW", "LW-NL", "RMT"])
def test_riemannian_kmeans_recovers_two_populations(method):
    windows, truth = two_populations(n_per=40)
    labels, inertia, histories = riemannian_kmeans(
        windows, 2, method=method, n_init=2, max_iter=15, seed=0, backend="numpy"
    )
    accuracy = clustering_accuracy(match_labels(labels, truth), truth)
    assert accuracy > 0.95, f"{method} recovered only {accuracy:.3f}"
    assert np.isfinite(inertia)


def test_riemannian_kmeans_rejects_single_precision():
    """Was unguarded: SCM, LW and LW-NL ran to completion in float32."""
    windows, _ = two_populations(n_per=20)
    with pytest.raises(TypeError, match="float64"):
        riemannian_kmeans(
            windows.astype(np.float32), 2, n_init=1, max_iter=2, backend="numpy"
        )


def test_riemannian_kmeans_refuses_jax():
    windows, _ = two_populations(n_per=20)
    with pytest.raises(ValueError, match="jax"):
        riemannian_kmeans(windows, 2, n_init=1, max_iter=2, backend="jax-cpu")


def test_riemannian_kmeans_estimation_is_hoisted_out_of_the_loop():
    """The covariances depend on the windows alone, so they are formed once.

    Guards the refactor that moved the estimator out of the re-estimation: if it
    ever drifts back inside, this count grows with max_iter.
    """
    import hdrlib.learning.clustering as clustering

    windows, _ = two_populations(n_per=20)
    calls = []
    original = clustering.scm

    def counted(data, backend):
        calls.append(1)
        return original(data, backend)

    clustering.scm = counted
    try:
        riemannian_kmeans(
            windows, 2, method="SCM", n_init=2, max_iter=6, seed=0, backend="numpy"
        )
    finally:
        clustering.scm = original
    assert calls, "scm was never called"
    assert len(calls) == 1, f"estimated {len(calls)} times, expected once per run"


# ── squared_fisher_distance ───────────────────────────────────────────────────


def test_fisher_distance_to_self_is_zero():
    batch = spd_batch()
    distance = squared_fisher_distance(batch[0], batch[0][None], "numpy")
    assert distance[0] == pytest.approx(0.0, abs=1e-18)


def test_fisher_distance_is_affine_invariant():
    """The defining property of the metric: congruence is an isometry.

    This is what makes the correction in rmt.py a *departure* from something,
    so it is worth pinning rather than assuming.
    """
    rng = np.random.default_rng(4)
    batch = spd_batch(count=12)
    reference = batch[0]
    transform = rng.standard_normal((batch.shape[-1], batch.shape[-1]))
    plain = squared_fisher_distance(reference, batch[1:], "numpy")
    moved = squared_fisher_distance(
        transform @ reference @ transform.T,
        transform @ batch[1:] @ np.swapaxes(transform, -1, -2),
        "numpy",
    )
    np.testing.assert_allclose(moved, plain, rtol=1e-8)


def test_fisher_distance_is_symmetric():
    batch = spd_batch(count=6)
    forward = [
        float(squared_fisher_distance(batch[0], batch[i][None], "numpy")[0])
        for i in range(1, 6)
    ]
    backward = [
        float(squared_fisher_distance(batch[i], batch[0][None], "numpy")[0])
        for i in range(1, 6)
    ]
    np.testing.assert_allclose(forward, backward, rtol=1e-8)


# ── scoring helpers ───────────────────────────────────────────────────────────


def test_match_labels_is_a_permutation_and_accuracy_follows():
    truth = np.array([0, 0, 1, 1, 2, 2])
    # The same partition, named differently: scoring must not punish that.
    permuted = np.array([2, 2, 0, 0, 1, 1])
    matched = match_labels(permuted, truth)
    np.testing.assert_array_equal(matched, truth)
    assert clustering_accuracy(matched, truth) == pytest.approx(1.0)


def test_mean_iou_is_one_on_a_perfect_partition_and_bounded():
    truth = np.array([0, 0, 1, 1, 2, 2])
    ious, miou = mean_iou(truth, truth)
    assert miou == pytest.approx(1.0)
    assert np.all((ious >= 0) & (ious <= 1))

    wrong = np.array([0, 1, 2, 0, 1, 2])
    _, worse = mean_iou(wrong, truth)
    assert 0.0 <= worse < 1.0
