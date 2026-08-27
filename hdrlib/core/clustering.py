# Riemannian K-means on the cone, with and without the RMT correction.
#
# The two algorithms differ by one thing only: the distance whose sum of
# squares the centroids minimise. Everything else — the alternation between
# assignment and re-estimation, the restarts, the choice of the best one by
# inertia — is common, and is written once here so that a comparison between
# them measures the correction and not the optimiser.
#
# That single difference is the whole point of sec:learning-frechet: the
# shrinkage baselines regularise every covariance *before* averaging, while
# the corrected version leaves the covariances alone and corrects the distance
# the average minimises.
#
# No scikit-learn anywhere, here or in rmt.py: it is numpy-only, and one import
# would pin the whole pipeline to the CPU. The shrinkage estimators are written
# out by hand for the same reason. scipy appears twice, and only for host-side
# work that never touches the data on the device: reading a .mat file, and a
# 16x16 assignment problem on the label numbering.

from typing import Callable, Optional, Tuple, Union

import numpy as np

from .backend import Array, Backend, get_backend_module, get_data_on_device, to_numpy
from .rmt import (
    analytical_shrinkage,
    frechet_mean_cholesky,
    ledoit_wolf_linear,
    rmt_corrected_squared_distance,
    rmt_frechet_mean,
    scm,
)


__all__ = [
    "squared_fisher_distance",
    "riemannian_kmeans",
    "match_labels",
    "clustering_accuracy",
    "mean_iou",
]


def squared_fisher_distance(
    reference: Array, covariances: Array, backend: Union[str, Backend] = "numpy"
) -> Array:
    """Squared affine-invariant distance from one SPD matrix to a set of them.

    Normalised by the dimension, like the corrected version it is compared
    against, so that the two are on the same scale.
    """
    be = get_backend_module(backend)
    n_features = covariances.shape[-1]
    inverse_factor = be.linalg.inv(be.linalg.cholesky(reference))
    transformed = inverse_factor @ covariances @ be.swapaxes(inverse_factor, -1, -2)
    logarithms = be.log(be.linalg.eigvalsh(transformed))
    return be.einsum("...i,...i->...", logarithms, logarithms) / n_features


# ── the two estimator families ────────────────────────────────────────────────


def _plain_centroid_factory(shrinkage: Optional[str]):
    """Centroid and distance of a two-step method: regularise, then average."""

    def centroids(windows, labels, n_clusters, n_samples, backend, max_iterations):
        be = get_backend_module(backend)
        if shrinkage is None:
            covariances = scm(windows, backend)
        elif shrinkage == "lw":
            covariances = ledoit_wolf_linear(windows, backend)
        elif shrinkage == "lwnl":
            covariances = analytical_shrinkage(windows, backend, shrink=0)
        else:
            raise ValueError(f"unknown shrinkage {shrinkage!r}")
        out = []
        for cluster in range(n_clusters):
            member = labels == cluster
            out.append(
                frechet_mean_cholesky(
                    covariances[member], max_iterations=max_iterations,
                    backend=backend,
                )[0]
            )
        return out, covariances

    def distances(centre, covariances, n_samples, backend):
        return squared_fisher_distance(centre, covariances, backend)

    return centroids, distances


def _rmt_centroid_factory():
    """Centroid and distance of the corrected method."""

    def centroids(windows, labels, n_clusters, n_samples, backend, max_iterations):
        covariances = scm(windows, backend)
        out = []
        for cluster in range(n_clusters):
            member = labels == cluster
            # Warm start on the plain mean of the cluster's SCMs: the corrected
            # descent has no closed-form optimality condition, so a decent
            # starting point saves most of its iterations.
            warm = frechet_mean_cholesky(
                covariances[member], max_iterations=20, backend=backend
            )[0]
            out.append(
                rmt_frechet_mean(
                    windows[member], init=warm, max_iterations=max_iterations,
                    backend=backend,
                )[0]
            )
        return out, covariances

    def distances(centre, covariances, n_samples, backend):
        return rmt_corrected_squared_distance(
            centre, covariances, n_samples, backend
        )

    return centroids, distances


METHODS = {
    "SCM": lambda: _plain_centroid_factory(None),
    "LW": lambda: _plain_centroid_factory("lw"),
    "LW-NL": lambda: _plain_centroid_factory("lwnl"),
    "RMT": _rmt_centroid_factory,
}


# ── the algorithm ─────────────────────────────────────────────────────────────


def _random_labels(rng, n_points: int, n_clusters: int) -> np.ndarray:
    """Random assignment in which every cluster is non-empty.

    An empty cluster has no centroid, so the first re-estimation would fail;
    redrawing is simpler and costs nothing at these sizes.
    """
    for _ in range(1000):
        labels = rng.integers(n_clusters, size=n_points)
        if len(np.unique(labels)) == n_clusters:
            return labels
    raise RuntimeError("could not draw an assignment with every cluster non-empty")


def _revive_empty_clusters(labels, distances, n_clusters):
    """Give every empty cluster a point, without emptying another one.

    K-means on a real scene does empty clusters: two crops can be close enough
    that one centroid captures both. A cluster with no member has no centroid,
    so the next re-estimation would average nothing and return NaN — and NaN
    then propagates silently through a whole run.

    Each empty cluster is handed the point that its own centroid serves best
    among those still available, taken from a cluster that can spare one. The
    "still available" matters: the naive version gives the same point to two
    empty clusters, which leaves the second one empty and reintroduces the bug
    it was meant to fix.
    """
    labels = labels.copy()
    for cluster in range(n_clusters):
        if np.any(labels == cluster):
            continue
        counts = np.bincount(labels, minlength=n_clusters)
        # Only points whose current cluster would survive losing them.
        movable = counts[labels] > 1
        if not np.any(movable):
            break
        candidates = np.where(movable)[0]
        labels[candidates[distances[candidates, cluster].argmin()]] = cluster
    return labels


def riemannian_kmeans(
    windows: Array,
    n_clusters: int,
    method: str = "SCM",
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-3,
    mean_iterations: int = 50,
    seed: int = 42,
    backend: Union[str, Backend] = "numpy",
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """K-means on the cone, alternating assignment and Fréchet re-estimation.

    Parameters
    ----------
    windows : Array of shape (n_points, n_samples, n_features)
        One block of samples per point to cluster — for an image, the pixel's
        neighbourhood. The covariances are formed inside, because the corrected
        method needs the samples and not only their covariance.
    n_clusters : int
    method : {"SCM", "LW", "LW-NL", "RMT"}
        What the centroids and the distances are. Only this changes between
        the curves of a comparison.
    n_init : int
        Restarts from a random assignment; the one of least inertia is kept.
    max_iter, tol : int, float
        Stop when fewer than ``tol`` of the points change cluster.
    mean_iterations : int
        Iteration budget of one Fréchet mean.
    seed : int
    backend : str or Backend
    verbose : bool

    Returns
    -------
    labels : ndarray of shape (n_points,)
    inertia : float
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    centroid_fn, distance_fn = METHODS[method]()

    device_windows = get_data_on_device(windows, backend)
    n_points, n_samples, _ = windows.shape
    rng = np.random.default_rng(seed)

    best_labels, best_inertia = None, np.inf
    for restart in range(n_init):
        labels = _random_labels(rng, n_points, n_clusters)
        centres, covariances = centroid_fn(
            device_windows, labels, n_clusters, n_samples, backend, mean_iterations
        )

        for iteration in range(max_iter):
            distances = np.stack(
                [
                    to_numpy(distance_fn(centre, covariances, n_samples, backend))
                    for centre in centres
                ],
                axis=1,
            )
            new_labels = distances.argmin(axis=1)
            new_labels = _revive_empty_clusters(new_labels, distances, n_clusters)
            moved = np.mean(new_labels != labels)
            labels = new_labels
            centres, covariances = centroid_fn(
                device_windows, labels, n_clusters, n_samples, backend,
                mean_iterations,
            )
            if moved <= tol:
                break

        distances = np.stack(
            [
                to_numpy(distance_fn(centre, covariances, n_samples, backend))
                for centre in centres
            ],
            axis=1,
        )
        inertia = float(
            sum(distances[labels == c, c].sum() for c in range(n_clusters))
        )
        if verbose:
            print(f"    {method} init {restart + 1}/{n_init}: "
                  f"inertia {inertia:.3f} in {iteration + 1} iterations")
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia

    return best_labels, best_inertia


# ── scoring an unsupervised partition ─────────────────────────────────────────


def match_labels(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Relabel a partition to agree as well as possible with a ground truth.

    A clustering has no reason to name its groups the way the ground truth
    does, so the two are matched by the assignment that maximises agreement
    (Hungarian algorithm) before any score is computed. Without this step every
    score would measure the arbitrary numbering of the clusters.
    """
    from scipy.optimize import linear_sum_assignment

    truth_classes = np.unique(truth)
    prediction_classes = np.unique(prediction)
    cost = np.zeros((len(truth_classes), len(prediction_classes)))
    for i, truth_class in enumerate(truth_classes):
        mask = truth == truth_class
        for j, prediction_class in enumerate(prediction_classes):
            cost[i, j] = -np.sum(prediction[mask] == prediction_class)

    rows, columns = linear_sum_assignment(cost)
    matched = prediction.copy()
    for row, column in zip(rows, columns):
        matched[prediction == prediction_classes[column]] = truth_classes[row]
    return matched


def clustering_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of labelled pixels put in the right group, after matching."""
    labelled = truth > 0
    return float(np.mean(prediction[labelled] == truth[labelled]))


def mean_iou(prediction: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, float]:
    """Intersection over union, per class and averaged, after matching.

    Reported alongside the accuracy because the two disagree in an informative
    way: accuracy is dominated by the large classes, mIoU is not. A method can
    win on one and lose on the other, and the memoir says so.
    """
    labelled = truth > 0
    classes = np.unique(truth[labelled])
    ious = []
    for cls in classes:
        intersection = np.sum((prediction == cls) & (truth == cls) & labelled)
        union = np.sum(((prediction == cls) | (truth == cls)) & labelled)
        ious.append(intersection / union if union else 0.0)
    ious = np.asarray(ious)
    return ious, float(ious.mean())
