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

from .backend import (
    Array,
    Backend,
    batched_eigh,
    cast_like,
    concatenate,
    get_backend_module,
    get_data_on_device,
    masked_set,
    require_double as _require_double,
    sample_uniform,
    to_numpy,
    to_scalar,
)
from .manifolds import logm_psd, sqrtm_invsqrtm_psd
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
    "SPD_METRICS",
    "spd_kmeans",
    "match_labels",
    "clustering_accuracy",
    "mean_iou",
    "reference_mean_iou",
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
    # batched_eigh rather than linalg.eigvalsh: on CUDA the latter reaches a
    # cuSOLVER path that refuses batches this routine is routinely given.
    eigenvalues, _ = batched_eigh(backend, transformed)
    logarithms = be.log(eigenvalues)
    return be.einsum("...i,...i->...", logarithms, logarithms) / n_features


# ── the two estimator families ────────────────────────────────────────────────


# Each family is three functions, and the split is the point: what the
# covariances are depends on the windows alone, what the centroids are depends
# on the labels. Estimating is therefore hoisted out of the alternation and
# done once per run, as it already is in spd_kmeans. It used to sit inside
# ``centroids``, which re-estimated all of them on every re-estimation — up to
# n_init * (max_iter + 1) times, and for LW-NL that is an eigendecomposition
# per window per round. Nothing about the result changes; the estimator is a
# function of the windows, and it was being asked the same question repeatedly.


def _plain_centroid_factory(shrinkage: Optional[str]):
    """Estimator, centroid and distance of a two-step method: regularise, then average."""

    def estimate(windows, backend):
        if shrinkage is None:
            return scm(windows, backend)
        if shrinkage == "lw":
            return ledoit_wolf_linear(windows, backend)
        if shrinkage == "lwnl":
            return analytical_shrinkage(windows, backend, shrink=0)
        raise ValueError(f"unknown shrinkage {shrinkage!r}")

    def centroids(
        windows, covariances, labels, n_clusters, n_samples, backend, max_iterations
    ):
        out = []
        for cluster in range(n_clusters):
            member = labels == cluster
            out.append(
                frechet_mean_cholesky(
                    covariances[member], max_iterations=max_iterations,
                    backend=backend,
                )[0]
            )
        return out

    def distances(centre, covariances, n_samples, backend):
        return squared_fisher_distance(centre, covariances, backend)

    return estimate, centroids, distances


def _rmt_centroid_factory():
    """Estimator, centroid and distance of the corrected method."""

    def estimate(windows, backend):
        return scm(windows, backend)

    def centroids(
        windows, covariances, labels, n_clusters, n_samples, backend, max_iterations
    ):
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
        return out

    def distances(centre, covariances, n_samples, backend):
        return rmt_corrected_squared_distance(
            centre, covariances, n_samples, backend
        )

    return estimate, centroids, distances


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
    histories : list of dict
        One entry per restart: its inertia, how many rounds it took, and what
        fraction of the points was still changing cluster when it stopped.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    # The same two guards spd_kmeans has, for the same reason. They were missing
    # here, and the omission was not uniform across the four methods: RMT would
    # raise from inside rmt_frechet_mean, while SCM, LW and LW-NL would run to
    # completion in single precision and return a partition. Refusing at the top
    # makes the four behave alike.
    if str(backend).startswith("jax"):
        raise ValueError(
            "the jax backends are refused here: jax defaults to float32 and "
            "nothing in this repository calls "
            "jax.config.update('jax_enable_x64', True), so the eigenvalue "
            "logarithms would be computed in single precision without any "
            "warning. Use torch-cuda, cupy or numpy."
        )
    estimate_fn, centroid_fn, distance_fn = METHODS[method]()

    device_windows = get_data_on_device(windows, backend)
    require_double(device_windows, f"the {method} K-means")
    n_points, n_samples, _ = windows.shape
    rng = np.random.default_rng(seed)

    # Once for the whole run, restarts included: the estimator is a function of
    # the windows and nothing below changes them.
    covariances = estimate_fn(device_windows, backend)

    best_labels, best_inertia = None, np.inf
    histories = []
    for restart in range(n_init):
        labels = _random_labels(rng, n_points, n_clusters)
        centres = centroid_fn(
            device_windows, covariances, labels, n_clusters, n_samples, backend,
            mean_iterations,
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
            centres = centroid_fn(
                device_windows, covariances, labels, n_clusters, n_samples,
                backend, mean_iterations,
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
        # ``moved`` is the fraction of points that changed cluster on the last
        # round. Reported because hitting the iteration cap is not by itself a
        # problem: a run that stops with 0.1% of the points still moving has
        # settled, one that stops with 5% moving has not, and only the second
        # makes a comparison meaningless.
        histories.append({"inertia": inertia, "iterations": iteration + 1,
                          "moved": float(moved)})
        if verbose:
            print(f"    {method} init {restart + 1}/{n_init}: "
                  f"inertia {inertia:.3f} in {iteration + 1} iterations, "
                  f"{moved:.3%} still moving", flush=True)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia

    return best_labels, best_inertia, histories


# ── scoring an unsupervised partition ─────────────────────────────────────────


def match_labels(
    prediction: np.ndarray, truth: np.ndarray, match_unlabelled: bool = True
) -> np.ndarray:
    """Relabel a partition to agree as well as possible with a ground truth.

    A clustering has no reason to name its groups the way the ground truth
    does, so the two are matched by the assignment that maximises agreement
    (Hungarian algorithm) before any score is computed. Without this step every
    score would measure the arbitrary numbering of the clusters.

    Parameters
    ----------
    prediction, truth : ndarray
    match_unlabelled : bool
        Whether class 0, the unannotated background, competes for a cluster.

        True reproduces the reference implementation and is the default. It has
        a consequence worth knowing: on these scenes the background is about
        half the image, so it wins one of the clusters outright, and since
        there are as many clusters as annotated classes, one annotated class is
        then left with no cluster at all and scores an IoU of exactly zero. On
        Indian Pines that is class 1 for SCM and RMT and class 9 for LW-NL.

        False matches on the annotated pixels only, so every cluster is spent
        on a class that is actually scored. It raises mIoU by a few thousandths
        and leaves the ranking of the methods alone.
    """
    from scipy.optimize import linear_sum_assignment

    if match_unlabelled:
        truth_classes = np.unique(truth)
        scored = np.ones(truth.shape, dtype=bool)
    else:
        scored = truth > 0
        truth_classes = np.unique(truth[scored])
    prediction_classes = np.unique(prediction)
    cost = np.zeros((len(truth_classes), len(prediction_classes)))
    for i, truth_class in enumerate(truth_classes):
        mask = (truth == truth_class) & scored
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


def reference_mean_iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    """mIoU as the reference implementation computes it, for comparison.

    Two things differ from :func:`mean_iou`. The prediction is first forced to
    agree with the ground truth wherever the ground truth is unannotated, and
    the average is then taken over every class including that background one —
    whose IoU is consequently close to 1 by construction.

    The result is a number a few hundredths above :func:`mean_iou` on the same
    partition: about +0.04 on Indian Pines and +0.02 on Salinas, being roughly
    ``(1 - mIoU) / n_classes``. It shifts every method by nearly the same
    amount and so changes no ranking, but a figure from this repository cannot
    be read against a published table without knowing which of the two it is.
    """
    forced = prediction.copy()
    forced[truth == 0] = 0
    classes = np.union1d(np.unique(forced), np.unique(truth)).astype(np.int64)
    classes = classes[classes >= 0]
    within = np.isin(truth, classes)
    ious = []
    for cls in classes:
        intersection = np.sum((forced == cls) & (truth == cls))
        union = np.sum(((forced == cls) | (truth == cls)) & within)
        ious.append(intersection / union if union else 0.0)
    return float(np.mean(ious))


# ── K-means on the SPD cone, device-resident ──────────────────────────────────
#
# Same alternation as riemannian_kmeans, but every step is written to run where
# the data already is. Three functions carry the geometry — a representation, a
# batched distance, a re-estimation — and the loop calling them is written once.
#
# Two things stay on the host by necessity: the fraction of points that changed
# cluster, which decides the stopping test, and the inertia, which picks the best
# restart. Both are control flow, so both must become Python numbers.

SPD_METRICS = ("euclid", "logeuclid", "riemann")


# Why these metrics in particular cannot be run in single precision. Every one
# of them ends in the eigenvalues of a small SPD matrix and two of the three
# take their logarithm; in float32 the smallest eigenvalue of a covariance
# estimated from a handful of samples has an unreliable sign, and its logarithm
# is then either a large negative number or a NaN — silently, in both cases.
_PRECISION_REASON = (
    "These metrics take the logarithms of the eigenvalues of a small "
    "covariance, which single precision resolves badly once the concentration "
    "approaches one."
)


def require_double(x: Array, what: str) -> None:
    """Refuse to run in single precision.

    Every metric here ends in the eigenvalues of a small SPD matrix and two of
    the three take their logarithm, so the whole comparison rests on how well
    those eigenvalues are resolved.

    How much that costs depends on the concentration. Measured on 200 000
    covariances of 5 variables from 25 samples — c = 0.2, the default
    configuration — float32 loses nothing that matters: no smallest eigenvalue
    changed sign, and the squared-log distance agreed with float64 to 1e-8,
    against method-to-method gaps of order 1e-1. The danger is at c near 1,
    where the smallest eigenvalue approaches zero, its sign stops being
    reliable, and its logarithm is then a large negative number or a NaN —
    silently, in both cases.

    float64 is therefore the default because the guard cannot see which regime
    it is in: it is handed a cube, not a concentration ratio. A caller that
    knows it is far from c = 1 and wants the speed — single precision is 64
    times the double-precision rate on a workstation NVIDIA card — is being
    refused something it could safely have, and that is a deliberate trade, not
    a numerical necessity.

    The check itself lives in :func:`hdrlib.core.backend.require_double`; this
    is the name the experiment scripts import, and it supplies the reason above.
    """
    _require_double(x, what, _PRECISION_REASON)


def _flatten(matrices: Array, backend) -> Array:
    """``(n, p, p) -> (n, p*p)``, so a batch of matrices can be averaged or
    gathered by a plain matrix product."""
    be = get_backend_module(backend)
    n_features = matrices.shape[-1]
    return be.reshape(matrices, (matrices.shape[0], n_features * n_features))


def _unflatten(flat: Array, n_features: int, backend) -> Array:
    be = get_backend_module(backend)
    return be.reshape(flat, (-1, n_features, n_features))


def _expm_sym(matrices: Array, backend) -> Array:
    """Matrix exponential of a batch of symmetric matrices."""
    be = get_backend_module(backend)
    eigenvalues, eigenvectors = batched_eigh(backend, matrices)
    return be.einsum(
        "...ij,...j,...kj->...ik", eigenvectors, be.exp(eigenvalues), eigenvectors
    )


# The labels are held on the device as floats carrying exact small integers,
# not as an integer array. The one-hot membership is built by comparing them
# against the class indices and that comparison has to be exact; every value
# involved is below the number of clusters, so it is. Keeping them in the
# covariances' own dtype means cast_like does every conversion, instead of each
# backend's integer type having to be named at every step.


def _membership(labels: Array, classes: Array, reference: Array, backend) -> Array:
    """One-hot membership, ``(n_clusters, n_points)``."""
    return cast_like(labels[None, :] == classes[:, None], reference, backend)


def _initial_labels(
    rng, n_points: int, n_clusters: int, classes: Array, reference: Array, backend
) -> Array:
    """A random assignment in which every cluster is non-empty.

    Drawing until no cluster is empty would need a host-side test of a device
    array on every redraw. Instead every cluster is seeded: ``n_clusters``
    distinct points are handed one cluster each and the rest are drawn
    uniformly, which makes non-emptiness structural rather than lucky.

    The host ``rng`` drives both draws, so a given seed produces the same
    starting partition whatever the metric.
    """
    be = get_backend_module(backend)
    uniform = sample_uniform(
        n_points, [], backend, seed=int(rng.integers(1, 2**31 - 1))
    )
    # torch.rand is float32 whatever the ambient dtype, and the labels have to
    # share the reference dtype for the comparison and the index write below.
    uniform = cast_like(uniform, reference, backend)
    labels = be.floor(uniform * n_clusters)
    labels = be.minimum(labels, classes[-1])  # guards a uniform draw of exactly 1
    seeded = get_data_on_device(
        rng.choice(n_points, size=n_clusters, replace=False), backend
    )
    return masked_set(labels, seeded, classes, backend)


def _nearest_and_revive(
    distances: Array, classes: Array, point_index: Array, n_clusters: int, backend
) -> Array:
    """Nearest centroid, then give every empty cluster a point.

    Same policy as :func:`_revive_empty_clusters` — each empty cluster takes the
    point its own centroid serves best among those whose cluster can spare one —
    written without branches so that nothing is read back to decide anything.
    The running count is a genuine sequential dependency, so the loop over
    clusters stays; what changes is that its trip count is ``n_clusters``, known
    in advance, rather than the number of empty clusters, which is not.
    """
    be = get_backend_module(backend)
    labels = cast_like(be.argmin(distances, axis=1), distances, backend)
    # Any value above every distance serves as the "not a candidate" sentinel.
    infinity = be.max(distances) + 1.0

    for cluster in range(n_clusters):
        one_hot = _membership(labels, classes, distances, backend)
        counts = be.sum(one_hot, axis=-1)
        # Exactly one row of one_hot is 1 in each column, so this reads off the
        # size of each point's own cluster.
        own_count = be.sum(one_hot * counts[:, None], axis=0)
        score = be.where(own_count > 1.5, distances[:, cluster], infinity)
        best = be.argmin(score)
        take = be.logical_and(point_index == best, counts[cluster] < 0.5)
        labels = be.where(take, classes[cluster], labels)
    return labels


def _flat_metric(logarithm: bool):
    """Euclidean geometry, on the covariances or on their logarithms.

    Both are flat, so they share every kernel and differ only in what they are
    flat on. The log-Euclidean centroids stay in the tangent representation and
    are never exponentiated back: the distance compares logarithms, so the SPD
    form is never needed. The re-estimation is therefore closed-form for both.
    """

    def precompute(covariances, backend):
        return logm_psd(covariances, backend) if logarithm else covariances

    def assign(representation, centroids, backend):
        be = get_backend_module(backend)
        points = _flatten(representation, backend)
        centres = _flatten(centroids, backend)
        # ||R - C||² = <R,R> - 2<R,C> + <C,C>: the cross term is the only real
        # work, one product for all n_points x n_clusters pairs.
        cross = points @ be.swapaxes(centres, -1, -2)
        point_norm = be.sum(points * points, axis=-1)[:, None]
        centre_norm = be.sum(centres * centres, axis=-1)[None, :]
        return point_norm - 2.0 * cross + centre_norm

    def update(representation, one_hot, centroids, backend, mean_iterations, mean_tol):
        be = get_backend_module(backend)
        weights = one_hot / be.sum(one_hot, axis=-1, keepdims=True)
        flat = weights @ _flatten(representation, backend)
        return _unflatten(flat, representation.shape[-1], backend), 0

    return precompute, assign, update


def _affine_invariant_metric(max_batch: int):
    """Affine-invariant geometry: centroids are Karcher means."""

    def precompute(covariances, backend):
        return covariances

    def assign(representation, centroids, backend):
        be = get_backend_module(backend)
        # One whitening per centroid, then every point is measured against all
        # of them at once, rather than looping the distance over centroids and
        # synchronising after each.
        #
        # In slices over the points, though: the eigenvalue problem this poses
        # has n_clusters * n_points matrices in it, and cuSOLVER's batched
        # symmetric solver rejects a batch that large outright — it fails while
        # sizing its workspace, before it has looked at a single value. Slicing
        # changes no result, since every matrix is independent of the others,
        # and it keeps the whitened block small enough to be worth materialising.
        _, inverse_root = sqrtm_invsqrtm_psd(centroids, backend)
        transposed = be.swapaxes(inverse_root, -1, -2)
        n_clusters, n_features = centroids.shape[0], centroids.shape[-1]
        step = max(1, max_batch // n_clusters)

        pieces = []
        for start in range(0, representation.shape[0], step):
            block = representation[start : start + step]
            whitened = inverse_root[:, None] @ block[None] @ transposed[:, None]
            # Through batched_eigh, not linalg.eigvalsh: the wrapper is where
            # this repository keeps its knowledge of what cuSOLVER accepts, and
            # eigvalsh takes a different, less forgiving path through it. The
            # eigenvectors are computed and dropped; that is the price of using
            # the call that works.
            eigenvalues, _ = batched_eigh(
                backend, be.reshape(whitened, (-1, n_features, n_features))
            )
            logarithms = be.log(be.abs(eigenvalues))
            squared = be.reshape(
                be.sum(logarithms * logarithms, axis=-1),
                (n_clusters, block.shape[0]),
            )
            pieces.append(be.swapaxes(squared, -1, -2))
        if len(pieces) == 1:
            return pieces[0]
        return concatenate(backend, pieces, axis=0)

    def update(representation, one_hot, centroids, backend, mean_iterations, mean_tol):
        be = get_backend_module(backend)
        n_features = representation.shape[-1]
        weights = one_hot / be.sum(one_hot, axis=-1, keepdims=True)

        if centroids is None:
            # The descent has no closed form, so it is warm-started: on the
            # arithmetic mean at the first re-estimation of a restart, and on
            # the previous centroids afterwards. Those barely move once the
            # partition settles, which is what lets mean_iterations stay small
            # without under-solving the mean.
            centroids = _unflatten(
                weights @ _flatten(representation, backend), n_features, backend
            )

        steps = 0
        for step in range(mean_iterations):
            root, inverse_root = sqrtm_invsqrtm_psd(centroids, backend)
            # Whiten every point by its own cluster's centroid. one_hot has a
            # single 1 per column, so its transpose gathers the right matrix for
            # each point with a matrix product — there is no scatter or gather
            # primitive common to numpy, torch, cupy and jax, and a product
            # needs none.
            per_point = _unflatten(
                be.swapaxes(one_hot, -1, -2) @ _flatten(inverse_root, backend),
                n_features,
                backend,
            )
            whitened = per_point @ representation @ be.swapaxes(per_point, -1, -2)
            tangent = logm_psd(whitened, backend)
            mean_tangent = _unflatten(
                weights @ _flatten(tangent, backend), n_features, backend
            )
            centroids = root @ _expm_sym(mean_tangent, backend) @ root
            steps = step + 1
            # The tangent mean is the gradient of the Karcher cost: it vanishes
            # exactly when the mean has converged.
            gradient = be.max(be.sum(mean_tangent * mean_tangent, axis=(-2, -1)))
            if to_scalar(gradient) <= mean_tol**2:
                break
        return centroids, steps

    return precompute, assign, update


# Every factory takes the batch bound, even the two that cannot exceed it:
# the flat metrics reduce their pairs with a matrix product, which has no such
# limit, so they ignore it.
_SPD_METRIC_FACTORIES = {
    "euclid": lambda max_batch: _flat_metric(logarithm=False),
    "logeuclid": lambda max_batch: _flat_metric(logarithm=True),
    "riemann": _affine_invariant_metric,
}


def spd_kmeans(
    covariances: Array,
    n_clusters: int,
    metric: str = "riemann",
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-3,
    mean_iterations: int = 10,
    mean_tol: float = 1e-6,
    max_batch: int = 16000,
    seed: int = 42,
    backend: Union[str, Backend] = "numpy",
    verbose: bool = False,
) -> Tuple[np.ndarray, float, list]:
    """K-means on a set of SPD matrices, alternating assignment and re-estimation.

    Parameters
    ----------
    covariances : Array of shape (n_points, n_features, n_features)
        One SPD matrix per point. Unlike :func:`riemannian_kmeans`, none of
        these metrics needs the samples the covariance came from, so they are
        formed once by the caller and never rebuilt inside the loop.
    n_clusters : int
    metric : {"euclid", "logeuclid", "riemann"}
        The geometry in which the centroids are means.
    n_init : int
        Restarts from a random assignment; the one of least inertia is kept.
        Inertia compares restarts of one metric and nothing else — the three
        metrics measure lengths in different geometries.
    max_iter, tol : int, float
        Stop when fewer than ``tol`` of the points change cluster.
    mean_iterations, mean_tol : int, float
        Budget and stopping gradient of one Karcher mean. Ignored by the flat
        metrics, whose mean is closed-form.
    max_batch : int
        Largest number of matrices in one whitened block. Only the
        affine-invariant metric is bounded by it. The default matches the CUDA
        chunk of :func:`~hdrlib.core.backend.batched_eigh`, which is where the
        cuSOLVER batch limit is actually enforced; this bound is about the size
        of the block being materialised. Changing it changes no result.
    seed : int
        Drives the starting partition, identically for every metric.
    backend : str or Backend
        ``jax-*`` is refused: nothing here enables ``x64``, so JAX would compute
        the whole thing in single precision without warning.
    verbose : bool

    Returns
    -------
    labels : ndarray of shape (n_points,)
        On the host, as int64.
    inertia : float
    histories : list of dict
        One entry per restart: its inertia, how many rounds it took, what
        fraction of the points was still changing cluster when it stopped, and
        how many Karcher steps the last re-estimation needed.
    """
    if metric not in _SPD_METRIC_FACTORIES:
        raise KeyError(
            f"unknown metric {metric!r}; known: {sorted(_SPD_METRIC_FACTORIES)}"
        )
    if str(backend).startswith("jax"):
        raise ValueError(
            "the jax backends are refused here: jax defaults to float32 and "
            "nothing in this repository calls "
            "jax.config.update('jax_enable_x64', True), so the eigenvalue "
            "logarithms would be computed in single precision without any "
            "warning. Use torch-cuda, cupy or numpy."
        )

    be = get_backend_module(backend)
    device_covariances = get_data_on_device(covariances, backend)
    require_double(device_covariances, f"the {metric} K-means")

    precompute, assign, update = _SPD_METRIC_FACTORIES[metric](max_batch)
    # Once for the whole run: the representation depends on the covariances,
    # never on the labels.
    representation = precompute(device_covariances, backend)

    n_points = representation.shape[0]
    rng = np.random.default_rng(seed)
    classes = cast_like(
        get_data_on_device(np.arange(n_clusters), backend), representation, backend
    )
    point_index = get_data_on_device(np.arange(n_points), backend)

    best_labels, best_inertia, histories = None, np.inf, []
    for restart in range(n_init):
        labels = _initial_labels(
            rng, n_points, n_clusters, classes, representation, backend
        )
        centroids, moved, mean_steps = None, 1.0, 0

        for iteration in range(max_iter):
            one_hot = _membership(labels, classes, representation, backend)
            centroids, mean_steps = update(
                representation, one_hot, centroids, backend, mean_iterations, mean_tol
            )
            distances = assign(representation, centroids, backend)
            new_labels = _nearest_and_revive(
                distances, classes, point_index, n_clusters, backend
            )
            moved = to_scalar(
                be.sum(cast_like(new_labels != labels, representation, backend))
            ) / n_points
            labels = new_labels
            if moved <= tol:
                break

        # Re-estimate once more, so that the inertia belongs to the labels the
        # loop stopped on rather than to the centroids that produced them.
        one_hot = _membership(labels, classes, representation, backend)
        centroids, _ = update(
            representation, one_hot, centroids, backend, mean_iterations, mean_tol
        )
        distances = assign(representation, centroids, backend)
        inertia = to_scalar(be.sum(distances * be.swapaxes(one_hot, -1, -2)))

        histories.append(
            {
                "inertia": inertia,
                "iterations": iteration + 1,
                "moved": float(moved),
                "mean_steps": int(mean_steps),
            }
        )
        if verbose:
            print(
                f"    {metric} init {restart + 1}/{n_init}: "
                f"inertia {inertia:.3f} in {iteration + 1} iterations, "
                f"{moved:.3%} still moving",
                flush=True,
            )
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia

    return to_numpy(best_labels).astype(np.int64), best_inertia, histories
