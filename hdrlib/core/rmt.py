# Random matrix theory corrections for the affine-invariant geometry.
#
# Port of the reference implementation of
#
#   Bouchard, Mian, Tiomoko, Ginolhac, Pascal, "Random matrix theory improved
#   Fréchet mean of symmetric positive definite matrices", ICML 2024,
#   https://github.com/AmmarMian/icml-rmt-2024
#
# onto hdrlib's backend layer, so the same code runs on numpy, torch, cupy and
# jax. What the port had to change, and what it deliberately did not, is
# documented in 3-learning/frechet_mse/README.md; the numerical agreement with
# the reference is checked by validate_against_paper.py, which imports both.
#
# Two points worth keeping in mind when reading this file.
#
# The reference works in *transformed* coordinates throughout: with L the
# Cholesky factor of the current iterate M, every SCM is carried as
# L^{-1} S L^{-T}, and the cost depends only on the eigenvalues of those. The
# retraction is applied in the same basis, which is what lets the line search
# re-evaluate the cost by a diagonal scaling instead of a fresh Cholesky. That
# structure is kept exactly — it is the reason the algorithm is cheap.
#
# The cost and gradient formulas contain several expressions of the form
# ``mat**3 + eye`` or ``mat**2 - 4*diagL**2``, where ``mat`` is the matrix of
# eigenvalue differences and vanishes on the diagonal. These are not
# regularisations: the ``+ eye`` terms exist so that the diagonal entries
# evaluate the *limit* of the off-diagonal expression rather than 0/0. They are
# reproduced verbatim, and are the part of this file most sensitive to a change
# of backend, since they divide by differences of eigenvalues.

from typing import Optional, Tuple, Union

import numpy as np

from .backend import (
    Array,
    Backend,
    batched_eigh,
    get_backend_module,
    get_data_on_device,
    to_numpy,
)


__all_reexport__ = ("get_backend_module", "get_data_on_device", "to_numpy")


__all__ = [
    "require_double",
    "scm",
    "analytical_shrinkage",
    "ledoit_wolf_linear",
    "oas",
    "rmt_corrected_squared_distance",
    "rmt_frechet_mean",
    "frechet_mean_cholesky",
]


# ── small helpers ─────────────────────────────────────────────────────────────


def _sym(be, x: Array) -> Array:
    return (x + be.swapaxes(x, -1, -2)) / 2


def _eye_like(be, backend, n: int, reference: Array) -> Array:
    """Identity of size ``n``, on the device and dtype of ``reference``."""
    eye = get_data_on_device(np.eye(n), backend)
    return _cast_like(be, eye, reference)


def _ones_like(be, backend, shape, reference: Array) -> Array:
    ones = get_data_on_device(np.ones(shape), backend)
    return _cast_like(be, ones, reference)


def _cast_like(be, x: Array, reference: Array) -> Array:
    """Match dtype of ``reference`` without assuming a backend-specific API."""
    if hasattr(x, "to"):  # torch
        return x.to(reference.dtype)
    if hasattr(x, "astype"):  # numpy, cupy, jax
        return x.astype(reference.dtype)
    return x


def require_double(x: Array, what: str) -> None:
    """Refuse to run the corrected path in single precision.

    The corrected cost and its gradient divide by differences of eigenvalues,
    and several terms are built so that a diagonal entry evaluates a finite
    limit of an expression that is 0/0 elsewhere. In float32 those terms lose
    every significant digit, silently — the descent still returns a matrix, and
    that matrix is wrong.

    The reference implementation has the same requirement without stating it:
    it calls ``dtrtri``, which exists only in double precision.

    The practical consequence is that ``torch-mps`` cannot run this code, since
    Metal has no float64. Use ``torch-cpu`` on Apple silicon.
    """
    dtype = str(getattr(x, "dtype", "unknown"))
    if "64" not in dtype and "double" not in dtype:
        raise TypeError(
            f"{what} needs float64, got {dtype}. The RMT correction is not "
            "numerically meaningful in single precision; on Apple silicon use "
            "--backend torch-cpu rather than torch-mps."
        )


def _cholesky_and_inverse(be, matrix: Array) -> Tuple[Array, Array]:
    """Lower Cholesky factor of ``matrix`` and its inverse.

    The reference calls LAPACK directly here — ``dpptrf`` for a Cholesky in
    packed storage, then ``dtrtri`` for the inverse of the triangular factor.
    Neither has a backend-agnostic equivalent, and neither needs one:

    * packed storage is a memory layout, not a different factorisation, so
      ``cholesky`` returns the same factor in full storage;
    * ``dtrtri`` exploits triangularity to save half the flops, while a general
      ``inv`` does not. The result differs only in rounding.

    ``dtrtri`` is also double-precision only (the ``d`` prefix), so the
    reference silently requires float64. This port does not force the dtype but
    inherits it from the data, which is why the callers below document that the
    RMT path wants float64.
    """
    factor = be.linalg.cholesky(matrix)
    return factor, be.linalg.inv(factor)


# ── covariance estimators used as baselines ───────────────────────────────────


def scm(data: Array, backend: Union[str, Backend] = "numpy") -> Array:
    """Sample covariance matrices of ``(..., n_samples, n_features)`` data.

    Data is assumed centred, as in the reference: the mean is known to be zero
    by construction in the simulations, and subtracting an estimated one would
    change the effective sample size and hence the concentration ratio.
    """
    be = get_backend_module(backend)
    return be.swapaxes(data, -1, -2) @ data / data.shape[-2]


def ledoit_wolf_linear(data: Array, backend: Union[str, Backend] = "numpy") -> Array:
    """Linear shrinkage towards a scaled identity (Ledoit and Wolf, 2004).

    Written out rather than taken from scikit-learn, which is numpy-only and
    would pin the whole experiment to one backend for a two-line formula. This
    is the ``assume_centered=True`` variant, matching the reference's call.
    """
    be = get_backend_module(backend)
    n_samples, n_features = data.shape[-2], data.shape[-1]
    sample = scm(data, backend)
    eye = _eye_like(be, backend, n_features, sample)

    mu = be.einsum("...ii->...", sample) / n_features
    target = be.einsum("...,ij->...ij", mu, eye)
    delta2 = be.sum((sample - target) ** 2, axis=(-2, -1)) / n_features

    # beta2: mean squared deviation of the per-sample outer products from the
    # SCM, which is what the optimal shrinkage intensity trades off.
    outer = be.einsum("...ki,...kj->...kij", data, data)
    beta2 = be.sum((outer - sample[..., None, :, :]) ** 2, axis=(-3, -2, -1))
    beta2 = beta2 / (n_features * n_samples**2)

    beta2 = be.minimum(beta2, delta2)
    shrinkage = beta2 / delta2
    return (
        be.einsum("...,...ij->...ij", shrinkage, target)
        + be.einsum("...,...ij->...ij", 1.0 - shrinkage, sample)
    )


def oas(data: Array, backend: Union[str, Backend] = "numpy") -> Array:
    """Oracle approximating shrinkage (Chen et al., 2010), centred variant.

    Follows scikit-learn's formulation rather than the one printed in the
    original paper: the two differ by ``(1 - 2/p)`` factors that scikit-learn
    drops, and it is scikit-learn that the reference implementation calls, so
    it is scikit-learn that the figures of the paper actually plot.
    """
    be = get_backend_module(backend)
    n_samples, n_features = data.shape[-2], data.shape[-1]
    sample = scm(data, backend)
    eye = _eye_like(be, backend, n_features, sample)

    mu = be.einsum("...ii->...", sample) / n_features
    target = be.einsum("...,ij->...ij", mu, eye)

    alpha = be.mean(sample * sample, axis=(-2, -1))
    numerator = alpha + mu**2
    denominator = (n_samples + 1.0) * (alpha - mu**2 / n_features)
    ratio = numerator / denominator
    shrinkage = be.minimum(ratio, be.ones_like(ratio))
    return (
        be.einsum("...,...ij->...ij", shrinkage, target)
        + be.einsum("...,...ij->...ij", 1.0 - shrinkage, sample)
    )


def analytical_shrinkage(
    data: Array, backend: Union[str, Backend] = "numpy", shrink: Optional[int] = None
) -> Array:
    """Analytical non-linear shrinkage (Ledoit and Wolf, 2020).

    Kernel estimate of the limiting spectral density and its Hilbert transform,
    used to shrink each sample eigenvalue individually. Transposed from the
    reference, with one deliberate change: the eigendecomposition uses ``eigh``
    rather than ``eig``. The matrix is symmetric by construction, ``eig``
    returns complex values for it on some backends, and the reference then
    sorts them — ``eigh`` returns them real and already ascending.

    Valid for ``n_features <= n_samples``, like the reference.
    """
    be = get_backend_module(backend)
    n_samples, n_features = data.shape[-2], data.shape[-1]

    if shrink is None:
        data = data - be.mean(data, axis=-2, keepdims=True)
        shrink = 1
    effective = n_samples - shrink

    sample = be.swapaxes(data, -1, -2) @ data / effective
    eigenvalues, eigenvectors = batched_eigh(backend, sample)

    size = min(n_features, effective)
    repmat = be.swapaxes(
        be.broadcast_to(eigenvalues[..., None, :], eigenvalues.shape[:-1] + (size, n_features)),
        -1,
        -2,
    )
    bandwidth = effective ** (-1 / 3) * be.swapaxes(repmat, -1, -2)
    ratio = (repmat - be.swapaxes(repmat, -1, -2)) / bandwidth

    zero = be.zeros_like(ratio)
    density = (3 / 4 / np.sqrt(5)) * be.mean(
        be.maximum(1 - ratio**2 / 5, zero) / bandwidth, axis=-1
    )

    root5 = np.sqrt(5.0)
    hilbert_kernel = (-3 / 10 / np.pi) * ratio + (
        3 / 4 / root5 / np.pi
    ) * (1 - ratio**2 / 5) * be.log(be.abs((root5 - ratio) / (root5 + ratio)))
    # The logarithm is singular exactly at |ratio| = sqrt(5); the kernel has a
    # finite limit there, which the reference substitutes by hand.
    singular = be.abs(ratio) == root5
    hilbert_kernel = be.where(singular, (-3 / 10 / np.pi) * ratio, hilbert_kernel)
    hilbert = be.mean(hilbert_kernel / bandwidth, axis=-1)

    concentration = n_features / effective
    shrunk = eigenvalues / (
        (np.pi * concentration * eigenvalues * density) ** 2
        + (1 - concentration - np.pi * concentration * eigenvalues * hilbert) ** 2
    )
    return be.einsum(
        "...ij,...j,...kj->...ik", eigenvectors, shrunk, eigenvectors
    )


# ── the corrected cost and its gradient ───────────────────────────────────────


def _rmt_cost_grad(
    be,
    backend,
    transformed: Array,
    n_features: int,
    n_samples: int,
    concentration: float,
    return_grad: bool = True,
):
    """Corrected Fréchet cost, and its gradient in canonical form.

    ``transformed`` holds ``L^{-1} S L^{-T}`` for every SCM ``S``, with ``L``
    the Cholesky factor of the current iterate. The returned gradient is in the
    same basis; a congruence with ``L`` turns it into the Riemannian gradient at
    the iterate, which the caller does implicitly through the retraction.

    This is a transposition of ``_aux_RMT_mean_cost_grad`` of the reference,
    term for term. Nothing is simplified: the expressions look redundant in
    places (``mat**3 + multi_eye``, ``mat**2 - 4*diagL**2``) but each of those
    additions is what makes a diagonal entry evaluate a finite limit instead of
    0/0, and rewriting them is the fastest way to break the gradient silently.
    """
    eye = _eye_like(be, backend, n_features, transformed)
    multi_eye = be.broadcast_to(eye, transformed.shape)
    one_vec = _ones_like(be, backend, (n_features,), transformed)
    ones_mat = _ones_like(be, backend, (n_features, n_features), transformed)

    eigenvalues, eigenvectors = batched_eigh(backend, transformed)
    root = be.sqrt(eigenvalues)
    inverse = 1 / eigenvalues
    logarithm = be.log(eigenvalues)

    diag_l = be.einsum("...i,...ij->...ij", eigenvalues, multi_eye)
    shifted, shifted_vectors = batched_eigh(
        backend,
        diag_l - be.einsum("...i,...j->...ij", root, root) / n_samples,
    )

    differences = be.einsum("...i,j->...ij", eigenvalues, one_vec)
    differences = differences - be.swapaxes(differences, -1, -2)
    correction = (
        be.einsum(
            "...i,...ij->...ij",
            eigenvalues,
            be.log(be.einsum("...i,...j->...ij", eigenvalues, inverse)),
        )
        - differences
        + multi_eye / 2
    ) / (differences**2 + diag_l)
    q = inverse * logarithm

    costs = (
        be.einsum("...i,...i->...", logarithm, logarithm) / (2 * n_features)
        + be.mean(logarithm, axis=-1)
        - be.einsum("...i,...ij->...", eigenvalues - shifted, correction) / n_features
        - (1 / concentration - 1) * (np.log(1 - concentration) ** 2) / 2
        - (1 / concentration - 1)
        * be.einsum("...i,...i->...", eigenvalues - shifted, q)
    )
    cost = be.mean(costs)

    if not return_grad:
        return cost

    delta = correction @ one_vec / n_features + (1 - concentration) / concentration * q
    delta_ = be.einsum(
        "...ik,...ki->...i",
        multi_eye
        - be.einsum("...i,...j->...ij", 1 / root, root) / n_samples,
        be.einsum(
            "...ik,...k,...jk->...ij", shifted_vectors, delta, shifted_vectors
        ),
    )
    a_mat = -(
        be.einsum("...i,j->...ij", logarithm, one_vec)
        - be.einsum("i,...j->...ij", one_vec, logarithm)
    ) * (
        be.einsum("...i,j->...ij", eigenvalues, one_vec)
        + be.einsum("i,...j->...ij", one_vec, eigenvalues)
    ) / (
        differences**3 + multi_eye
    ) - 2 / (
        2 * diag_l**2 - differences**2
    )
    b_mat = (
        ((multi_eye - ones_mat) * be.einsum("i,...j->...ij", one_vec, inverse))
        / (differences + multi_eye)
        + 2
        * (
            be.einsum("...i,j->...ij", eigenvalues * logarithm, one_vec)
            - be.einsum("...i,...j->...ij", eigenvalues, logarithm)
        )
        / (differences**3 + multi_eye)
        - 2 * ones_mat / (differences**2 - 4 * diag_l**2)
    )
    diag_correction = (
        be.einsum("...ik,...i->...i", a_mat, eigenvalues - shifted)
        + be.einsum("...k,...ki->...i", eigenvalues - shifted, b_mat)
    ) / n_features

    grad_eigenvalues = (
        (logarithm + one_vec) * inverse / n_features
        - delta
        + delta_
        - diag_correction
        - (1 - concentration)
        / concentration
        * (1 - logarithm)
        * (eigenvalues - shifted)
        * inverse**2
    )
    grad_canonical = -be.mean(
        be.einsum(
            "...ik,...k,...jk->...ij",
            eigenvectors,
            eigenvalues * grad_eigenvalues,
            eigenvectors,
        ),
        axis=0,
    )
    return cost, grad_canonical


def _plain_cost_grad(be, backend, transformed: Array, n_features: int, return_grad=True):
    """Uncorrected Fréchet cost in the same transformed coordinates."""
    eigenvalues, eigenvectors = batched_eigh(backend, transformed)
    logarithm = be.log(eigenvalues)
    cost = be.mean(
        be.einsum("...i,...i->...", logarithm, logarithm) / (2 * n_features)
    )
    if not return_grad:
        return cost
    grad_canonical = -be.mean(
        be.einsum(
            "...ik,...k,...jk->...ij",
            eigenvectors,
            logarithm / n_features,
            eigenvectors,
        ),
        axis=0,
    )
    return cost, grad_canonical


def _linesearch(
    be,
    directions: Array,
    rotated: Array,
    cost_fn,
    cost: float,
    old_cost: Optional[float] = None,
    tol_cost: float = 10.0,
) -> float:
    """Backtracking line search along the second-order retraction.

    The step is evaluated without a new Cholesky: in the basis that
    diagonalises the descent direction, the retraction acts on the transformed
    matrices as a diagonal congruence, so a candidate cost costs one scaling
    and one eigendecomposition.
    """
    alpha0 = 1.0
    optimism = 2.0
    sufficient_decrease = 1e-4
    contraction = 0.5
    max_iterations = 25

    grad_norm_sq = float(to_numpy(be.sum(directions**2)))
    if old_cost is None:
        alpha = alpha0 / grad_norm_sq**0.5
    else:
        alpha = -2 * (float(cost) - float(old_cost)) / grad_norm_sq * optimism

    def candidate(step):
        scaling = (1 + step * directions + step**2 * directions**2 / 2) ** (-0.5)
        return be.einsum("i,...ij,j->...ij", scaling, rotated, scaling)

    new_cost = float(to_numpy(cost_fn(candidate(alpha))))
    iterations = 0
    n_features = rotated.shape[-1]
    while (
        new_cost > float(cost) - sufficient_decrease * alpha * grad_norm_sq
        and iterations < max_iterations
        and new_cost > -tol_cost / n_features
    ):
        alpha *= contraction
        new_cost = float(to_numpy(cost_fn(candidate(alpha))))
        iterations += 1

    # A step that increases the cost is rejected outright rather than shrunk
    # further: the retraction is only second-order accurate and a tiny step
    # buys nothing once the line search has failed.
    return 0.0 if new_cost > float(cost) else alpha


# ── the two means ─────────────────────────────────────────────────────────────


def _descent(
    be,
    backend,
    matrices: Array,
    cost_grad,
    init: Optional[Array],
    max_iterations: int,
    tol: float,
    tol_cost: Optional[float],
):
    """Shared Riemannian descent of the two means.

    Both means minimise a cost that depends on the current iterate only through
    ``L^{-1} S L^{-T}``, so both run the same loop: Cholesky, transform, cost
    and gradient, eigendecomposition of the descent direction, line search,
    retraction in the rotated basis.
    """
    n_features = matrices.shape[-1]
    one_vec = _ones_like(be, backend, (n_features,), matrices)

    mean = (
        _eye_like(be, backend, n_features, matrices) if init is None else init
    )
    history = {"cost": [], "error": []}
    error = np.inf
    cost = np.inf
    old_cost = None

    for _ in range(max_iterations):
        if tol_cost is not None and cost <= -tol_cost / n_features:
            break
        if error <= tol:
            break

        factor, inverse_factor = _cholesky_and_inverse(be, mean)
        transformed = inverse_factor @ matrices @ be.swapaxes(inverse_factor, -1, -2)

        cost, grad_canonical = cost_grad(transformed, return_grad=True)
        cost = float(to_numpy(cost))
        direction = -_sym(be, grad_canonical)
        directions, rotation = batched_eigh(backend, direction)
        rotated = be.swapaxes(rotation, -1, -2) @ transformed @ rotation

        alpha = _linesearch(
            be,
            directions,
            rotated,
            lambda mats: cost_grad(mats, return_grad=False),
            cost,
            old_cost,
            tol_cost=tol_cost if tol_cost is not None else np.inf,
        )

        step = alpha * directions
        tmp = factor @ rotation
        new_mean = _sym(
            be,
            be.einsum(
                "ik,k,kj->ij",
                tmp,
                one_vec + step + step**2 / 2,
                be.swapaxes(tmp, -1, -2),
            ),
        )

        error = float(
            to_numpy(be.linalg.norm(mean - new_mean) / be.linalg.norm(mean))
        )
        mean = new_mean
        old_cost = cost
        history["cost"].append(cost)
        history["error"].append(error)

    return mean, history


def rmt_frechet_mean(
    data: Array,
    n_dof: Optional[int] = None,
    init: Optional[Array] = None,
    max_iterations: int = 100,
    tol: float = 1e-6,
    tol_cost: float = 10.0,
    backend: Union[str, Backend] = "numpy",
) -> Tuple[Array, dict]:
    """Fréchet mean of the *true* covariances behind a set of data matrices.

    Minimises the random-matrix-theory corrected Fréchet cost over the cone,
    starting from the identity. The correction makes the estimate consistent in
    the regime where the dimension and the sample size grow proportionally,
    which is exactly where the plain Fréchet mean of the SCMs is not.

    Wants float64: the gradient divides by differences of eigenvalues, and the
    reference implementation calls a double-precision-only LAPACK routine, so
    single precision is outside what has ever been validated.

    Parameters
    ----------
    data : Array of shape (n_matrices, n_samples, n_features)
    n_dof : int, optional
        Degrees of freedom behind each SCM; ``n_samples`` by default.
    init : Array, optional
        Starting point; the identity by default.
    max_iterations, tol, tol_cost : see the reference.
    backend : str or Backend

    Returns
    -------
    mean : Array of shape (n_features, n_features)
    history : dict with lists ``cost`` and ``error``
    """
    be = get_backend_module(backend)
    require_double(data, "rmt_frechet_mean")
    n_samples, n_features = data.shape[-2], data.shape[-1]
    if n_dof is None:
        n_dof = n_samples
    concentration = n_features / n_dof

    covariances = scm(data, backend)

    def cost_grad(transformed, return_grad=True):
        return _rmt_cost_grad(
            be, backend, transformed, n_features, n_dof, concentration,
            return_grad=return_grad,
        )

    return _descent(
        be, backend, covariances, cost_grad, init, max_iterations, tol, tol_cost
    )


def frechet_mean_cholesky(
    covariances: Array,
    init: Optional[Array] = None,
    max_iterations: int = 100,
    tol: float = 1e-3,
    backend: Union[str, Backend] = "numpy",
) -> Tuple[Array, dict]:
    """Plain Fréchet mean, by the same descent as :func:`rmt_frechet_mean`.

    ``hdrlib.core.estimation.frechet_mean_affine_invariant`` computes the same
    object by a different route (log-Euclidean start, fixed-step exponential
    updates). This one exists so that the corrected and uncorrected means of a
    comparison differ *only* by the correction, and not also by the optimiser —
    which is what the MSE figures of the ICML paper measure.
    """
    be = get_backend_module(backend)
    n_features = covariances.shape[-1]

    def cost_grad(transformed, return_grad=True):
        return _plain_cost_grad(
            be, backend, transformed, n_features, return_grad=return_grad
        )

    return _descent(
        be, backend, covariances, cost_grad, init, max_iterations, tol, None
    )


def rmt_corrected_squared_distance(
    reference: Array,
    covariances: Array,
    n_samples: int,
    backend: Union[str, Backend] = "numpy",
) -> Array:
    """Corrected squared Fisher distance between a fixed SPD matrix and SCMs.

    Estimates ``delta^2(reference, Sigma_k)`` from ``reference`` and the sample
    covariances of the ``Sigma_k``, divided by the dimension. Consistent in the
    regime of :func:`rmt_frechet_mean`, where the plain distance is not.
    """
    be = get_backend_module(backend)
    require_double(covariances, "rmt_corrected_squared_distance")
    n_features = covariances.shape[-1]
    concentration = n_features / n_samples

    _, inverse_factor = _cholesky_and_inverse(be, reference)
    transformed = inverse_factor @ covariances @ be.swapaxes(inverse_factor, -1, -2)

    eye = _eye_like(be, backend, n_features, transformed)
    multi_eye = be.broadcast_to(eye, transformed.shape)
    one_vec = _ones_like(be, backend, (n_features,), transformed)

    eigenvalues = batched_eigh(backend, transformed)[0]
    root = be.sqrt(eigenvalues)
    inverse = 1 / eigenvalues
    logarithm = be.log(eigenvalues)

    diag_l = be.einsum("...i,...ij->...ij", eigenvalues, multi_eye)
    shifted = batched_eigh(
        backend,
        diag_l - be.einsum("...i,...j->...ij", root, root) / n_samples,
    )[0]

    differences = be.einsum("...i,j->...ij", eigenvalues, one_vec)
    differences = differences - be.swapaxes(differences, -1, -2)
    correction = (
        be.einsum(
            "...i,...ij->...ij",
            eigenvalues,
            be.log(be.einsum("...i,...j->...ij", eigenvalues, inverse)),
        )
        - differences
        + multi_eye / 2
    ) / (differences**2 + diag_l)

    return (
        be.einsum("...i,...i->...", logarithm, logarithm) / n_features
        + 2 * be.mean(logarithm, axis=-1)
        - 2 / n_features * be.einsum("...i,...ij->...", eigenvalues - shifted, correction)
        - (1 / concentration - 1) * np.log(1 - concentration) ** 2
        - 2
        * (1 / concentration - 1)
        * be.einsum("...i,...i->...", eigenvalues - shifted, inverse * logarithm)
    )
