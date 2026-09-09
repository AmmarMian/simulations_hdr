# Random matrix theory corrections for the affine-invariant geometry.
#
# Implements the corrected Fisher distance and the corrected Fréchet mean of
#
#   F. Bouchard, A. Mian, M. Tiomoko, G. Ginolhac and F. Pascal, "Random
#   matrix theory improved Fréchet mean of symmetric positive definite
#   matrices", Proceedings of the 41st International Conference on Machine
#   Learning, PMLR 235:4403-4415, 2024. arXiv:2405.06558
#
# on hdrlib's backend layer, so the same code runs on numpy, torch, cupy and
# jax. Needs float64 — see require_double.
#
# Two points worth knowing before editing this file.
#
# Everything works in *transformed* coordinates: with L the Cholesky factor of
# the current iterate M, every SCM is carried as L^{-1} S L^{-T}, and the cost
# depends only on the eigenvalues of those. The retraction is applied in the
# same basis, which is what lets the line search re-evaluate the cost by a
# diagonal scaling instead of a fresh Cholesky. That structure is the reason
# the algorithm is cheap, and it should be kept.
#
# The cost and gradient contain several expressions of the form ``mat**3 + eye``
# or ``mat**2 - 4*diagL**2``, where ``mat`` is the matrix of eigenvalue
# differences and vanishes on the diagonal. These are not regularisations: the
# ``+ eye`` terms exist so that the diagonal entries evaluate the *limit* of the
# off-diagonal expression rather than 0/0. Rewriting them is the fastest way to
# break the gradient silently.

from typing import Optional, Tuple, Union

import numpy as np

from ..core.backend import (
    Array,
    Backend,
    batched_eigh,
    concatenate,
    get_backend_module,
    get_data_on_device,
    require_double as _require_double,
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


# Why this module in particular cannot be run in single precision. The corrected
# cost and its gradient divide by differences of eigenvalues, and several terms
# are built so that a diagonal entry evaluates a finite limit of an expression
# that is 0/0 elsewhere; in float32 those terms lose every significant digit
# silently — the descent still returns a matrix, and that matrix is wrong.
_PRECISION_REASON = (
    "The RMT correction is not numerically meaningful in single precision: its "
    "gradient divides by differences of eigenvalues."
)


def require_double(x: Array, what: str) -> None:
    """Refuse to run the corrected path in single precision.

    Thin wrapper over :func:`hdrlib.core.backend.require_double`, which owns the
    check; this one only supplies the reason specific to the correction. Kept as
    a name in this module because callers outside it import it from here.
    """
    _require_double(x, what, _PRECISION_REASON)


def _logm(be, backend, matrices: Array) -> Array:
    """Matrix logarithm of a batch of SPD matrices."""
    eigenvalues, eigenvectors = batched_eigh(backend, matrices)
    return be.einsum(
        "...ij,...j,...kj->...ik", eigenvectors, be.log(eigenvalues), eigenvectors
    )


def _expm(be, backend, matrix: Array) -> Array:
    """Matrix exponential of one symmetric matrix."""
    eigenvalues, eigenvectors = batched_eigh(backend, _sym(be, matrix))
    return _sym(
        be,
        be.einsum(
            "ij,j,kj->ik", eigenvectors, be.exp(eigenvalues), eigenvectors
        ),
    )


def _cholesky_and_inverse(be, matrix: Array) -> Tuple[Array, Array]:
    """Lower Cholesky factor of ``matrix`` and its inverse."""
    factor = be.linalg.cholesky(matrix)
    return factor, be.linalg.inv(factor)


# ── covariance estimators used as baselines ───────────────────────────────────


def scm(data: Array, backend: Union[str, Backend] = "numpy") -> Array:
    r"""Sample covariance matrices of ``(..., n_samples, n_features)`` data.

    For $N$ samples $\boldsymbol{x}_1,\dots,\boldsymbol{x}_N \in \mathbb{R}^d$
    stacked as the rows of $\boldsymbol{X}$,

    $$
    \widehat{\boldsymbol{\Sigma}}
      = \frac{1}{N} \boldsymbol{X}^{\mathrm{T}} \boldsymbol{X}
      = \frac{1}{N} \sum_{k=1}^{N}
        \boldsymbol{x}_k \boldsymbol{x}_k^{\mathrm{T}}.
    $$

    Data is assumed centred, as in the reference implementation: the mean is
    known to be zero by construction in the simulations, and subtracting an
    estimated one would change the effective sample size and hence the
    concentration ratio $c = d/N$.
    """
    be = get_backend_module(backend)
    return be.swapaxes(data, -1, -2) @ data / data.shape[-2]


def ledoit_wolf_linear(data: Array, backend: Union[str, Backend] = "numpy") -> Array:
    r"""Linear shrinkage of the sample covariance towards a scaled identity.

    The estimator pulls $\widehat{\boldsymbol{\Sigma}}$ towards the target
    $\boldsymbol{T} = \mu \boldsymbol{I}_d$ of the same trace,

    $$
    \widehat{\boldsymbol{\Sigma}}_{\mathrm{lw}}
      = \alpha\, \boldsymbol{T} + (1 - \alpha)\,
        \widehat{\boldsymbol{\Sigma}},
    \qquad
    \mu = \frac{1}{d} \operatorname{Tr}\!\left\{
          \widehat{\boldsymbol{\Sigma}} \right\},
    $$

    by the intensity that minimises the expected squared Frobenius error,
    estimated from the data as

    $$
    \alpha = \min\!\left(\frac{b^2}{a^2},\, 1\right),
    \qquad
    a^2 = \frac{1}{d}\left\| \widehat{\boldsymbol{\Sigma}} - \boldsymbol{T}
          \right\|_{\mathbb{F}}^2,
    \qquad
    b^2 = \frac{1}{d\,N^2} \sum_{k=1}^{N}
          \left\| \boldsymbol{x}_k \boldsymbol{x}_k^{\mathrm{T}}
                  - \widehat{\boldsymbol{\Sigma}}
          \right\|_{\mathbb{F}}^2 .
    $$

    Centred variant: the data of these simulations is known to have zero mean
    by construction.

    **Reference**

    O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional
    covariance matrices", *Journal of Multivariate Analysis*, 88(2):365-411,
    2004. [doi:10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)
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
    r"""Oracle approximating shrinkage, centred variant.

    Same shrinkage towards a scaled identity as
    :func:`ledoit_wolf_linear`, with an intensity derived instead by
    iterating the oracle solution under a Gaussian assumption:

    $$
    \widehat{\boldsymbol{\Sigma}}_{\mathrm{oas}}
      = \alpha\, \mu \boldsymbol{I}_d + (1 - \alpha)\,
        \widehat{\boldsymbol{\Sigma}},
    \qquad
    \alpha = \min\!\left(
        \frac{s + \mu^2}{(N + 1)\left(s - \mu^2 / d\right)},
        \, 1 \right),
    $$

    where $\mu = \frac{1}{d}\operatorname{Tr}\!\left\{
    \widehat{\boldsymbol{\Sigma}}\right\}$ and
    $s = \frac{1}{d^2}\operatorname{Tr}\!\left\{
    \widehat{\boldsymbol{\Sigma}}^2\right\}$.

    Note that two formulations circulate: the one printed in the original
    paper, and the one in common use, which drops its $(1 - 2/d)$ factors.
    The second is implemented here, since it is the one every published
    comparison actually plots.

    **Reference**

    Y. Chen, A. Wiesel, Y. C. Eldar and A. O. Hero, "Shrinkage algorithms for
    MMSE covariance estimation", *IEEE Transactions on Signal Processing*,
    58(10):5016-5029, 2010.
    [doi:10.1109/TSP.2010.2053029](https://doi.org/10.1109/TSP.2010.2053029)
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


# Where the direct form of the Epanechnikov Hilbert transform is abandoned for
# the series, and how many terms the series then needs. At the switch
# |v| = sqrt(5)/20 < 0.112, so twelve terms put the truncation below 1e-17 —
# well under the 1e-13 the direct form still holds there. See the comment in
# analytical_shrinkage for what the two forms are and why one is not enough.
_HILBERT_SWITCH = 20.0
_HILBERT_TERMS = 12


def _hilbert_kernel(ratio: Array, backend: Union[str, Backend] = "numpy") -> Array:
    """Hilbert transform of the Epanechnikov kernel, evaluated stably.

    Written the way the reference writes it,

        -3/(10 pi) r + 3/(4 sqrt5 pi) (1 - r^2/5) log|(sqrt5 - r)/(sqrt5 + r)|,

    this is a difference of two terms that cancel. At |r| = 1.8e6 each term is
    1.7e5 and their difference is 2e-6: eleven of the sixteen digits are gone,
    and what comes back is twelve times too large. At 1e7 it is four orders of
    magnitude too large. That is not a loss of precision, it is the wrong
    number — and it is wrong *differently* on every backend, which is how it
    was found: LW-NL disagreed between numpy and CUDA while SCM, LW and RMT
    agreed to 1e-16.

    The regime is not exotic. On Salinas with 5x5 windows, 31% of the
    off-diagonal ratios exceed 100 and 1% exceed 1e6, so every window loses
    digits here and a sixth of them hold an entry with none left.

    Substituting ``v = sqrt(5) / r`` removes the cancellation outright:

        kernel = -(3 / (sqrt5 pi)) * sum_k v^(2k+1) / ((2k+1)(2k+3)),

    every term of which has the same sign. That form is used above the switch,
    where it converges in a dozen terms; below it the two terms of the direct
    form are of different sizes and the direct form is exact. The two agree to
    machine precision on either side.
    """
    be = get_backend_module(backend)
    root5 = np.sqrt(5.0)

    direct = (-3 / 10 / np.pi) * ratio + (
        3 / 4 / root5 / np.pi
    ) * (1 - ratio**2 / 5) * be.log(be.abs((root5 - ratio) / (root5 + ratio)))
    # The logarithm is singular exactly at |ratio| = sqrt(5); the kernel has a
    # finite limit there, which the reference substitutes by hand.
    singular = be.abs(ratio) == root5
    direct = be.where(singular, (-3 / 10 / np.pi) * ratio, direct)

    # Both branches are evaluated everywhere and then selected, rather than
    # gathered: the backends share no scatter primitive, and each is a handful
    # of elementwise passes. The placeholder keeps the reciprocal finite where
    # the series is discarded, including on the diagonal where ratio is zero.
    large = be.abs(ratio) > _HILBERT_SWITCH
    v = root5 / be.where(large, ratio, be.ones_like(ratio))
    v_squared = v * v
    series = be.zeros_like(v)
    for term in range(_HILBERT_TERMS - 1, -1, -1):
        series = series * v_squared + 1.0 / ((2 * term + 1) * (2 * term + 3))
    series = (-3 / root5 / np.pi) * v * series

    return be.where(large, series, direct)


def analytical_shrinkage(
    data: Array, backend: Union[str, Backend] = "numpy", shrink: Optional[int] = None
) -> Array:
    r"""Analytical non-linear shrinkage of the sample eigenvalues.

    The eigenvectors of $\widehat{\boldsymbol{\Sigma}}$ are kept and each
    eigenvalue $\lambda_i$ is shrunk on its own, by the amount random matrix
    theory says it is inflated by:

    $$
    \widetilde{\lambda}_i = \frac{\lambda_i}
    {\left[\pi c\, \lambda_i\, \widehat{f}(\lambda_i)\right]^2
     + \left[1 - c - \pi c\, \lambda_i\,
       \mathsf{H}\widehat{f}(\lambda_i)\right]^2},
    \qquad c = \frac{d}{N},
    $$

    where $\widehat{f}$ is an Epanechnikov kernel estimate of the limiting
    spectral density, with the variable bandwidth $h_i = N^{-1/3}\lambda_i$,
    and $\mathsf{H}\widehat{f}$ its Hilbert transform. The returned matrix is
    $\boldsymbol{U} \operatorname{diag}(\widetilde{\boldsymbol{\lambda}})
    \boldsymbol{U}^{\mathrm{T}}$.

    The matrix is symmetric by construction, so the decomposition is ``eigh``:
    real eigenvalues, already ascending.

    Valid for ``n_features <= n_samples``.

    **Reference**

    O. Ledoit and M. Wolf, "Analytical nonlinear shrinkage of large-dimensional
    covariance matrices", *The Annals of Statistics*, 48(5):3043-3065, 2020.
    [doi:10.1214/19-AOS1921](https://doi.org/10.1214/19-AOS1921)
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

    hilbert_kernel = _hilbert_kernel(ratio, backend)
    hilbert = be.mean(hilbert_kernel / bandwidth, axis=-1)

    concentration = n_features / effective
    shrunk = eigenvalues / (
        (np.pi * concentration * eigenvalues * density) ** 2
        + (1 - concentration - np.pi * concentration * eigenvalues * hilbert) ** 2
    )
    return be.einsum(
        "...ij,...j,...kj->...ik", eigenvectors, shrunk, eigenvectors
    )


# Minimum spacing imposed on the eigenvalues before the corrected gradient is
# formed, relative to the largest of them. The gradient contains several
# expressions that are 0/0 between two *equal* eigenvalues -- individually
# singular, though their sum is finite, because the cost is an analytic
# symmetric function of the spectrum. Rather than take that limit in closed
# form, the spectrum is separated by a hair and the exact gradient of the
# separated spectrum is returned.
#
# The value is cbrt(machine epsilon), the usual balance for a difference that
# cancels to second order, and it is measured rather than assumed. On a 5x5
# whitened SCM at c = 1/8, moving two eigenvalues apart by a relative eps:
#
#     eps      1e-4     1e-5     1e-6     1e-7     1e-8     1e-9     1e-12
#     grad  0.169777 0.170380 0.170435 0.170131 0.183957 2.91e+00 6.65e-02
#
# converged to six digits at 1e-6 and destroyed by cancellation below 1e-8.
_EIGENVALUE_FLOOR = float(np.cbrt(np.finfo(np.float64).eps))


def _separate_eigenvalues(
    be, backend, eigenvalues: Array, n_features: int
) -> Array:
    """Push apart eigenvalues that coincide, leaving separated ones alone.

    ``batched_eigh`` returns them ascending and positive, so the largest is the
    last and imposing a minimum gap is a running maximum: each eigenvalue is
    raised to at least its predecessor plus ``_EIGENVALUE_FLOOR`` times the
    scale of the spectrum. A spectrum already separated by more than that comes
    back bit-for-bit unchanged, which is what keeps the ordinary case -- and
    every result computed so far -- exactly as it was.

    The loop runs over the dimension (5 to 16 here), not over the matrices: one
    pass can only push a gap one position along, and a whole spectrum could in
    principle be constant.
    """
    scale = eigenvalues[..., -1:]
    gap = _EIGENVALUE_FLOOR * scale
    separated = eigenvalues
    for _ in range(n_features - 1):
        raised = be.maximum(separated[..., 1:], separated[..., :-1] + gap)
        separated = concatenate(
            backend, [separated[..., :1], raised], axis=-1
        )
    return separated


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

    Nothing here is simplified: the expressions look redundant in places
    (``mat**3 + multi_eye``, ``mat**2 - 4*diagL**2``) but each of those
    additions is what makes a diagonal entry evaluate a finite limit instead of
    0/0. Those additions handle the *diagonal* only; two distinct positions
    carrying the same eigenvalue hit the same 0/0 off the diagonal, which is
    what :func:`_separate_eigenvalues` is for.
    """
    eye = _eye_like(be, backend, n_features, transformed)
    multi_eye = be.broadcast_to(eye, transformed.shape)
    one_vec = _ones_like(be, backend, (n_features,), transformed)
    ones_mat = _ones_like(be, backend, (n_features, n_features), transformed)

    eigenvalues, eigenvectors = batched_eigh(backend, transformed)
    # Coincident eigenvalues are a pole of several terms below. See
    # _separate_eigenvalues: without this a single-matrix cluster, whose
    # whitened SCM is the identity, returns a NaN gradient.
    eigenvalues = _separate_eigenvalues(be, backend, eigenvalues, n_features)
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

    if init is None:
        # The identity, as in the reference implementation. Both descents there
        # start from it, and because the corrected cost is flat near its
        # optimum the starting point decides where the iteration settles: with
        # the log-Euclidean mean instead, the corrected mean lands 4e-2 away
        # from the reference's, against 7e-6 from here.
        #
        # The log-Euclidean mean is the better start on badly scaled data — it
        # carries the scale, where the identity leaves the first gradient
        # enormous on matrices whose eigenvalues reach 1e8, as raw radiance
        # does. Pass it explicitly as ``init`` when that is the situation;
        # every experiment here normalises the scene first.
        mean = _eye_like(be, backend, n_features, matrices)
    else:
        mean = init
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
    r"""Fréchet mean of the *true* covariances behind a set of data matrices.

    Given $K$ data matrices drawn from unknown covariances
    $\boldsymbol{\Sigma}_1, \dots, \boldsymbol{\Sigma}_K$, the quantity of
    interest is the affine-invariant barycentre of those covariances,

    $$
    \overline{\boldsymbol{\Sigma}}
      = \operatorname*{arg\,min}_{\boldsymbol{\Sigma} \in \mathcal{S}_d^{++}}
        \frac{1}{K} \sum_{k=1}^{K}
        \delta^2\!\left(\boldsymbol{\Sigma}, \boldsymbol{\Sigma}_k\right),
    $$

    which cannot be evaluated, since only the sample covariances
    $\widehat{\boldsymbol{\Sigma}}_k$ are observed. Substituting them —
    which is what :func:`frechet_mean_cholesky` minimises — is biased as soon
    as the concentration ratio $c = d/N$ is not small. This function minimises
    instead

    $$
    \overline{\boldsymbol{\Sigma}}_{\mathrm{rmt}}
      = \operatorname*{arg\,min}_{\boldsymbol{\Sigma} \in \mathcal{S}_d^{++}}
        \frac{1}{K} \sum_{k=1}^{K}
        \widehat{\delta}^2\!\left(\boldsymbol{\Sigma},
                                  \boldsymbol{\Sigma}_k\right),
    $$

    with $\widehat{\delta}$ the consistent estimator of
    :func:`rmt_corrected_squared_distance`. The minimisation is a Riemannian
    steepest descent with backtracking, started from the identity and carried
    out in the coordinates of the Cholesky factor of the current iterate.

    Wants float64: the gradient divides by differences of eigenvalues, which
    single precision cannot carry — see :func:`require_double`.

    **Reference**

    F. Bouchard, A. Mian, M. Tiomoko, G. Ginolhac and F. Pascal, "Random matrix
    theory improved Fréchet mean of symmetric positive definite matrices",
    *Proceedings of the 41st International Conference on Machine Learning*,
    PMLR 235:4403-4415, 2024.
    [arXiv:2405.06558](https://arxiv.org/abs/2405.06558)

    Parameters
    ----------
    data : Array of shape (n_matrices, n_samples, n_features)
    n_dof : int, optional
        Degrees of freedom behind each SCM; ``n_samples`` by default.
    init : Array, optional
        Starting point; the identity by default.
    max_iterations, tol, tol_cost : int, float, float
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
    r"""Plain Fréchet mean, by the same descent as :func:`rmt_frechet_mean`.

    The uncorrected barycentre of the matrices it is given,

    $$
    \overline{\boldsymbol{\Sigma}}
      = \operatorname*{arg\,min}_{\boldsymbol{\Sigma} \in \mathcal{S}_d^{++}}
        \frac{1}{K} \sum_{k=1}^{K}
        \delta^2\!\left(\boldsymbol{\Sigma}, \boldsymbol{\Sigma}_k\right),
    \qquad
    \delta^2(\boldsymbol{A}, \boldsymbol{B})
      = \left\| \operatorname{logm}\!\left(
        \boldsymbol{A}^{-1/2} \boldsymbol{B} \boldsymbol{A}^{-1/2}\right)
        \right\|_{\mathbb{F}}^2 .
    $$

    ``hdrlib.core.estimation.frechet_mean_affine_invariant`` computes the same
    object by a different route (log-Euclidean start, fixed-step exponential
    updates). This one exists so that the corrected and uncorrected means of a
    comparison differ *only* by the correction, and not also by the optimiser.
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
    r"""Corrected squared Fisher distance between a fixed SPD matrix and SCMs.

    The affine-invariant distance between two SPD matrices, normalised here by
    the dimension so that it stays $\mathcal{O}(1)$ as $d$ grows, is

    $$
    \frac{1}{d}\,\delta^2\!\left(\boldsymbol{\Sigma}_0,
                                 \boldsymbol{\Sigma}\right)
    = \frac{1}{d} \left\| \operatorname{logm}\!\left(
      \boldsymbol{\Sigma}_0^{-1/2} \boldsymbol{\Sigma}
      \boldsymbol{\Sigma}_0^{-1/2}\right)
      \right\|_{\mathbb{F}}^2 .
    $$

    Replacing $\boldsymbol{\Sigma}$ by its sample covariance leaves an
    $\mathcal{O}(1)$ bias when the concentration ratio $c = d/N$ is not small.
    This function returns instead a consistent estimate of
    $\frac{1}{d}\delta^2(\boldsymbol{\Sigma}_0, \boldsymbol{\Sigma}_k)$ built
    from the observed $\widehat{\boldsymbol{\Sigma}}_k$. Writing
    $\boldsymbol{L}$ for the Cholesky factor of ``reference``,
    $\lambda_1, \dots, \lambda_d$ for the eigenvalues of
    $\boldsymbol{L}^{-1} \widehat{\boldsymbol{\Sigma}}_k
    \boldsymbol{L}^{-\mathrm{T}}$, and $z_1, \dots, z_d$ for those of
    $\operatorname{diag}(\boldsymbol{\lambda}) -
    \frac{1}{N}\sqrt{\boldsymbol{\lambda}}
    \sqrt{\boldsymbol{\lambda}}^{\mathrm{T}}$, the estimator is

    $$
    \begin{aligned}
    \widehat{\delta}^2
    &= \frac{1}{d} \sum_{i} \log^2 \lambda_i
     + \frac{2}{d} \sum_{i} \log \lambda_i
     - \frac{2}{d} \sum_{i,j} (\lambda_i - z_i)\, Q_{ij} \\
    &\quad - \left(\tfrac{1}{c} - 1\right) \log^2 (1 - c)
     - 2 \left(\tfrac{1}{c} - 1\right) \sum_{i} (\lambda_i - z_i)
       \frac{\log \lambda_i}{\lambda_i},
    \end{aligned}
    $$

    with

    $$
    Q_{ij} = \frac{\lambda_i \log(\lambda_i / \lambda_j)
                    - (\lambda_i - \lambda_j)
                    + \tfrac{1}{2}\mathbb{1}_{i=j}}
                   {(\lambda_i - \lambda_j)^2
                    + \lambda_i \mathbb{1}_{i=j}} .
    $$

    The $\mathbb{1}_{i=j}$ terms are not regularisations: on the diagonal they
    make $Q_{ii}$ evaluate the limit of the off-diagonal expression as
    $\lambda_j \to \lambda_i$ rather than $0/0$.

    **Reference**

    F. Bouchard, A. Mian, M. Tiomoko, G. Ginolhac and F. Pascal, "Random matrix
    theory improved Fréchet mean of symmetric positive definite matrices",
    *Proceedings of the 41st International Conference on Machine Learning*,
    PMLR 235:4403-4415, 2024.
    [arXiv:2405.06558](https://arxiv.org/abs/2405.06558)
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
    # Same pole as in _rmt_cost_grad: the correction divides by
    # (lambda_i - lambda_j)^2 off the diagonal, so two coincident eigenvalues
    # give 0/0. Comparing a matrix with itself whitens to the identity and hits
    # it exactly.
    eigenvalues = _separate_eigenvalues(be, backend, eigenvalues, n_features)
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
