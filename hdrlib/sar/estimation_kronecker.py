# Kronecker-structured covariance estimation via MM algorithm
# Reference: Y. Sun, P. Babu and D. P. Palomar, "Robust estimation of
# structured covariance matrix for heavy-tailed elliptical distributions",
# IEEE Transactions on Signal Processing, 64(14):3576-3590, 2016.
# doi:10.1109/TSP.2016.2546222
# Author: Ammar Mian

from typing import Tuple

from ..core.backend import (
    Array,
    get_backend_module,
    get_data_on_device,
    make_writable_copy,
    batched_trace,
)
from ..core.manifolds import sqrtm_psd, inv_sqrtm_invsqrtm_psd, KroneckerHermitianPositiveScaledGaussian


def _kronecker_quadratic_forms(
    M_i: Array,
    iA: Array,
    iB: Array,
    backend_name: str,
) -> Array:
    """Per-sample quadratic forms via the Kronecker trace identity.

    For x = vec_col(M_i[n]):
        Q[n] = Re(trace(A^{-1} M_i[n]^H B^{-1} M_i[n])) = Re(x^H Sigma^{-1} x)
    with Sigma = kron(A^T, B) = kron(conj(A), B), since
    vec(M)^H (A^T (x) B)^{-1} vec(M) = trace(M^H B^{-1} M A^{-1}).

    CONVENTION. The A factor estimated everywhere in this module is therefore
    the TRANSPOSE of the A of the model Sigma = A (x) B: feeding data whose
    covariance is A (x) B back through these routines returns conj(A), not A.
    Detection statistics are unaffected (they only depend on the fitted Sigma),
    but any comparison of an estimate against a ground-truth A must conjugate
    one side -- see sar.icrb.kronecker_component_errors, which does it.

    Parameters
    ----------
    M_i : Array of shape (..., N, b, a)
        Data in matrix form (Fortran reshape of last axis).
    iA : Array of shape (..., a, a)
        Inverse of A factor.
    iB : Array of shape (..., b, b)
        Inverse of B factor.
    backend_name : str

    Returns
    -------
    Array of shape (..., N)
        Per-sample quadratic forms.
    """
    be = get_backend_module(backend_name)
    # iB @ M_i: (..., 1, b, b) @ (..., N, b, a) -> (..., N, b, a)
    iB_M = iB[..., None, :, :] @ M_i
    # M_i^H @ iB @ M_i: (..., N, a, b) @ (..., N, b, a) -> (..., N, a, a)
    MH_iB_M = be.swapaxes(M_i, -1, -2).conj() @ iB_M
    # iA @ (M_i^H @ iB @ M_i): (..., 1, a, a) @ (..., N, a, a) -> (..., N, a, a)
    iA_MH_iB_M = iA[..., None, :, :] @ MH_iB_M
    return be.real(batched_trace(backend_name, iA_MH_iB_M))  # (..., N)


def _kronecker_mm_step(
    M_i: Array,
    A: Array,
    B: Array,
    N_eff: int,
    a: int,
    b: int,
    backend_name: str,
) -> Tuple[Array, Array]:
    """One MM geodesic update step for (A, B).

    Parameters
    ----------
    M_i : Array of shape (..., N_eff, b, a)
        Data in matrix form. N_eff = N for H1, T*N for H0.
    A : Array of shape (..., a, a)
    B : Array of shape (..., b, b)
    N_eff : int
        Total number of samples used in the sum.
    a, b : int
        Kronecker factor sizes.
    backend_name : str

    Returns
    -------
    A_new : Array of shape (..., a, a)
    B_new : Array of shape (..., b, b)
    """
    be = get_backend_module(backend_name)

    # iA = be.linalg.inv(A)  # (..., a, a)
    # iB = be.linalg.inv(B)  # (..., b, b)
    iA, sqrtm_A, isqrtm_A = inv_sqrtm_invsqrtm_psd(A, backend_name)
    iB, sqrtm_B, isqrtm_B = inv_sqrtm_invsqrtm_psd(B, backend_name)

    # M_num_A[n] = M_i[n]^H @ iB @ M_i[n]: (..., N_eff, a, a)
    iB_M = iB[..., None, :, :] @ M_i  # (..., N_eff, b, a)
    M_num_A = be.swapaxes(M_i, -1, -2).conj() @ iB_M  # (..., N_eff, a, a)

    # denominators[n] = Re(trace(iA @ M_num_A[n])): (..., N_eff)
    denominators = be.real(batched_trace(backend_name, iA[..., None, :, :] @ M_num_A))

    # M_A = (a / N_eff) * sum_n M_num_A[n] / denom[n]: (..., a, a)
    M_A = (a / N_eff) * (M_num_A / denominators[..., None, None]).sum(-3)

    # M_num_B[n] = M_i[n] @ iA @ M_i[n]^H: (..., N_eff, b, b)
    M_num_B = (M_i @ iA[..., None, :, :]) @ be.swapaxes(M_i, -1, -2).conj()
    M_B = (b / N_eff) * (M_num_B / denominators[..., None, None]).sum(-3)

    # Geodesic update: A_new = sqrtm(A) @ sqrtm(isqrtm(A) @ M_A @ isqrtm(A)) @ sqrtm(A)
    A_new = sqrtm_A @ sqrtm_psd(isqrtm_A @ M_A @ isqrtm_A, backend_name) @ sqrtm_A
    B_new = sqrtm_B @ sqrtm_psd(isqrtm_B @ M_B @ isqrtm_B, backend_name) @ sqrtm_B

    return A_new, B_new


def kronecker_mm_h1(
    X: Array,
    a: int,
    b: int,
    tol: float = 1e-4,
    iter_max: int = 30,
    backend_name: str = "numpy",
) -> Tuple[Array, Array, Array]:
    """Per-date Kronecker MM estimator (H1 model).

    Estimates one (A_t, B_t) per time date independently, treating
    the T dimension as part of the batch.

    Parameters
    ----------
    X : Array of shape (..., T, N, p) where p = a*b
    a, b : int
        Kronecker factor sizes.
    tol : float
        Convergence tolerance on relative Frobenius norm.
    iter_max : int
        Maximum number of iterations.
    backend_name : str

    Returns
    -------
    A : Array of shape (..., T, a, a)
    B : Array of shape (..., T, b, b)
    tau : Array of shape (..., T, N)
        Per-sample textures tau_{t,n} = Q_{t,n} / p.
    """
    be = get_backend_module(backend_name)
    N = X.shape[-2]
    batch_shape = X.shape[:-2]  # (..., T)

    # Fortran reshape: x of shape (p,) -> M of shape (b, a)
    # Equivalent to C reshape to (a, b) then swapaxes(-1, -2)
    M_i = X.reshape(*X.shape[:-1], a, b).swapaxes(-1, -2)  # (..., T, N, b, a)

    eye_a = get_data_on_device(be.eye(a, dtype=X.dtype), backend_name)
    eye_b = get_data_on_device(be.eye(b, dtype=X.dtype), backend_name)
    A = make_writable_copy(backend_name, be.broadcast_to(eye_a, batch_shape + (a, a)))
    B = make_writable_copy(backend_name, be.broadcast_to(eye_b, batch_shape + (b, b)))

    converged = get_data_on_device(be.zeros(batch_shape, dtype=bool), backend_name)
    for _ in range(iter_max):
        A_new, B_new = _kronecker_mm_step(M_i, A, B, N, a, b, backend_name)

        delta_A = be.linalg.norm(A_new - A, axis=(-2, -1)) / be.linalg.norm(
            A, axis=(-2, -1)
        )
        delta_B = be.linalg.norm(B_new - B, axis=(-2, -1)) / be.linalg.norm(
            B, axis=(-2, -1)
        )
        converged = converged | ((delta_A <= tol) & (delta_B <= tol))
        mask = (~converged)[..., None, None]
        A = be.where(mask, A_new, A)
        B = be.where(mask, B_new, B)
        del A_new, B_new, delta_A, delta_B, mask

    iA = be.linalg.inv(A)
    iB = be.linalg.inv(B)
    tau = _kronecker_quadratic_forms(M_i, iA, iB, backend_name) / (a * b)  # (..., T, N)

    return A, B, tau


def kronecker_mm_h0(
    X: Array,
    a: int,
    b: int,
    tol: float = 1e-4,
    iter_max: int = 30,
    backend_name: str = "numpy",
) -> Tuple[Array, Array, Array]:
    """Joint Kronecker MM estimator (H0 model).

    Estimates a single (A, B) shared across all T dates by pooling
    all T*N samples together.

    Parameters
    ----------
    X : Array of shape (..., T, N, p) where p = a*b
    a, b : int
        Kronecker factor sizes.
    tol : float
        Convergence tolerance on relative Frobenius norm.
    iter_max : int
        Maximum number of iterations.
    backend_name : str

    Returns
    -------
    A : Array of shape (..., a, a)
    B : Array of shape (..., b, b)
    tau : Array of shape (..., N)
        Per-sample textures tau_n = mean_t Q_{t,n} / p.
    """
    be = get_backend_module(backend_name)
    T = X.shape[-3]
    N = X.shape[-2]
    batch_shape = X.shape[:-3]  # (...)

    # Fortran reshape: (..., T, N, b, a)
    M_i = X.reshape(*X.shape[:-1], a, b).swapaxes(-1, -2)

    # Pool T and N into a single dimension for the MM step
    M_i_flat = M_i.reshape(*batch_shape, T * N, b, a)  # (..., T*N, b, a)

    eye_a = get_data_on_device(be.eye(a, dtype=X.dtype), backend_name)
    eye_b = get_data_on_device(be.eye(b, dtype=X.dtype), backend_name)
    A = make_writable_copy(backend_name, be.broadcast_to(eye_a, batch_shape + (a, a)))
    B = make_writable_copy(backend_name, be.broadcast_to(eye_b, batch_shape + (b, b)))

    converged = get_data_on_device(be.zeros(batch_shape, dtype=bool), backend_name)
    for _ in range(iter_max):
        A_new, B_new = _kronecker_mm_step(M_i_flat, A, B, T * N, a, b, backend_name)

        delta_A = be.linalg.norm(A_new - A, axis=(-2, -1)) / be.linalg.norm(
            A, axis=(-2, -1)
        )
        delta_B = be.linalg.norm(B_new - B, axis=(-2, -1)) / be.linalg.norm(
            B, axis=(-2, -1)
        )
        converged = converged | ((delta_A <= tol) & (delta_B <= tol))
        mask = (~converged)[..., None, None]
        A = be.where(mask, A_new, A)
        B = be.where(mask, B_new, B)
        del A_new, B_new, delta_A, delta_B, mask

    # tau_n = mean_t Q_{t,n} / p  (average quadratic form over dates)
    iA = be.linalg.inv(A)
    iB = be.linalg.inv(B)
    Q = _kronecker_quadratic_forms(M_i_flat, iA, iB, backend_name)  # (..., T*N)
    Q = Q.reshape(*batch_shape, T, N)  # (..., T, N)
    tau = Q.mean(-2) / (a * b)  # mean over T: (..., N)

    return A, B, tau


# -----------------------------------------------------------------------
# Riemannian natural gradient for Kronecker scaled Gaussian model
# -----------------------------------------------------------------------

def _neg_log_likelihood_kronecker_scaled_gaussian(
    X: Array,
    A: Array,
    B: Array,
    tau: Array,
    a: int,
    b: int,
    backend_name: str,
) -> Array:
    """Negative log-likelihood for Kronecker scaled Gaussian (det(A)=det(B)=1).

    Parameters
    ----------
    X : Array of shape (..., N, p) where p = a*b
    A : Array of shape (..., a, a)
    B : Array of shape (..., b, b)
    tau : Array of shape (..., N, 1)
    a, b : int

    Returns
    -------
    Array of shape (...)
    """
    be = get_backend_module(backend_name)
    N, p = X.shape[-2], X.shape[-1]
    M_i = X.reshape(*X.shape[:-1], a, b).swapaxes(-1, -2)  # (..., N, b, a)
    iA = be.linalg.inv(A)
    iB = be.linalg.inv(B)
    Q = _kronecker_quadratic_forms(M_i, iA, iB, backend_name)  # (..., N)
    tau_flat = tau[..., 0]  # (..., N)
    L = p * be.log(tau_flat) + Q / tau_flat
    return be.sum(L, axis=-1) / (N * p)


def _rgrad_kronecker_scaled_gaussian(
    X: Array,
    A: Array,
    B: Array,
    tau: Array,
    manifold: KroneckerHermitianPositiveScaledGaussian,
    be,
    a: int,
    b: int,
    backend_name: str,
) -> Tuple[Array, Array, Array]:
    """Riemannian gradient of neg-log-likelihood for Kronecker scaled Gaussian.

    Gradient on the product manifold SHPD(a) x SHPD(b) x SPV(N) with FIM
    weights [b/p, a/p, 1/N]. Ported from rgrad_scaledgaussian_kronecker in
    the original non-backend-agnostic implementation.

    Parameters
    ----------
    X : Array of shape (..., N, p)
    A : Array of shape (..., a, a)
    B : Array of shape (..., b, b)
    tau : Array of shape (..., N, 1)
    manifold : KroneckerHermitianPositiveScaledGaussian
    be : backend module
    a, b : int

    Returns
    -------
    r_A : Array of shape (..., a, a)
    r_B : Array of shape (..., b, b)
    r_tau : Array of shape (..., N)
    """
    N, p = X.shape[-2], X.shape[-1]
    M_i = X.reshape(*X.shape[:-1], a, b).swapaxes(-1, -2)  # (..., N, b, a)
    M_i_H = be.swapaxes(M_i, -1, -2).conj()  # (..., N, a, b)

    iA = be.linalg.inv(A)
    iB = be.linalg.inv(B)
    tau_flat = tau[..., 0]  # (..., N)

    # Gradient for A: proj_SHPD(A, M_i^H @ iB @ M_i).
    # No conjugation: with q_i = tr(A^-1 M_i^H B^-1 M_i), which is the
    # quadratic form this module actually implements (Sigma = A^T (x) B),
    # D q_i[xi_A] = -tr(A^-1 xi_A A^-1 N_A) = -<N_A, xi_A>_A with
    # N_A = M_i^H B^-1 M_i.  The conjugated form belongs to the Sigma = A (x) B
    # convention of the reference implementation, and mixing the two makes this
    # gradient inconsistent with _neg_log_likelihood_kronecker_scaled_gaussian.
    iB_M = iB[..., None, :, :] @ M_i  # (..., N, b, a)
    M_num_A = M_i_H @ iB_M  # (..., N, a, a) = M_i^H @ iB @ M_i
    weighted_A = (M_num_A / (b * tau_flat[..., None, None])).sum(-3) / N
    r_A = -manifold.manifolds[0].proj(A, weighted_A)  # (..., a, a)

    # Gradient for B: proj_SHPD(B, M_i @ iA @ M_i^H), same argument with
    # q_i = tr(B^-1 M_i A^-1 M_i^H).
    M_grad_B = M_i @ iA[..., None, :, :] @ M_i_H  # (..., N, b, b)
    weighted_B = (M_grad_B / (a * tau_flat[..., None, None])).sum(-3) / N
    r_B = -manifold.manifolds[1].proj(B, weighted_B)  # (..., b, b)

    # Gradient for tau: tau_n - Q_n/p
    Q = _kronecker_quadratic_forms(M_i, iA, iB, backend_name)  # (..., N)
    r_tau = tau_flat - Q / p  # (..., N)

    return r_A, r_B, r_tau


def _armijo_backtracking_kronecker_scaled_gaussian(
    X: Array,
    A: Array,
    B: Array,
    tau: Array,
    r_A: Array,
    r_B: Array,
    r_tau: Array,
    manifold: KroneckerHermitianPositiveScaledGaussian,
    be,
    a: int,
    b: int,
    alpha_0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_backtracks: int = 30,
    backend_name: str = "numpy",
    alpha_0_tau: "float | None" = None,
    max_exp_tau: "float | None" = None,
) -> Tuple[Array, Array, Array, Array]:
    """Per-batch Armijo backtracking line search for Kronecker scaled Gaussian.

    Parameters
    ----------
    X : Array of shape (..., N, p)
    A, B : Arrays of shape (..., a, a) and (..., b, b)
    tau : Array of shape (..., N, 1)
    r_A, r_B : Riemannian gradients (..., a, a) and (..., b, b)
    r_tau : Riemannian gradient (..., N)
    manifold : KroneckerHermitianPositiveScaledGaussian
    a, b : int
    alpha_0_tau : separate initial step size for tau (None → use alpha_0 for all).
    max_exp_tau : if set, caps the tau exp-map argument to this value to prevent
        overshoot when tau << Q/p.  Only applied when alpha_0_tau is given.

    Returns
    -------
    alpha : Array of shape (...)
    A_new, B_new, tau_new : updated parameters
    """
    tau_v = tau[..., 0]  # (..., N)
    f0 = _neg_log_likelihood_kronecker_scaled_gaussian(X, A, B, tau, a, b, backend_name)
    sq_grad_norm = be.abs(be.real(manifold.inner(
        [A, B, tau_v], [r_A, r_B, r_tau], [r_A, r_B, r_tau]
    )))
    grad_norm_finite = be.isfinite(sq_grad_norm)

    if alpha_0_tau is not None:
        # Separate step sizes for (A, B) and tau.
        # Safety cap: clamp exp argument for tau so tau never overshoots by more
        # than exp(max_exp_tau) in one step (prevents Armijo accepting a
        # numerically huge tau that has moved past the MLE).
        if max_exp_tau is not None:
            tau_v_safe = be.where(tau_v > 1e-12, tau_v, be.ones_like(tau_v) * 1e-12)
            max_ratio = float(be.real(be.max(be.abs(r_tau) / tau_v_safe)))
            if max_ratio > 0:
                alpha_0_tau = min(alpha_0_tau, max_exp_tau / max_ratio)

        alpha_AB  = get_data_on_device(be.ones(f0.shape, dtype=f0.dtype) * alpha_0,     backend_name)
        alpha_tau = get_data_on_device(be.ones(f0.shape, dtype=f0.dtype) * alpha_0_tau, backend_name)
        accepted  = get_data_on_device(be.zeros(f0.shape, dtype=bool), backend_name)
        last_A, last_B, last_tau = A, B, tau

        for _ in range(max_backtracks):
            result = manifold.retr(
                [A, B, tau_v],
                [
                    -(alpha_AB[..., None, None]  * r_A),
                    -(alpha_AB[..., None, None]  * r_B),
                    -(alpha_tau[..., None] * r_tau),
                ],
            )
            A_new, B_new, tau_v_new = result[0], result[1], result[2]
            tau_new = tau_v_new[..., None]

            f_new = _neg_log_likelihood_kronecker_scaled_gaussian(
                X, A_new, B_new, tau_new, a, b, backend_name
            )
            armijo_ok = be.where(
                grad_norm_finite,
                f_new <= f0 - c * alpha_AB * sq_grad_norm,
                be.isfinite(f_new) & (f_new < f0),
            )

            newly_accepted = armijo_ok & ~accepted
            last_A    = be.where(newly_accepted[..., None, None], A_new,   last_A)
            last_B    = be.where(newly_accepted[..., None, None], B_new,   last_B)
            last_tau  = be.where(newly_accepted[..., None, None], tau_new, last_tau)
            accepted  = accepted | armijo_ok
            alpha_AB  = be.where(accepted, alpha_AB,  alpha_AB  * rho)
            alpha_tau = be.where(accepted, alpha_tau, alpha_tau * rho)

        return alpha_AB, last_A, last_B, last_tau

    # Original joint-alpha path (backward compatible).
    alpha = get_data_on_device(be.ones(f0.shape, dtype=f0.dtype) * alpha_0, backend_name)
    accepted = get_data_on_device(be.zeros(f0.shape, dtype=bool), backend_name)
    last_A, last_B, last_tau = A, B, tau

    for _ in range(max_backtracks):
        result = manifold.retr(
            [A, B, tau_v],
            [
                -(alpha[..., None, None] * r_A),
                -(alpha[..., None, None] * r_B),
                -(alpha[..., None] * r_tau),
            ],
        )
        A_new, B_new, tau_v_new = result[0], result[1], result[2]
        tau_new = tau_v_new[..., None]

        f_new = _neg_log_likelihood_kronecker_scaled_gaussian(
            X, A_new, B_new, tau_new, a, b, backend_name
        )
        armijo_ok = be.where(
            grad_norm_finite,
            f_new <= f0 - c * alpha * sq_grad_norm,
            be.isfinite(f_new) & (f_new < f0),
        )

        newly_accepted = armijo_ok & ~accepted
        last_A = be.where(newly_accepted[..., None, None], A_new, last_A)
        last_B = be.where(newly_accepted[..., None, None], B_new, last_B)
        last_tau = be.where(newly_accepted[..., None, None], tau_new, last_tau)
        accepted = accepted | armijo_ok
        alpha = be.where(accepted, alpha, alpha * rho)

    return alpha, last_A, last_B, last_tau


def kronecker_riemannian_gd_h0(
    X: Array,
    a: int,
    b: int,
    iter_max: int = 200,
    tol: float = 1e-8,
    alpha_0: float = 1.0,
    backend_name: str = "numpy",
    init: "tuple | None" = None,
) -> Tuple[Array, Array, Array]:
    """Offline MLE of (A, B, tau) under H0 by Riemannian gradient descent.

    This is the batch counterpart of OnlineKroneckerEstimator: same manifold,
    same Fisher metric, same gradient, but the gradient is averaged over all T
    dates at every iteration and an Armijo line search sets the step.  It is
    the "GD" reference of the paper below, against which the recursive
    estimator is compared; kronecker_mm_h0 targets the same MLE by a different
    algorithm and can be used instead.

    **Reference**

    A. Mian, G. Ginolhac, F. Bouchard and A. Breloy, "Online change detection in
    SAR time-series with Kronecker product structured scaled Gaussian models",
    *Signal Processing*, 224:109589, 2024.
    [doi:10.1016/j.sigpro.2024.109589](https://doi.org/10.1016/j.sigpro.2024.109589)

    Parameters
    ----------
    X : Array of shape (..., T, N, p)
    a, b : int
    iter_max : int
        Maximum number of gradient iterations.
    tol : float
        Stop when the relative decrease of the negative log-likelihood falls
        below this value.
    alpha_0 : float
        Initial Armijo step.
    backend_name : str
    init : (A, B, tau) or None
        Starting point; identity and ones when None.

    Returns
    -------
    A : Array of shape (..., a, a)
    B : Array of shape (..., b, b)
    tau : Array of shape (..., N, 1)
    """
    be = get_backend_module(backend_name)
    T, N, p = X.shape[-3], X.shape[-2], X.shape[-1]
    batch_shape = X.shape[:-3]
    manifold = KroneckerHermitianPositiveScaledGaussian(a, b, N, backend_name=backend_name)

    if init is None:
        A = get_data_on_device(
            be.broadcast_to(be.eye(a, dtype=X.dtype), batch_shape + (a, a)) * 1, backend_name)
        B = get_data_on_device(
            be.broadcast_to(be.eye(b, dtype=X.dtype), batch_shape + (b, b)) * 1, backend_name)
        tau = get_data_on_device(
            be.ones(batch_shape + (N, 1), dtype=be.real(X[..., :1, :1, :1]).dtype), backend_name)
    else:
        A, B, tau = init

    f_prev = None
    for _ in range(iter_max):
        # Gradient of the full-data cost: mean over dates of the per-date gradients.
        r_A = r_B = r_tau = None
        for t in range(T):
            g_A, g_B, g_tau = _rgrad_kronecker_scaled_gaussian(
                X[..., t, :, :], A, B, tau, manifold, be, a, b, backend_name)
            if r_A is None:
                r_A, r_B, r_tau = g_A, g_B, g_tau
            else:
                r_A, r_B, r_tau = r_A + g_A, r_B + g_B, r_tau + g_tau
        r_A, r_B, r_tau = r_A / T, r_B / T, r_tau / T

        # Armijo on the full-data cost, evaluated as the mean over dates.
        tau_v = tau[..., 0]
        alpha = alpha_0
        f0 = sum(
            _neg_log_likelihood_kronecker_scaled_gaussian(
                X[..., t, :, :], A, B, tau, a, b, backend_name)
            for t in range(T)
        ) / T
        accepted = False
        for _ in range(40):
            A_new, B_new, tau_v_new = manifold.retr(
                [A, B, tau_v], [-alpha * r_A, -alpha * r_B, -alpha * r_tau])
            tau_new = tau_v_new[..., None]
            f_new = sum(
                _neg_log_likelihood_kronecker_scaled_gaussian(
                    X[..., t, :, :], A_new, B_new, tau_new, a, b, backend_name)
                for t in range(T)
            ) / T
            if bool(be.all(be.isfinite(f_new) & (f_new < f0))):
                A, B, tau = A_new, B_new, tau_new
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        f_cur = float(be.real(be.max(f_new)))
        if f_prev is not None and abs(f_prev - f_cur) <= tol * max(abs(f_prev), 1.0):
            break
        f_prev = f_cur

    return A, B, tau
