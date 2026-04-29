# Kronecker-structured covariance estimation via MM algorithm
# Reference: Sun, Babu, Palomar, IEEE TSP 2016
# Author: Ammar Mian

from typing import Tuple, Union

from .backend import (
    Array,
    Backend,
    get_backend_module,
    get_data_on_device,
    batched_eigh,
    get_diagembed,
    is_complex,
    make_writable_copy,
    batched_trace,
)
from .manifolds import sqrtm_psd, invsqrtm_psd


def _eigh_psd(X: Array, backend_name: str):
    """Symmetrize then eigh. Returns (eigenvalues, eigenvectors)."""
    be = get_backend_module(backend_name)
    X = 0.5 * (X + be.swapaxes(X, -1, -2).conj())
    return batched_eigh(backend_name, X)


def _inv_sqrtm_invsqrtm_psd(X: Array, backend_name: str) -> Tuple[Array, Array, Array]:
    """Compute inv, sqrtm and invsqrtm without computing EVD two times"""
    be = get_backend_module(backend_name)
    eigenvalues, eigenvectors = _eigh_psd(X, backend_name)
    sqrt_eigvals = be.sqrt(be.abs(eigenvalues))
    inv_sqrt_eigvals = 1.0 / be.sqrt(be.abs(eigenvalues))
    inv_eigvals = 1.0 / be.abs(eigenvalues)
    if is_complex(backend_name, X):
        inv_sqrt_eigvals = inv_sqrt_eigvals + 0j
        sqrt_eigvals = sqrt_eigvals + 0j
        inv_eigvals = inv_eigvals + 0j

    inv = be.einsum(
        "...ab,...bc,...cd->...ad",
        eigenvectors,
        get_diagembed(backend_name, inv_eigvals),
        be.swapaxes(eigenvectors, -1, -2).conj(),
    )
    sqrtm = be.einsum(
        "...ab,...bc,...cd->...ad",
        eigenvectors,
        get_diagembed(backend_name, sqrt_eigvals),
        be.swapaxes(eigenvectors, -1, -2).conj(),
    )

    invsqrtm = be.einsum(
        "...ab,...bc,...cd->...ad",
        eigenvectors,
        get_diagembed(backend_name, inv_sqrt_eigvals),
        be.swapaxes(eigenvectors, -1, -2).conj(),
    )
    return inv, sqrtm, invsqrtm


def _kronecker_quadratic_forms(
    M_i: Array,
    iA: Array,
    iB: Array,
    backend_name: str,
) -> Array:
    """Per-sample quadratic forms via the Kronecker trace identity.

    For x = vec_F(M_i[n]) and Sigma = kron(A, B):
        Q[n] = Re(x^H Sigma^{-1} x) = Re(trace(A^{-1} M_i[n]^H B^{-1} M_i[n]))

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
    iA, sqrtm_A, isqrtm_A = _inv_sqrtm_invsqrtm_psd(A, backend_name)
    iB, sqrtm_B, isqrtm_B = _inv_sqrtm_invsqrtm_psd(B, backend_name)

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

    for _ in range(iter_max):
        A_new, B_new = _kronecker_mm_step(M_i, A, B, N, a, b, backend_name)

        delta_A = be.linalg.norm(A_new - A, axis=(-2, -1)) / be.linalg.norm(
            A, axis=(-2, -1)
        )
        delta_B = be.linalg.norm(B_new - B, axis=(-2, -1)) / be.linalg.norm(
            B, axis=(-2, -1)
        )
        A, B = A_new, B_new

        if bool(be.all(delta_A <= tol)) and bool(be.all(delta_B <= tol)):
            break

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

    for _ in range(iter_max):
        A_new, B_new = _kronecker_mm_step(M_i_flat, A, B, T * N, a, b, backend_name)

        delta_A = be.linalg.norm(A_new - A, axis=(-2, -1)) / be.linalg.norm(
            A, axis=(-2, -1)
        )
        delta_B = be.linalg.norm(B_new - B, axis=(-2, -1)) / be.linalg.norm(
            B, axis=(-2, -1)
        )
        A, B = A_new, B_new

        if bool(be.all(delta_A <= tol)) and bool(be.all(delta_B <= tol)):
            break

    # tau_n = mean_t Q_{t,n} / p  (average quadratic form over dates)
    iA = be.linalg.inv(A)
    iB = be.linalg.inv(B)
    Q = _kronecker_quadratic_forms(M_i_flat, iA, iB, backend_name)  # (..., T*N)
    Q = Q.reshape(*batch_shape, T, N)  # (..., T, N)
    tau = Q.mean(-2) / (a * b)  # mean over T: (..., N)

    return A, B, tau
