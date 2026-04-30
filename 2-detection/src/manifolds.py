# Riemannian manifolds with backend abstraction
# Author: Ammar Mian
# Ported from older pymanopt-based code to backend-agnostic patterns

from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Optional

from .backend import (
    Backend,
    Array,
    get_backend_module,
    batched_eigh,
    batched_trace,
    get_diagembed,
    is_complex,
    sample_standard_normal,
    sample_uniform,
)


# ============================================================================
# Helper Functions (Module-level)
# ============================================================================
def multiherm(A: Array, backend: Union[str, Backend]) -> Array:
    """Hermitian symmetrization: (A + A^H) / 2

    Parameters
    ----------
    A : Array
        Input array of shape (..., n, n)
    backend : Union[str, Backend]
        Backend specification

    Returns
    -------
    Array
        Hermitian symmetrized array
    """
    be = get_backend_module(backend)
    return 0.5 * (A + be.swapaxes(A, -1, -2).conj())


def eigh_psd(X: Array, backend_name: Union[str, Backend]):
    """Symmetrize then eigh. Returns (eigenvalues, eigenvectors)."""
    be = get_backend_module(backend_name)
    X = 0.5 * (X + be.swapaxes(X, -1, -2).conj())
    return batched_eigh(backend_name, X)


def sqrtm_invsqrtm_psd(
    X: Array, backend_name: Union[str, Backend]
) -> Tuple[Array, Array]:
    """Compute inv, sqrtm and invsqrtm without computing EVD two times"""
    be = get_backend_module(backend_name)
    eigenvalues, eigenvectors = eigh_psd(X, backend_name)
    sqrt_eigvals = be.sqrt(be.abs(eigenvalues))
    inv_sqrt_eigvals = 1.0 / be.sqrt(be.abs(eigenvalues))
    inv_eigvals = 1.0 / be.abs(eigenvalues)
    if is_complex(backend_name, X):
        inv_sqrt_eigvals = inv_sqrt_eigvals + 0j
        sqrt_eigvals = sqrt_eigvals + 0j
        inv_eigvals = inv_eigvals + 0j

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
    return sqrtm, invsqrtm


def inv_sqrtm_invsqrtm_psd(
    X: Array, backend_name: Union[str, Backend]
) -> Tuple[Array, Array, Array]:
    """Compute inv, sqrtm and invsqrtm without computing EVD three times"""
    be = get_backend_module(backend_name)
    eigenvalues, eigenvectors = eigh_psd(X, backend_name)
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


def sqrtm_psd(X: Array, backend: Union[str, Backend]) -> Array:
    """Matrix square root of positive semi-definite matrix via eigendecomposition.

    For Hermitian PSD X, computes X^{1/2} = Q @ diag(sqrt(d)) @ Q^H where
    X = Q @ diag(d) @ Q^H from eigendecomposition.

    Parameters
    ----------
    X : Array
        Hermitian PSD matrices of shape (..., n, n)
    backend : Union[str, Backend]
        Backend specification

    Returns
    -------
    Array
        Square root of X, same shape as input
    """
    be = get_backend_module(backend)
    eigenvalues, eigenvectors = batched_eigh(backend, X)
    sqrt_eigvals = be.sqrt(be.abs(eigenvalues))
    if is_complex(backend, X):
        sqrt_eigvals = (
            sqrt_eigvals.astype(eigenvectors.dtype)
            if hasattr(sqrt_eigvals, "astype")
            else sqrt_eigvals + 0j
        )
    diag_sqrt = get_diagembed(backend, sqrt_eigvals)

    return be.einsum(
        "...ab,...bc,...cd->...ad",
        eigenvectors,
        diag_sqrt,
        be.swapaxes(eigenvectors, -1, -2).conj(),
    )


def invsqrtm_psd(X: Array, backend: Union[str, Backend]) -> Array:
    """Matrix inverse square root of PSD matrix via eigendecomposition.

    For Hermitian PSD X, computes X^{-1/2} = Q^H @ diag(1/sqrt(d)) @ Q where
    X = Q @ diag(d) @ Q^H from eigendecomposition.

    Parameters
    ----------
    X : Array
        Hermitian PSD matrices of shape (..., n, n)
    backend : Union[str, Backend]
        Backend specification

    Returns
    -------
    Array
        Inverse square root of X, same shape as input
    """
    be = get_backend_module(backend)
    eigenvalues, eigenvectors = batched_eigh(backend, X)
    inv_sqrt_eigvals = 1.0 / be.sqrt(be.abs(eigenvalues))
    if is_complex(backend, X):
        inv_sqrt_eigvals = (
            inv_sqrt_eigvals.astype(eigenvectors.dtype)
            if hasattr(inv_sqrt_eigvals, "astype")
            else inv_sqrt_eigvals + 0j
        )
    diag_inv_sqrt = get_diagembed(backend, inv_sqrt_eigvals)

    return be.einsum(
        "...ab,...bc,...cd->...ad",
        eigenvectors,
        diag_inv_sqrt,
        be.swapaxes(eigenvectors, -1, -2).conj(),
    )


def logm_psd(X: Array, backend: Union[str, Backend]) -> Array:
    """Matrix logarithm of positive definite matrix via eigendecomposition.

    For Hermitian PD X, computes log(X) = Q @ diag(log(d)) @ Q^H where
    X = Q @ diag(d) @ Q^H from eigendecomposition.

    Parameters
    ----------
    X : Array
        Hermitian PD matrices of shape (..., n, n)
    backend : Union[str, Backend]
        Backend specification

    Returns
    -------
    Array
        Matrix logarithm of X, same shape as input
    """
    be = get_backend_module(backend)
    eigenvalues, eigenvectors = batched_eigh(backend, X)
    log_eigvals = be.log(be.abs(eigenvalues))
    if is_complex(backend, X):
        log_eigvals = (
            log_eigvals.astype(eigenvectors.dtype)
            if hasattr(log_eigvals, "astype")
            else log_eigvals + 0j
        )
    diag_log = get_diagembed(backend, log_eigvals)

    return be.einsum(
        "...ab,...bc,...cd->...ad",
        eigenvectors,
        diag_log,
        be.swapaxes(eigenvectors, -1, -2).conj(),
    )


# ============================================================================
# Base Class
# ============================================================================
class Manifold(ABC):
    """Abstract base class for Riemannian manifolds.

    All manifolds support backend-agnostic operations (numpy/torch).
    """

    backend_name: Union[str, Backend]

    @abstractmethod
    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Inner product of tangent vectors u, v at point x.

        Parameters
        ----------
        x : Array
            Point on manifold
        u : Array
            Tangent vector at x
        v : Array
            Tangent vector at x

        Returns
        -------
        Array
            Inner product (scalar or batch of scalars)
        """
        pass

    @abstractmethod
    def proj(self, x: Array, u: Array) -> Array:
        """Orthogonal projection onto tangent space at x.

        Parameters
        ----------
        x : Array
            Point on manifold
        u : Array
            Vector to project

        Returns
        -------
        Array
            Projected tangent vector
        """
        pass

    @abstractmethod
    def exp(self, x: Array, u: Array) -> Array:
        """Exponential map: curve from x along tangent vector u.

        Parameters
        ----------
        x : Array
            Base point on manifold
        u : Array
            Tangent vector at x

        Returns
        -------
        Array
            exp_x(u) on manifold
        """
        pass

    @abstractmethod
    def log(self, x: Array, y: Array) -> Array:
        """Logarithmic map: tangent vector from x to y.

        Parameters
        ----------
        x : Array
            Base point on manifold
        y : Array
            Target point on manifold

        Returns
        -------
        Array
            Tangent vector u such that exp_x(u) ≈ y
        """
        pass

    @abstractmethod
    def dist(self, x: Array, y: Array) -> Array:
        """Riemannian distance between points x and y.

        Parameters
        ----------
        x : Array
            Point on manifold
        y : Array
            Point on manifold

        Returns
        -------
        Array
            Distance (scalar or batch of scalars)
        """
        pass

    @abstractmethod
    def rand(self, batch_shape: Tuple = ()) -> Array:
        """Sample random point on manifold.

        Parameters
        ----------
        batch_shape : Tuple, optional
            Batch dimensions. Default empty (single point).

        Returns
        -------
        Array
            Random point of shape (*batch_shape, *manifold_shape)
        """
        pass

    def norm(self, x: Array, u: Array) -> Array:
        """Norm of tangent vector u at point x.

        Parameters
        ----------
        x : Array
            Point on manifold
        u : Array
            Tangent vector at x

        Returns
        -------
        Array
            Norm (scalar or batch of scalars)
        """
        be = get_backend_module(self.backend_name)
        return be.sqrt(self.inner(x, u, u))

    @abstractmethod
    def zerovec(self, x: Array) -> Array:
        """Zero tangent vector at point x."""
        pass

    @abstractmethod
    def randvec(self, x: Array) -> Array:
        """Random unit tangent vector at point x."""
        pass

    def retr(self, x: Array, u: Array) -> Array:
        """Retraction: approximate exponential map (cheaper, manifold-specific).

        Defaults to NotImplementedError; override in subclasses where a
        cheaper retraction than exp is available.
        """
        raise NotImplementedError()

    def egrad2rgrad(self, x: Array, egrad: Array) -> Array:
        """Convert Euclidean gradient to Riemannian gradient.

        For submanifolds, this involves projection and metric conversion.

        Parameters
        ----------
        x : Array
            Point on manifold
        egrad : Array
            Euclidean gradient at x

        Returns
        -------
        Array
            Riemannian gradient at x
        """
        raise NotImplementedError()

    def ehess2rhess(self, x: Array, egrad: Array, ehess: Array, u: Array) -> Array:
        """Convert Euclidean Hessian to Riemannian Hessian.

        Parameters
        ----------
        x : Array
            Point on manifold
        egrad : Array
            Euclidean gradient at x
        ehess : Array
            Euclidean Hessian at x
        u : Array
            Tangent direction (used in second-order derivative)

        Returns
        -------
        Array
            Riemannian Hessian at x in direction u
        """
        raise NotImplementedError()

    def transp(self, x1: Array, x2: Array, d: Array) -> Array:
        """Parallel transport of tangent vector d from x1 to x2.

        Parameters
        ----------
        x1 : Array
            Base point (where d lives)
        x2 : Array
            Target point
        d : Array
            Tangent vector at x1

        Returns
        -------
        Array
            Transported tangent vector at x2
        """
        raise NotImplementedError()


# ============================================================================
# Hermitian Positive Definite Manifold
# ============================================================================
class HermitianPositiveDefinite(Manifold):
    """Manifold of Hermitian positive definite matrices.

    Points are (..., n, n) complex Hermitian PD matrices.
    Tangent vectors are Hermitian matrices.
    Uses canonical geometry (generalized eigenvalue problem) and retraction via matrix exponential.
    """

    def __init__(self, n: int, backend_name: Union[str, Backend] = "numpy"):
        """Initialize HPD manifold.

        Parameters
        ----------
        n : int
            Size of matrices (n x n)
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        self.n = n
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Inner product: <u, v>_x = tr(solve(x, u) @ solve(x, v)^H)"""
        sol_u = self.be.linalg.solve(x, u)
        sol_v = self.be.linalg.solve(x, v)
        return self.be.real(
            self.be.einsum(
                "...ij,...ij->...",
                sol_u,
                self.be.swapaxes(sol_v, -1, -2).conj(),
            )
        )

    def norm(self, x: Array, u: Array) -> Array:
        """Norm via Cholesky: ||u||_x = ||L^{-1} u (L^{-H})||_F where x = LL^H"""
        c = self.be.linalg.cholesky(x)
        c_inv = self.be.linalg.inv(c)
        temp = c_inv @ u @ self.be.swapaxes(c_inv, -1, -2).conj()
        return self.be.linalg.norm(temp, axis=(-2, -1))

    def proj(self, x: Array, u: Array) -> Array:
        """Project u onto tangent space: multiherm(u)"""
        return multiherm(u, self.backend_name)

    def egrad2rgrad(self, x: Array, egrad: Array) -> Array:
        """Riemannian gradient: x @ multiherm(egrad) @ x"""
        return x @ multiherm(egrad, self.backend_name) @ x

    def exp(self, x: Array, u: Array) -> Array:
        """Matrix exponential retraction: exp_x(u) = sqrtm(x) exp(X^{-1/2} u X^{-1/2}) sqrtm(x)

        where X = x.
        """
        x_sqrt = sqrtm_psd(x, self.backend_name)
        x_isqrt = invsqrtm_psd(x, self.backend_name)

        # Compute exponent argument
        exp_arg = x_isqrt @ u @ x_isqrt
        eigvals, eigvecs = batched_eigh(self.backend_name, exp_arg)
        exp_eigvals = self.be.exp(eigvals)
        diag_exp = get_diagembed(self.backend_name, exp_eigvals)

        exp_mat = self.be.einsum(
            "...ab,...bc,...cd->...ad",
            eigvecs,
            diag_exp,
            self.be.swapaxes(eigvecs, -1, -2).conj(),
        )

        # exp_x(u) = sqrtm @ exp_mat @ sqrtm
        result = x_sqrt @ exp_mat @ x_sqrt
        return multiherm(result, self.backend_name)

    def log(self, x: Array, y: Array) -> Array:
        """Matrix logarithm: log_x(y) = sqrtm(x) log(X^{-1/2} Y X^{-1/2}) sqrtm(x)"""
        x_sqrt, x_isqrt = sqrtm_invsqrtm_psd(x, self.backend_name)

        # Argument for logarithm
        log_arg = x_isqrt @ y @ x_isqrt
        log_mat = logm_psd(log_arg, self.backend_name)

        # log_x(y) = sqrtm @ log_mat @ sqrtm
        result = x_sqrt @ log_mat @ x_sqrt
        return multiherm(result, self.backend_name)

    def dist(self, x: Array, y: Array) -> Array:
        """Riemannian distance via Cholesky + logarithm"""
        c = self.be.linalg.cholesky(x)
        c_inv = self.be.linalg.inv(c)
        logm = logm_psd(
            c_inv @ y @ self.be.swapaxes(c_inv, -1, -2).conj(), self.backend_name
        )
        return self.be.linalg.norm(logm, axis=(-2, -1))

    def rand(self, batch_shape: Tuple = ()) -> Array:
        """Random HPD via Gaussian QR decomposition.

        Generates Q from QR of random complex Gaussian, then X = Q @ diag(1+rand) @ Q^H.
        """
        # Eigenvalues uniformly in [1, 2]
        d_shape = batch_shape + (self.n,)
        r = sample_uniform(1, list(d_shape), self.backend_name, low=1.0, high=2.0)[0]
        # r shape is now d_shape, values in [1, 2]

        # Random orthogonal matrix via QR
        z_shape = batch_shape + (self.n, self.n)
        z = sample_standard_normal(1, list(z_shape), self.backend_name)[0]
        q, _ = self.be.linalg.qr(z)

        # X = Q @ diag(r) @ Q^H
        return self.be.einsum(
            "...ab,...bc,...cd->...ad",
            q,
            get_diagembed(self.backend_name, r),
            self.be.swapaxes(q, -1, -2).conj(),
        )

    def zerovec(self, x: Array) -> Array:
        """Zero tangent vector at point x."""
        return self.be.zeros_like(x)

    def randvec(self, x: Array) -> Array:
        """Random unit tangent vector at point x."""
        u = sample_standard_normal(1, list(x.shape), self.backend_name)[0]
        u = multiherm(
            u + 0j if is_complex(self.backend_name, x) else u, self.backend_name
        )
        u = u / self.norm(x, u)
        return u

    def retr(self, x: Array, u: Array) -> Array:
        """Second-order retraction: x + u + (1/2) u x^{-1} u"""
        return x + u + 0.5 * u @ self.be.linalg.solve(x, u)

    def transp(self, x1: Array, x2: Array, d: Array) -> Array:
        """Parallel transport via scaled geodesic (not standard)."""
        return self.proj(x2, d)


# ============================================================================
# Special Hermitian Positive Definite Manifold (unit determinant)
# ============================================================================


class SpecialHermitianPositiveDefinite(Manifold):
    """Manifold of Hermitian positive definite matrices with unit determinant.

    Points are (..., n, n) Hermitian PD with det(X) = 1.
    This is a totally geodesic submanifold of HPD.
    """

    def __init__(self, n: int, backend_name: Union[str, Backend] = "numpy"):
        """Initialize SHPD manifold.

        Parameters
        ----------
        n : int
            Size of matrices (n x n)
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        self.n = n
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self._hpd = HermitianPositiveDefinite(n, backend_name)

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Same as HPD"""
        return self._hpd.inner(x, u, v)

    def norm(self, x: Array, u: Array) -> Array:
        """Same as HPD"""
        return self._hpd.norm(x, u)

    def proj(self, x: Array, u: Array) -> Array:
        """Project onto SHPD tangent space: HPD proj minus trace term.

        Tangent space at x: {u : multiherm(u), tr(x^{-1} u) = 0}
        """
        u = multiherm(u, self.backend_name)
        # Compute trace of x^{-1} @ u
        trace_term = batched_trace(self.backend_name, self.be.linalg.solve(x, u))
        # Subtract (1/n) * tr * x to make trace zero
        return u - (1 / self.n) * self.be.real(trace_term)[..., None, None] * x

    def egrad2rgrad(self, x: Array, egrad: Array) -> Array:
        """Convert Euclidean to Riemannian gradient, then project."""
        rgrad = self._hpd.egrad2rgrad(x, egrad)
        return self.proj(x, rgrad)

    def exp(self, x: Array, u: Array) -> Array:
        """Exponential map with normalization."""
        exp_hpd = self._hpd.exp(x, u)
        # Normalize determinant to 1
        det_val = self.be.real(self.be.linalg.slogdet(exp_hpd)[1])
        scale = self.be.exp(-det_val / self.n)
        return scale[..., None, None] * exp_hpd

    def log(self, x: Array, y: Array) -> Array:
        """Same as HPD"""
        return self._hpd.log(x, y)

    def dist(self, x: Array, y: Array) -> Array:
        """Same as HPD"""
        return self._hpd.dist(x, y)

    def rand(self, batch_shape: Tuple = ()) -> Array:
        """Random SHPD: HPD random with det normalized to 1."""
        x = self._hpd.rand(batch_shape)
        det_val = self.be.real(self.be.linalg.slogdet(x)[1])
        scale = self.be.exp(-det_val / self.n)
        return scale[..., None, None] * x

    def zerovec(self, x: Array) -> Array:
        """Zero tangent vector at point x."""
        return self.be.zeros_like(x)

    def randvec(self, x: Array) -> Array:
        """Random unit tangent vector at point x."""
        u = sample_standard_normal(1, list(x.shape), self.backend_name)[0]
        u = u + 0j if is_complex(self.backend_name, x) else u
        u = self.proj(x, u)
        u = u / self.norm(x, u)
        return u

    def retr(self, x: Array, u: Array) -> Array:
        """Retraction: HPD second-order retr, then det-normalize."""
        r = self._hpd.retr(x, u)
        det_val = self.be.real(self.be.linalg.det(r))
        return r / det_val[..., None, None] ** (1.0 / self.n)

    def transp(self, x1: Array, x2: Array, d: Array) -> Array:
        """Transport with projection back to SHPD tangent space."""
        transp_d = self._hpd.transp(x1, x2, d)
        return self.proj(x2, transp_d)


# ============================================================================
# Strictly Positive Vectors Manifold
# ============================================================================


class StrictlyPositiveVectors(Manifold):
    """Manifold of strictly positive real vectors with Euclidean-Fisher metric.

    Points are (..., n) strictly positive real vectors.
    Inner product: <u, v>_x = sum(u_i * v_i / x_i^2)
    """

    def __init__(self, n: int, backend_name: Union[str, Backend] = "numpy"):
        """Initialize StrictlyPositiveVectors manifold.

        Parameters
        ----------
        n : int
            Dimension of vectors
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        self.n = n
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)

    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Inner product with Fisher metric: sum(u_i * v_i / x_i^2)"""
        return self.be.sum((u * v) / (x * x), axis=-1)

    def norm(self, x: Array, u: Array) -> Array:
        """Norm: sqrt(sum(u_i^2 / x_i^2))"""
        return self.be.sqrt(self.be.sum((u * u) / (x * x), axis=-1))

    def proj(self, x: Array, u: Array) -> Array:
        """Projection onto positive orthant: just return u (no constraint)"""
        return u

    def egrad2rgrad(self, x: Array, egrad: Array) -> Array:
        """Riemannian gradient: egrad * x^2 (metric inversion)"""
        return egrad * x * x

    def exp(self, x: Array, u: Array) -> Array:
        """Exponential map: element-wise exp(x) @ exp(u/x)"""
        return x * self.be.exp(u / x)

    def log(self, x: Array, y: Array) -> Array:
        """Logarithmic map: x * log(y/x)"""
        return x * self.be.log(y / x)

    def dist(self, x: Array, y: Array) -> Array:
        """Distance: norm(log(y/x))"""
        log_ratio = self.be.log(y / x)
        return self.be.linalg.norm(log_ratio, axis=-1)

    def rand(self, batch_shape: Tuple = ()) -> Array:
        """Random strictly positive vector: exp(randn)"""
        z_shape = batch_shape + (self.n,)
        return self.be.exp(
            sample_standard_normal(1, list(z_shape), self.backend_name)[0]
        )

    def zerovec(self, x: Array) -> Array:
        """Zero tangent vector at point x."""
        return self.be.zeros_like(x)

    def randvec(self, x: Array) -> Array:
        """Random unit tangent vector at point x."""
        u = sample_standard_normal(1, list(x.shape), self.backend_name)[0]
        u = u / self.norm(x, u)
        return u

    def retr(self, x: Array, u: Array) -> Array:
        """Retraction: same as exp (x * exp(u/x)), always positive."""
        return self.exp(x, u)


# ============================================================================
# Product Manifold
# ============================================================================


class ProductManifold(ABC):
    """Product of multiple manifolds.

    Points are tuples of points, one from each component manifold.
    Inner product is sum of component inner products (unweighted).
    """

    def __init__(
        self,
        manifolds: List[Manifold],
        backend_name: Union[str, Backend] = "numpy",
    ):
        """Initialize product manifold.

        Parameters
        ----------
        manifolds : List[Manifold]
            List of component manifolds
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        self.manifolds = manifolds
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)

    def inner(self, x: List[Array], u: List[Array], v: List[Array]) -> Array:
        """Inner product: sum of component inner products"""
        terms = [m.inner(x[i], u[i], v[i]) for i, m in enumerate(self.manifolds)]
        return self.be.sum(self.be.stack(terms, axis=-1), axis=-1)

    def norm(self, x: List[Array], u: List[Array]) -> Array:
        """Norm: sqrt(sum of component norms squared)"""
        norms = [m.norm(x[i], u[i]) for i, m in enumerate(self.manifolds)]
        return self.be.sqrt(
            self.be.sum(self.be.stack([n**2 for n in norms], axis=-1), axis=-1)
        )

    def proj(self, x: List[Array], u: List[Array]) -> List[Array]:
        """Project each component"""
        return [m.proj(x[i], u[i]) for i, m in enumerate(self.manifolds)]

    def exp(self, x: List[Array], u: List[Array]) -> List[Array]:
        """Exponential map: apply component-wise"""
        return [m.exp(x[i], u[i]) for i, m in enumerate(self.manifolds)]

    def retr(self, x: List[Array], u: List[Array]) -> List[Array]:
        """Retraction: apply component-wise"""
        return [m.retr(x[i], u[i]) for i, m in enumerate(self.manifolds)]

    def log(self, x: List[Array], y: List[Array]) -> List[Array]:
        """Logarithmic map: apply component-wise"""
        return [m.log(x[i], y[i]) for i, m in enumerate(self.manifolds)]

    def dist(self, x: List[Array], y: List[Array]) -> Array:
        """Distance: sqrt(sum of component distances squared)"""
        dists = [m.dist(x[i], y[i]) for i, m in enumerate(self.manifolds)]
        return self.be.sqrt(
            self.be.sum(self.be.stack([d**2 for d in dists], axis=-1), axis=-1)
        )

    def rand(self, batch_shape: Tuple = ()) -> List[Array]:
        """Random point: sample each component"""
        return [m.rand(batch_shape) for m in self.manifolds]

    def randvec(self, x: List[Array]) -> List[Array]:
        """Random tangent vector: sample each component"""
        return [m.randvec(x[i]) for i, m in enumerate(self.manifolds)]

    def zerovec(self, x: List[Array]) -> List[Array]:
        """Zero vector: zero in each component"""
        return [m.zerovec(x[i]) for i, m in enumerate(self.manifolds)]

    def egrad2rgrad(self, x: List[Array], egrad: List[Array]) -> List[Array]:
        """Riemannian gradient: apply component-wise"""
        return [m.egrad2rgrad(x[i], egrad[i]) for i, m in enumerate(self.manifolds)]

    def transp(self, x1: List[Array], x2: List[Array], d: List[Array]) -> List[Array]:
        """Transport: apply component-wise"""
        return [m.transp(x1[i], x2[i], d[i]) for i, m in enumerate(self.manifolds)]


# ============================================================================
# Weighted Product Manifold
# ============================================================================
class WeightedProductManifold(ProductManifold):
    """Product manifold with weighted metrics.

    Inner product: weighted sum of component inner products.
    Distance: sqrt(weighted sum of squared distances).
    """

    def __init__(
        self,
        manifolds: List[Manifold],
        weights: Optional[List[float]] = None,
        backend_name: Union[str, Backend] = "numpy",
    ):
        """Initialize weighted product manifold.

        Parameters
        ----------
        manifolds : List[Manifold]
            List of component manifolds
        weights : List[float], optional
            Weights for each component. Default: uniform (all ones).
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        super().__init__(manifolds, backend_name)
        if weights is None:
            weights = [1.0] * len(manifolds)
        self.weights = tuple(weights)

    def inner(self, x: List[Array], u: List[Array], v: List[Array]) -> Array:
        """Weighted inner product"""
        terms = [
            w * m.inner(x[i], u[i], v[i])
            for i, (m, w) in enumerate(zip(self.manifolds, self.weights))
        ]
        return self.be.sum(self.be.stack(terms, axis=-1), axis=-1)

    def dist(self, x: List[Array], y: List[Array]) -> Array:
        """Weighted distance"""
        dists = [
            w * m.dist(x[i], y[i]) ** 2
            for i, (m, w) in enumerate(zip(self.manifolds, self.weights))
        ]
        return self.be.sqrt(self.be.sum(self.be.stack(dists, axis=-1), axis=-1))

    def egrad2rgrad(self, x: List[Array], egrad: List[Array]) -> List[Array]:
        """Weighted Riemannian gradient: (1/w_k) * rgrad_k"""
        return [
            (1 / w) * m.egrad2rgrad(x[i], egrad[i])
            for i, (m, w) in enumerate(zip(self.manifolds, self.weights))
        ]


# ============================================================================
# Concrete Product Manifolds
# ============================================================================
class ScaledGaussianFIM(WeightedProductManifold):
    """Product of SHPD_d and StrictlyPositiveVectors_n for scaled Gaussian with Fisher metric.

    Parameters are (Sigma, tau) where Sigma is shape matrix and tau are texture parameters.
    Weights are (1/d, 1/n).
    """

    def __init__(
        self,
        d: int,
        n: int,
        backend_name: Union[str, Backend] = "numpy",
    ):
        """Initialize ScaledGaussianFIM manifold.

        Parameters
        ----------
        d : int
            Size of covariance matrix (d x d)
        n : int
            Number of texture parameters
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        self._d = d
        self._n = n
        manifolds = [
            SpecialHermitianPositiveDefinite(d, backend_name),
            StrictlyPositiveVectors(n, backend_name),
        ]
        weights = [1.0 / d, 1.0 / n]
        super().__init__(manifolds, weights, backend_name)


class KroneckerHermitianPositiveElliptical(WeightedProductManifold):
    """Kronecker product manifold for elliptical models.

    Covariance model: Sigma = A kron B where A is SHPD_a and B is HPD_b (sized such that ab = n).
    The Fisher metric includes an elliptical coupling term (alpha parameter).
    """

    def __init__(
        self,
        a: int,
        b: int,
        alpha: float = 1.0,
        backend_name: Union[str, Backend] = "numpy",
    ):
        """Initialize KroneckerHermitianPositiveElliptical manifold.

        Parameters
        ----------
        a : int
            Size of first Kronecker factor (a x a)
        b : int
            Size of second Kronecker factor (b x b)
        alpha : float, optional
            Elliptical coupling parameter. Default 1.0.
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        self._a = a
        self._b = b
        self._alpha = alpha
        manifolds = [
            SpecialHermitianPositiveDefinite(a, backend_name),
            HermitianPositiveDefinite(b, backend_name),
        ]
        weights = [alpha * b, alpha * a]
        super().__init__(manifolds, weights, backend_name)


class KroneckerHermitianPositiveScaledGaussian(WeightedProductManifold):
    """Product manifold for Kronecker structured scaled Gaussian model.

    Parameters are (A, B, tau) where A is SHPD_a, B is SHPD_b, and tau are texture parameters.
    Kronecker structure: Sigma = A kron B with det(A) = det(B) = 1.
    """

    def __init__(
        self,
        a: int,
        b: int,
        n: int,
        backend_name: Union[str, Backend] = "numpy",
    ):
        """Initialize KroneckerHermitianPositiveScaledGaussian manifold.

        Parameters
        ----------
        a : int
            Size of first Kronecker factor (a x a)
        b : int
            Size of second Kronecker factor (b x b)
        n : int
            Number of texture parameters
        backend_name : Union[str, Backend], optional
            Backend specification. Default "numpy".
        """
        self._a = a
        self._b = b
        self._n = n
        manifolds = [
            SpecialHermitianPositiveDefinite(a, backend_name),
            SpecialHermitianPositiveDefinite(b, backend_name),
            StrictlyPositiveVectors(n, backend_name),
        ]
        weights = [b / (a * b), a / (a * b), 1.0 / n]
        super().__init__(manifolds, weights, backend_name)
