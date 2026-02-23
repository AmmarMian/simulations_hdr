# SAR Specific detection tools

import sys
from pathlib import Path
import os
from typing import Optional

from torch import slogdet

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.backend import get_backend_module, get_data_on_device, batched_trace, make_writable_copy, Array
from src.detection import Detector
from src.estimation import SCMEstimator, TylerEstimator, StudentTEstimator


# Gaussian GLRT
# -------------
class GaussianGLRT(Detector):
    def __init__(self, backend_name: str = "numpy") -> None:
        """Gaussian GLRT on multiple covariance equality testing

        Parameters
        -----------
        backend_name: str
            backend to use. Choices are : numpy, torch-cpu, torch-cuda
        """
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)

    def compute(self, X: Array, *args, **kwargs) -> Array:
        """Compute the Gaussian GLRT statistic.

        Parameters
        ----------
        X: Array (torch or numpy)
            data to compute statistic on. Shape (..., n_times, n_samples, n_features),
            where ... are batches dimensions.

        Returns
        -------
        Array (torch or numpy)
            result of test statistic over all batches dimensions.

        """
        X = get_data_on_device(X, self.backend_name)
        covariances = (
            self.be.einsum("...ab,...bc->...ac", self.be.swapaxes(X, -1, -2).conj(), X)
            / X.shape[-2]
        )
        cov_mean = covariances.mean(axis=-3)
        log_det_cov_mean = self.be.linalg.slogdet(cov_mean)[1]

        # This solution with vectorized operations seems more efficient only on bigger matrices
        if X.shape[-1] >= 50:
            slogdet_t = self.be.linalg.slogdet(covariances)[1]
            res = log_det_cov_mean * X.shape[-3] - slogdet_t.sum(-1)
        else:
            res = log_det_cov_mean * X.shape[-3]
            for t in range(X.shape[-3]):
                cov_t = covariances[..., t, :, :]
                log_det_cov_t = self.be.linalg.slogdet(cov_t)[1]
                res -= log_det_cov_t
        return res


# 2-step Detectors
# ----------------
class TwoStepStudentGaussianGLRT(Detector):
    def __init__(
        self,
        backend_name: str = "numpy",
        df: float = 3,
        normalization: Optional[str] = None,
        tol: float = 1e-4,
        iter_max: int = 5,
        verbosity: bool = False,
        debug: bool = False,
        iteration_chunk_size: Optional[int] = None,
        init: Optional[Array] = None,
    ) -> None:
        """Gaussian GLRT on multiple covariance equality testing but with a Student-t
        estimator of covariances.

        Parameters
        -----------
        backend_name: str
            backend to use. Choices are : numpy, torch-cpu, torch-cuda

        df: float
            Degrees of freedom of Student-t distribution. By default 3 (Heavy tailed).

        normalization : str or None, optional
            Normalization method. Options: 'trace', 'det', 'diag', or None.
            By default None.

        tol : float, optional
            Convergence tolerance for the fixed-point algorithm. By default 1e-4.

        iter_max : int, optional
            Maximum number of iterations. By default 5.

        verbosity: bool
            Whether to show progress of matrix estimation. By default False.

        debug: bool
            Print debug information during iterations. By default False.

        iteration_chunk_size : int or None, optional
            Process matrices in chunks of this size within each iteration to reduce
            memory usage. Useful for large batches on GPU. By default None (no chunking).

        init : Array or None, optional
            Initial covariance estimate. If None, uses identity matrix. By default None.

        """
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.df = df
        self.normalization = normalization
        self.tol = tol
        self.iter_max = iter_max
        self.verbosity = verbosity
        self.debug = debug
        self.iteration_chunk_size = iteration_chunk_size
        self.init = init

        self.estimator = StudentTEstimator(
            df=self.df,
            normalization=self.normalization,
            backend_name=self.backend_name,
            tol=self.tol,
            iter_max=self.iter_max,
            verbosity=self.verbosity,
            debug=self.debug,
            iteration_chunk_size=self.iteration_chunk_size,
            init=self.init,
        )

    def compute(self, X: Array, *args, **kwargs) -> Array:
        """Compute the Gaussian GLRT statistic.

        Parameters
        ----------
        X: Array (torch or numpy)
            data to compute statistic on. Shape (..., n_times, n_samples, n_features),
            where ... are batches dimensions.

        Returns
        -------
        Array (torch or numpy)
            result of test statistic over all batches dimensions.

        """
        X = get_data_on_device(X, self.backend_name)
        if self.verbosity:
            print("Computing covariances for all t")
        covariances = self.estimator.compute(X)
        if self.verbosity:
            print("Computing covariance under H0")
        cov_mean = self.estimator.compute(
            self.be.reshape(X, (*X.shape[:-3], X.shape[-3] * X.shape[-2], X.shape[-1]))
        )
        log_det_cov_mean = self.be.linalg.slogdet(cov_mean)[1]
        del cov_mean

        # This solution with vectorized operations seems more efficient only on bigger matrices
        if X.shape[-1] >= 50:
            slogdet_t = self.be.linalg.slogdet(covariances)[1]
            res = log_det_cov_mean * X.shape[-3] - slogdet_t.sum(-1)
        else:
            res = log_det_cov_mean * X.shape[-3]
            for t in range(X.shape[-3]):
                cov_t = covariances[..., t, :, :]
                log_det_cov_t = self.be.linalg.slogdet(cov_t)[1]
                res -= log_det_cov_t
        return res


class TwoStepTylerGaussianGLRT(Detector):
    def __init__(
        self,
        backend_name: str = "numpy",
        tol: float = 1e-4,
        iter_max: int = 50,
        verbosity: bool = False,
        debug: bool = False,
        iteration_chunk_size: Optional[int] = None,
        init: Optional[Array] = None,
    ) -> None:
        """Gaussian GLRT on multiple covariance equality testing but with Tyler's
        M-estimator of covariances. Trace normalization is always applied, as it
        is required for Tyler's estimator to converge (det normalization is not
        meaningful for a shape-only estimator).

        Parameters
        -----------
        backend_name: str
            backend to use. Choices are : numpy, torch-cpu, torch-cuda

        tol : float, optional
            Convergence tolerance for the fixed-point algorithm. By default 1e-4.

        iter_max : int, optional
            Maximum number of iterations. By default 50.

        verbosity: bool
            Whether to show progress of matrix estimation. By default False.

        debug: bool
            Print debug information during iterations. By default False.

        iteration_chunk_size : int or None, optional
            Process matrices in chunks of this size within each iteration to reduce
            memory usage. Useful for large batches on GPU. By default None (no chunking).

        init : Array or None, optional
            Initial covariance estimate. If None, uses identity matrix. By default None.

        """
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.tol = tol
        self.iter_max = iter_max
        self.verbosity = verbosity
        self.debug = debug
        self.iteration_chunk_size = iteration_chunk_size
        self.init = init

        self.estimator = TylerEstimator(
            normalization="trace",
            backend_name=self.backend_name,
            tol=self.tol,
            iter_max=self.iter_max,
            verbosity=self.verbosity,
            debug=self.debug,
            iteration_chunk_size=self.iteration_chunk_size,
            init=self.init,
        )

    def compute(self, X: Array, *args, **kwargs) -> Array:
        """Compute the Tyler-based GLRT statistic.

        Parameters
        ----------
        X: Array (torch or numpy)
            data to compute statistic on. Shape (..., n_times, n_samples, n_features),
            where ... are batches dimensions.

        Returns
        -------
        Array (torch or numpy)
            result of test statistic over all batches dimensions.

        """
        X = get_data_on_device(X, self.backend_name)
        if self.verbosity:
            print("Computing covariances for all t")
        covariances = self.estimator.compute(X)
        if self.verbosity:
            print("Computing covariance under H0")
        cov_mean = self.estimator.compute(
            self.be.reshape(X, (*X.shape[:-3], X.shape[-3] * X.shape[-2], X.shape[-1]))
        )
        log_det_cov_mean = self.be.linalg.slogdet(cov_mean)[1]
        del cov_mean

        # This solution with vectorized operations seems more efficient only on bigger matrices
        if X.shape[-1] >= 50:
            slogdet_t = self.be.linalg.slogdet(covariances)[1]
            res = log_det_cov_mean * X.shape[-3] - slogdet_t.sum(-1)
        else:
            res = log_det_cov_mean * X.shape[-3]
            for t in range(X.shape[-3]):
                cov_t = covariances[..., t, :, :]
                log_det_cov_t = self.be.linalg.slogdet(cov_t)[1]
                res -= log_det_cov_t
        return res


# MatAndText Tyler fixed-point estimator
# ---------------------------------------
def _tyler_matandtext_fixed_point(
    X: Array,
    tol: float = 1e-4,
    iter_max: int = 50,
    backend_name: str = "numpy",
) -> Array:
    """Fixed-point Tyler estimator for the MatAndText model.

    Jointly estimates a single shape matrix Σ with textures τ_n shared across
    all T time steps. This is the H0 estimator for the deterministic compound
    Gaussian GLRT: the model assumes x_{n,t} = sqrt(τ_n) * A * w_{n,t} where
    τ_n is the same deterministic texture for sample n at every date.

    Algorithm per iteration:
        τ_n ← sum_t  x_{n,t}^H Σ^{-1} x_{n,t}          (shared texture)
        Σ   ← (p/N) * sum_t sum_n x_{n,t} x_{n,t}^H / τ_n
        Σ   ← p * Σ / Tr(Σ)                              (trace normalization)

    Parameters
    ----------
    X : Array of shape (batch..., T, N, p)
    tol : float
        Convergence tolerance on relative Frobenius norm. Default 1e-4.
    iter_max : int
        Maximum number of iterations. Default 50.
    backend_name : str

    Returns
    -------
    Array of shape (batch..., p, p)
        Estimated shape matrix Σ (trace-normalized).
    """
    be = get_backend_module(backend_name)
    n_times, n_samples, n_features = X.shape[-3], X.shape[-2], X.shape[-1]

    # Initialize Σ = I broadcast to (batch..., p, p)
    cov_shape = X.shape[:-3] + (n_features, n_features)
    eye = be.eye(n_features, dtype=X.dtype)
    eye = get_data_on_device(eye, backend_name)
    Sigma = make_writable_copy(backend_name, be.broadcast_to(eye, cov_shape))

    for _ in range(iter_max):
        iSigma = be.linalg.inv(Sigma)  # (batch..., p, p)

        # Shared textures: τ_n = sum_t x_{n,t}^H Σ^{-1} x_{n,t}
        # iSigma is (batch..., p, p); insert T dim to broadcast with X (batch..., T, N, p)
        temp = X.conj() @ iSigma[..., None, :, :]  # (batch..., T, N, p)
        tau = be.real(temp * X).sum(-1).sum(-2)     # sum over p then T → (batch..., N)

        # X_scaled[:,n,:] = X[:,n,:] / sqrt(τ_n), broadcast over T and p
        X_scaled = X / be.sqrt(tau)[..., None, :, None]  # (batch..., T, N, p)

        # Σ_new = (p/N) * sum_t X_scaled_t^H @ X_scaled_t
        # (batch..., T, p, N) @ (batch..., T, N, p) = (batch..., T, p, p), sum over T
        Sigma_new = (n_features / n_samples) * (
            be.swapaxes(X_scaled, -1, -2).conj() @ X_scaled
        ).sum(-3)

        # Trace normalization: Tr(Σ_new) = p
        trace = batched_trace(backend_name, Sigma_new)  # (batch...,)
        Sigma_new = Sigma_new * (n_features / trace[..., None, None])

        # Convergence: relative Frobenius norm over all batch elements
        diff = Sigma_new - Sigma
        delta = be.sqrt(
            be.real(be.einsum("...ij,...ij->...", diff.conj(), diff))
            / be.real(be.einsum("...ij,...ij->...", Sigma.conj(), Sigma))
        )
        Sigma = Sigma_new
        if be.all(delta <= tol):
            break

    return Sigma


# Deterministic Compound Gaussian GLRT
# -------------------------------------
class DeterministicCompoundGaussianGLRT(Detector):
    def __init__(
        self,
        backend_name: str = "numpy",
        tol: float = 1e-4,
        iter_max: int = 50,
        verbosity: bool = False,
        debug: bool = False,
        iteration_chunk_size: Optional[int] = None,
        init: Optional[Array] = None,
    ) -> None:
        """GLRT for testing equality of scale and shape in a deterministic
        compound Gaussian (SIRV) model.

        Under H0, a single shape matrix Σ_0 and per-sample textures τ_n are
        shared across all T dates (MatAndText model). Under H1, each date t has
        its own shape Σ_t and textures τ_{n,t} (independent Tyler estimates).

        The log-statistic is:
            λ = T·N·log|Σ_0| - N·Σ_t log|Σ_t|
              + T·p·Σ_n log(τ̂_0n) - p·Σ_t Σ_n log(τ̂_{tn})

        Reference: Mian et al., WCCM 2019.

        Parameters
        ----------
        backend_name : str
            Backend to use. Choices are: numpy, torch-cpu, torch-cuda.
        tol : float
            Convergence tolerance for Tyler fixed-point algorithms. Default 1e-4.
        iter_max : int
            Maximum iterations for Tyler algorithms. Default 50.
        verbosity : bool
            Print progress. Default False.
        debug : bool
            Print debug information. Default False.
        iteration_chunk_size : int or None
            Chunk size for the per-date Tyler estimator (H1). Default None.
        init : Array or None
            Initial covariance estimate. Default None (identity).
        """
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.tol = tol
        self.iter_max = iter_max
        self.verbosity = verbosity
        self.debug = debug
        self.iteration_chunk_size = iteration_chunk_size
        self.init = init

        # Standard Tyler for per-date shape matrices (H1)
        # TylerEstimator.compute(X) treats (batch..., T) as batch dims
        self.tyler_estimator = TylerEstimator(
            normalization="trace",
            backend_name=self.backend_name,
            tol=self.tol,
            iter_max=self.iter_max,
            verbosity=False,
            debug=self.debug,
            iteration_chunk_size=self.iteration_chunk_size,
            init=self.init,
        )

    def compute(self, X: Array, *args, **kwargs) -> Array:
        """Compute the deterministic compound Gaussian GLRT statistic.

        Parameters
        ----------
        X : Array of shape (..., T, N, p)

        Returns
        -------
        Array of shape (...,)
            Log-likelihood ratio statistic per spatial position.
        """
        X = get_data_on_device(X, self.backend_name)
        n_times, n_samples, n_features = X.shape[-3], X.shape[-2], X.shape[-1]

        # --- H0: MatAndText Tyler ---
        if self.verbosity:
            print("Computing Σ_0 via MatAndText Tyler (H0)...")
        Sigma_0 = _tyler_matandtext_fixed_point(
            X, tol=self.tol, iter_max=self.iter_max, backend_name=self.backend_name
        )  # (batch..., p, p)
        iSigma_0 = self.be.linalg.inv(Sigma_0)  # (batch..., p, p)

        # Texture under H0: τ_0n = (1/T) * sum_t x_{n,t}^H Σ_0^{-1} x_{n,t}
        # iSigma_0 is (batch..., p, p); insert T dim to broadcast with X (batch..., T, N, p)
        temp_0 = X.conj() @ iSigma_0[..., None, :, :]           # (batch..., T, N, p)
        tau_0 = self.be.real(temp_0 * X).sum(-1).sum(-2) / n_times  # (batch..., N)

        log_det_Sigma_0 = self.be.real(self.be.linalg.slogdet(Sigma_0)[1])  # (batch...,)

        # --- H1: Standard Tyler per date ---
        if self.verbosity:
            print("Computing Σ_t via standard Tyler (H1)...")
        Sigma_t = self.tyler_estimator.compute(X)   # (batch..., T, p, p)
        iSigma_t = self.be.linalg.inv(Sigma_t)      # (batch..., T, p, p)

        # Texture under H1: τ_{t,n} = x_{n,t}^H Σ_t^{-1} x_{n,t}
        # (batch..., T, N, p) @ (batch..., T, p, p) → (batch..., T, N, p)
        temp_t = X.conj() @ iSigma_t
        tau_t = self.be.real(temp_t * X).sum(-1)  # (batch..., T, N)

        log_det_Sigma_t = self.be.real(self.be.linalg.slogdet(Sigma_t)[1])  # (batch..., T)

        # --- Log-quadratic terms ---
        log_tau_0_sum = self.be.log(self.be.abs(tau_0)).sum(-1)          # (batch...,)
        log_tau_t_sum = self.be.log(self.be.abs(tau_t)).sum(-1).sum(-1)  # (batch...,)

        # λ = T·N·log|Σ_0| - N·Σ_t log|Σ_t| + T·p·Σ_n log(τ_0n) - p·Σ_t Σ_n log(τ_tn)
        return (
            n_times * n_samples * log_det_Sigma_0
            - n_samples * log_det_Sigma_t.sum(-1)
            + n_times * n_features * log_tau_0_sum
            - n_features * log_tau_t_sum
        )
