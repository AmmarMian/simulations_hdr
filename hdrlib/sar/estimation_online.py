# Online scaled Gaussian estimation
# Author: Ammar Mian

import logging
from typing import List, Tuple, Union

from ..core.backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
    make_writable_copy,
    to_scalar,
)
from ..core.estimation import (
    natural_gradient_scaled_gaussian,
    _rgrad_scaled_gaussian,
    _armijo_backtracking_scaled_gaussian,
)
from .estimation_kronecker import (
    kronecker_mm_h0,
    _rgrad_kronecker_scaled_gaussian,
    _armijo_backtracking_kronecker_scaled_gaussian,
)
from ..core.manifolds import ScaledGaussianFIM, KroneckerHermitianPositiveScaledGaussian

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Online Scaled Gaussian Natural Gradient Estimator
# -----------------------------------------------------------------------
def online_natural_gradient_scaled_gaussian(
    X_batches: Array,
    alpha_0: float = 0.1,
    armijo_c: float = 1e-4,
    armijo_rho: float = 0.5,
    armijo_max_backtracks: int = 5,
    verbosity: bool = False,
    backend_name: Union[str, Backend] = "numpy",
) -> Tuple[Array, Array, List]:
    """Online Riemannian gradient descent for scaled Gaussian MLE (batched spatial dims).

    Processes time batches sequentially. The first batch warm-starts (Sigma, tau)
    via the full batch estimator. Subsequent batches each apply one Riemannian
    gradient step with Armijo backtracking.

    Parameters
    ----------
    X_batches : Array of shape (n_batches, ..., n_samples, n_features)
        Sequence of data batches. (...) are spatial batch dims.
        Each batch shares the same n_samples and n_features.
    alpha_0 : float
        Initial step size for Armijo backtracking. Default 0.1.
    armijo_c : float
        Sufficient decrease constant. Default 1e-4.
    armijo_rho : float
        Step reduction factor. Default 0.5.
    armijo_max_backtracks : int
        Maximum number of backtracking steps. Default 5.
    verbosity : bool
        Print per-batch info.
    backend_name : str or Backend

    Returns
    -------
    Sigma : Array of shape (..., n_features, n_features)
    tau   : Array of shape (..., n_samples, 1)
    history : list of (Sigma, tau) snapshots after each batch
    """
    be = get_backend_module(backend_name)
    X_batches = get_data_on_device(X_batches, backend_name)
    n_batches = X_batches.shape[0]
    n_samples = X_batches.shape[-2]
    n_features = X_batches.shape[-1]

    manifold = ScaledGaussianFIM(n_features, n_samples, backend_name=backend_name)

    # Warm-start: run full batch estimator on first batch
    Sigma, tau = natural_gradient_scaled_gaussian(
        X_batches[0], backend_name=backend_name
    )
    history = [(Sigma, tau)]

    if verbosity:
        logger.debug("Warm-started from batch 0")
        logger.debug("%-7s %-12s %-12s", "Batch", "||rS||", "||rt||")
        logger.debug("-" * 34)

    for t in range(1, n_batches):
        X_t = X_batches[t]  # (..., n_samples, n_features)
        r_Sigma, r_tau = _rgrad_scaled_gaussian(X_t, Sigma, tau, manifold, be)
        _, Sigma, tau = _armijo_backtracking_scaled_gaussian(
            X_t, Sigma, tau, r_Sigma, r_tau, manifold, be,
            alpha_0=alpha_0, c=armijo_c, rho=armijo_rho,
            max_backtracks=armijo_max_backtracks,
            backend_name=backend_name,
        )
        history.append((Sigma, tau))

        if verbosity:
            nrS = to_scalar(be.real(be.sum(r_Sigma * r_Sigma.conj()))) ** 0.5
            nrt = to_scalar(be.real(be.sum(r_tau * r_tau))) ** 0.5
            logger.debug("%-7d %-12.4e %-12.4e", t, nrS, nrt)

    return Sigma, tau, history


class OnlineScaledGaussianEstimator:
    """Stateful online estimator for scaled Gaussian model.

    Maintains a running estimate of (Sigma, tau) updated each time a new
    batch of data arrives via .update(). The first call to .update() runs
    the full batch estimator to warm-start (Sigma, tau); subsequent calls
    each apply one Riemannian gradient step with Armijo backtracking.

    Parameters
    ----------
    n_features : int
        Dimension of each observation.
    n_samples : int
        Number of spatial positions per batch (fixed across all batches).
    alpha_0 : float
        Initial step size for Armijo backtracking. Default 0.1.
    armijo_c : float
        Sufficient decrease constant. Default 1e-4.
    armijo_rho : float
        Step reduction factor. Default 0.5.
    armijo_max_backtracks : int
        Maximum backtracking steps per update. Default 5.
    backend_name : str or Backend
    """

    def __init__(
        self,
        n_features: int,
        n_samples: int,
        alpha_0: float = 0.1,
        armijo_c: float = 1e-4,
        armijo_rho: float = 0.5,
        armijo_max_backtracks: int = 5,
        iter_max: int = 200,
        tol: float = 1e-8,
        backend_name: Union[str, Backend] = "numpy",
    ):
        self.n_features = n_features
        self.n_samples = n_samples
        self.alpha_0 = alpha_0
        self.armijo_c = armijo_c
        self.armijo_rho = armijo_rho
        self.armijo_max_backtracks = armijo_max_backtracks
        self.iter_max = iter_max
        self.tol = tol
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self._manifold = ScaledGaussianFIM(
            n_features, n_samples, backend_name=backend_name
        )
        self._t = 0

    def update(self, X: Array) -> Tuple[Array, Array]:
        """Update estimate with a new batch of data (supports spatial batch dims).

        The first call warm-starts (Sigma, tau) from this batch using the
        full batch estimator. Subsequent calls apply one gradient step.

        Parameters
        ----------
        X : Array of shape (..., n_samples, n_features)
            New data batch. (...) are spatial batch dimensions.

        Returns
        -------
        Sigma : Array of shape (..., n_features, n_features)
            Current shape estimate
        tau   : Array of shape (..., n_samples, 1)
            Current texture estimate
        """
        X = get_data_on_device(X, self.backend_name)
        assert X.shape[-2] == self.n_samples, (
            f"n_samples mismatch in OnlineScaledGaussianEstimator.update: "
            f"expected {self.n_samples}, got {X.shape[-2]}"
        )
        if self._t == 0:
            self.Sigma, self.tau = natural_gradient_scaled_gaussian(
                X, iter_max=self.iter_max, tol=self.tol, backend_name=self.backend_name
            )
            self._t = 1
            return self.Sigma, self.tau

        r_Sigma, r_tau = _rgrad_scaled_gaussian(
            X, self.Sigma, self.tau, self._manifold, self.be
        )
        _, self.Sigma, self.tau = _armijo_backtracking_scaled_gaussian(
            X, self.Sigma, self.tau, r_Sigma, r_tau, self._manifold, self.be,
            alpha_0=self.alpha_0, c=self.armijo_c, rho=self.armijo_rho,
            max_backtracks=self.armijo_max_backtracks,
            backend_name=self.backend_name,
        )
        self._t += 1
        return self.Sigma, self.tau

    def reset(self):
        """Reset to uninitialised state (next update will warm-start again)."""
        self._t = 0
        self.Sigma = None
        self.tau = None


# -----------------------------------------------------------------------
# Online Kronecker Scaled Gaussian Estimator
# -----------------------------------------------------------------------
class OnlineKroneckerEstimator:
    """Stateful online estimator for Kronecker structured scaled Gaussian model.

    Maintains running estimates of (A, B, tau) on the product manifold
    SHPD(a) x SHPD(b) x StrictlyPositiveVectors(N) with Fisher Information
    Metric. The first call to .update() warm-starts from the Kronecker MM
    algorithm; subsequent calls each apply one Riemannian natural gradient
    step with Armijo backtracking.

    Parameters
    ----------
    a, b : int
        Sizes of the Kronecker factors (p = a*b).
    n_samples : int
        Number of samples per batch (fixed across all batches).
    alpha_0 : float
        Initial step size for Armijo backtracking. Default 0.1.
    armijo_c : float
        Sufficient decrease constant. Default 1e-4.
    armijo_rho : float
        Step reduction factor. Default 0.5.
    armijo_max_backtracks : int
        Maximum backtracking steps per update. Default 5.
    iter_max : int
        Maximum MM iterations for the warm-start. Default 30.
    tol : float
        Convergence tolerance for the warm-start MM. Default 1e-4.
    backend_name : str or Backend
    """

    def __init__(
        self,
        a: int,
        b: int,
        n_samples: int,
        alpha_0: float = 0.1,
        armijo_c: float = 1e-4,
        armijo_rho: float = 0.5,
        armijo_max_backtracks: int = 5,
        iter_max: int = 30,
        tol: float = 1e-4,
        backend_name: Union[str, Backend] = "numpy",
        max_exp_tau: float = 3.0,
    ):
        self.a = a
        self.b = b
        self.n_samples = n_samples
        self.alpha_0 = alpha_0
        self.armijo_c = armijo_c
        self.armijo_rho = armijo_rho
        self.armijo_max_backtracks = armijo_max_backtracks
        self.iter_max = iter_max
        self.tol = tol
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self._manifold = KroneckerHermitianPositiveScaledGaussian(
            a, b, n_samples, backend_name=backend_name
        )
        self.max_exp_tau = max_exp_tau
        self._t = 0
        self.A = None
        self.B = None
        self.tau = None

    def update(self, X: Array) -> Tuple[Array, Array, Array]:
        """Update estimate with a new batch of data.

        The first call warm-starts (A, B, tau) via the Kronecker MM algorithm.
        Subsequent calls each apply one Riemannian natural gradient step with
        Armijo backtracking on the product manifold SHPD(a) x SHPD(b) x SPV(N).

        Parameters
        ----------
        X : Array of shape (..., n_samples, p) where p = a*b

        Returns
        -------
        A : Array of shape (..., a, a)
        B : Array of shape (..., b, b)
        tau : Array of shape (..., n_samples, 1)
        """
        X = get_data_on_device(X, self.backend_name)
        if self._t == 0:
            # Warm-start: run Kronecker MM on this batch (T=1)
            X_t = X[..., None, :, :]  # (..., 1, N, p) — add T dimension
            self.A, self.B, tau_flat = kronecker_mm_h0(
                X_t, self.a, self.b,
                tol=self.tol, iter_max=self.iter_max,
                backend_name=self.backend_name,
            )
            self.tau = tau_flat[..., None]  # (..., N, 1)
            self._t = 1
            return self.A, self.B, self.tau

        r_A, r_B, r_tau = _rgrad_kronecker_scaled_gaussian(
            X, self.A, self.B, self.tau,
            self._manifold, self.be,
            self.a, self.b, self.backend_name,
        )
        alpha_t = self.alpha_0 / self._t
        _, self.A, self.B, self.tau = _armijo_backtracking_kronecker_scaled_gaussian(
            X, self.A, self.B, self.tau,
            r_A, r_B, r_tau,
            self._manifold, self.be,
            self.a, self.b,
            alpha_0=alpha_t, c=self.armijo_c, rho=self.armijo_rho,
            max_backtracks=self.armijo_max_backtracks,
            backend_name=self.backend_name,
            alpha_0_tau=alpha_t,
            max_exp_tau=self.max_exp_tau,
        )
        self._t += 1
        return self.A, self.B, self.tau

    def reset(self):
        """Reset to uninitialised state (next update will warm-start again)."""
        self._t = 0
        self.A = None
        self.B = None
        self.tau = None
