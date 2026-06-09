# Online scaled Gaussian estimation
# Author: Ammar Mian

import logging
from typing import Callable, List, Optional, Tuple, Union

from .backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
    to_scalar,
)
from .estimation import natural_gradient_scaled_gaussian, _rgrad_scaled_gaussian
from .manifolds import ScaledGaussianFIM

logger = logging.getLogger(__name__)


class _InverseTStep:
    """Inverse-t step size schedule: ``lr / t`` (1-indexed).

    Implemented as a picklable callable class instead of a lambda so that
    estimator instances can be serialised with ``pickle`` / ``joblib``.
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def __call__(self, t: int) -> float:
        return self.lr / t


# -----------------------------------------------------------------------
# Online Scaled Gaussian Natural Gradient Estimator
# -----------------------------------------------------------------------
def online_natural_gradient_scaled_gaussian(
    X_batches: Array,
    lr: float = 1.0,
    step_fn=None,
    verbosity: bool = False,
    backend_name: Union[str, Backend] = "numpy",
) -> Tuple[Array, Array, List]:
    """Online Riemannian gradient descent for scaled Gaussian MLE (batched spatial dims).

    Processes time batches sequentially. The first batch is used to warm-start
    (Sigma, tau) via the full batch estimator. Subsequent batches each
    apply one Riemannian gradient step with decreasing step size so early
    batches dominate and later ones refine the estimate.

    Parameters
    ----------
    X_batches : Array of shape (n_batches, ..., n_samples, n_features)
        Sequence of data batches. (...) are spatial batch dims.
        Each batch shares the same n_samples and n_features.
    lr : float
        Base learning rate. Step at batch t is step_fn(t) or lr/t
        (1-indexed, since batch 0 is the warm-start).
    step_fn : callable (int -> float) or None
        Custom step size schedule. Receives 1-based batch index.
        Default: lambda t: lr / t
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

    if step_fn is None:
        step_fn = _InverseTStep(lr)

    manifold = ScaledGaussianFIM(n_features, n_samples, backend_name=backend_name)

    # Warm-start: run full batch estimator on first batch
    Sigma, tau = natural_gradient_scaled_gaussian(
        X_batches[0], backend_name=backend_name
    )
    history = [(Sigma, tau)]

    if verbosity:
        logger.debug("Warm-started from batch 0")
        logger.debug("%-7s %-10s %-12s %-12s", "Batch", "step", "||rS||", "||rt||")
        logger.debug("-" * 44)

    for t in range(1, n_batches):
        X_t = X_batches[t]  # (..., n_samples, n_features)
        r_Sigma, r_tau = _rgrad_scaled_gaussian(X_t, Sigma, tau, manifold, be)
        step = step_fn(t)
        tau_v = tau[..., 0]  # (..., n_samples) for manifold
        Sigma, tau_v_new = manifold.retr(
            [Sigma, tau_v],
            [-step * r_Sigma, -step * r_tau[..., 0]],
        )
        tau = tau_v_new[..., None]  # (..., n_samples, 1)
        history.append((Sigma, tau))

        if verbosity:
            nrS = to_scalar(be.real(be.sum(r_Sigma * r_Sigma.conj()))) ** 0.5
            nrt = to_scalar(be.real(be.sum(r_tau * r_tau))) ** 0.5
            logger.debug("%-7d %-10.4f %-12.4e %-12.4e", t, step, nrS, nrt)

    return Sigma, tau, history


class OnlineScaledGaussianEstimator:
    """Stateful online estimator for scaled Gaussian model.

    Maintains a running estimate of (Sigma, tau) updated each time a new
    batch of data arrives via .update(). The first call to .update() runs
    the full batch estimator to warm-start (Sigma, tau); subsequent calls
    each apply one Riemannian gradient step with step lr/t.

    Parameters
    ----------
    n_features : int
        Dimension of each observation.
    n_samples : int
        Number of spatial positions per batch (fixed across all batches).
    lr : float
        Base learning rate. Step at call t (1-indexed) is step_fn(t) or lr/t.
    step_fn : callable (int -> float) or None
        Custom step size schedule. Receives 1-based batch index.
    backend_name : str or Backend
    """

    def __init__(
        self,
        n_features: int,
        n_samples: int,
        lr: float = 1.0,
        step_fn=None,
        backend_name: Union[str, Backend] = "numpy",
    ):
        self.n_features = n_features
        self.n_samples = n_samples
        self.lr = lr
        self.step_fn = step_fn if step_fn is not None else _InverseTStep(lr)
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
        if self._t == 0:
            self.Sigma, self.tau = natural_gradient_scaled_gaussian(
                X, backend_name=self.backend_name
            )
            self._t = 1
            return self.Sigma, self.tau

        r_Sigma, r_tau = _rgrad_scaled_gaussian(
            X, self.Sigma, self.tau, self._manifold, self.be
        )
        step = self.step_fn(self._t)
        tau_v = self.tau[..., 0]  # (..., n_samples) for manifold
        self.Sigma, tau_v_new = self._manifold.retr(
            [self.Sigma, tau_v],
            [-step * r_Sigma, -step * r_tau[..., 0]],
        )
        self.tau = tau_v_new[..., None]  # (..., n_samples, 1)
        self._t += 1
        return self.Sigma, self.tau

    def reset(self):
        """Reset to uninitialised state (next update will warm-start again)."""
        self._t = 0
        self.Sigma = None
        self.tau = None
