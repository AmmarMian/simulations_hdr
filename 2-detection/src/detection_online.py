# Online change detection for scaled Gaussian model
# Author: Ammar Mian

from dataclasses import dataclass
from typing import Union, Tuple, Any

from .backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
)
from .detection import OnlineDetector
from .estimation import ScaledGaussianNaturalGradientEstimator
from .estimation_online import OnlineScaledGaussianEstimator


# -----------------------------------------------------------------------
# Online DCG Detector (Date-Class Gaussian with Change Detection)
# -----------------------------------------------------------------------
@dataclass
class OnlineDCGDetectorState:
    """State for OnlineDCGDetector.

    Attributes
    ----------
    h0_estimator : OnlineScaledGaussianEstimator
        Online estimator for H0 (pooled across all dates)
    log_likelihood_h0 : Array
        Accumulated log-likelihood under H0
    log_likelihood_h1_total : Array
        Sum of per-date H1 log-likelihoods
    n_times : int
        Number of time steps processed
    """

    h0_estimator: OnlineScaledGaussianEstimator
    log_likelihood_h0: Array
    log_likelihood_h1_total: Array
    n_times: int


class OnlineDCGDetector(OnlineDetector):
    """Online Date-Class Gaussian change point detector.

    Uses a generalized likelihood ratio test (GLRT) to detect changes in the
    scaled Gaussian model parameters over time.

    H0: Single (Sigma, tau) estimate pooled across all dates
    H1: Separate (Sigma_t, tau_t) estimate per date

    Test statistic: GLRT = 2 * [log L(data | H0_pooled) - sum_t log L(data_t | H1_t)]

    Parameters
    ----------
    backend_name : str or Backend
        Backend specification. By default 'numpy'.
    h0_lr : float
        Learning rate for online H0 estimator. By default 1.0.
    **h1_kwargs
        Additional keyword arguments for H1 estimator (e.g., iter_max, tol).
    """

    def __init__(
        self,
        backend_name: Union[str, Backend] = "numpy",
        h0_lr: float = 1.0,
        **h1_kwargs,
    ):
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.h0_lr = h0_lr
        self.state = None
        self._h1_estimator = ScaledGaussianNaturalGradientEstimator(
            backend_name=backend_name, **h1_kwargs
        )

    def initialize(self, X: Array) -> Array:
        """Initialize detector with first two time steps.

        Parameters
        ----------
        X : Array of shape (..., 2, n_samples, n_features)
            Data from two time steps

        Returns
        -------
        Array of shape (...)
            Initial DCG statistic (should be ~0 with only 2 dates)
        """
        X = get_data_on_device(X, self.backend_name)
        assert X.shape[-3] == 2, "Initialize expects exactly 2 time steps"

        n_samples = X.shape[-2]
        n_features = X.shape[-1]
        batch_shape = X.shape[:-3]

        # H0: initialize and update with both dates
        h0_estimator = OnlineScaledGaussianEstimator(
            n_features, n_samples, lr=self.h0_lr, backend_name=self.backend_name
        )
        X_0 = X[..., 0, :, :]  # (..., n, p)
        Sigma_h0, tau_h0 = h0_estimator.update(X_0)
        log_lik_h0 = self._compute_log_likelihood(X_0, Sigma_h0, tau_h0)

        X_1 = X[..., 1, :, :]  # (..., n, p)
        Sigma_h0, tau_h0 = h0_estimator.update(X_1)
        log_lik_h0 = log_lik_h0 + self._compute_log_likelihood(X_1, Sigma_h0, tau_h0)

        # H1: separate estimates for each date
        Sigma_h1_0, tau_h1_0 = self._h1_estimator.compute(X_0)
        log_lik_h1_0 = self._compute_log_likelihood(X_0, Sigma_h1_0, tau_h1_0)

        Sigma_h1_1, tau_h1_1 = self._h1_estimator.compute(X_1)
        log_lik_h1_1 = self._compute_log_likelihood(X_1, Sigma_h1_1, tau_h1_1)

        log_lik_h1_total = log_lik_h1_0 + log_lik_h1_1

        # Initialize state
        self.state = OnlineDCGDetectorState(
            h0_estimator=h0_estimator,
            log_likelihood_h0=log_lik_h0,
            log_likelihood_h1_total=log_lik_h1_total,
            n_times=2,
        )

        # Initial GLRT (should be small)
        glrt = 2 * (log_lik_h0 - log_lik_h1_total)
        return glrt

    def compute(
        self, past_value: Array, X: Array, state: Any, *args, **kwargs
    ) -> Tuple[Array, Any]:
        """Update detector with new time step.

        Parameters
        ----------
        past_value : Array of shape (...)
            Previous GLRT statistic
        X : Array of shape (..., n_samples, n_features)
            Data for current time step
        state : OnlineDCGDetectorState
            Current detector state

        Returns
        -------
        Tuple[Array, OnlineDCGDetectorState]
            (Updated GLRT statistic, updated state)
        """
        X = get_data_on_device(X, self.backend_name)

        # H0: update pooled estimator and compute likelihood on current batch
        Sigma_h0, tau_h0 = state.h0_estimator.update(X)
        log_lik_h0_batch = self._compute_log_likelihood(X, Sigma_h0, tau_h0)

        # H1: per-date estimate for this time step
        Sigma_h1, tau_h1 = self._h1_estimator.compute(X)
        log_lik_h1 = self._compute_log_likelihood(X, Sigma_h1, tau_h1)

        # Accumulate log-likelihoods
        new_log_lik_h0 = state.log_likelihood_h0 + log_lik_h0_batch
        new_log_lik_h1_total = state.log_likelihood_h1_total + log_lik_h1
        new_n_times = state.n_times + 1

        new_state = OnlineDCGDetectorState(
            h0_estimator=state.h0_estimator,
            log_likelihood_h0=new_log_lik_h0,
            log_likelihood_h1_total=new_log_lik_h1_total,
            n_times=new_n_times,
        )

        # GLRT statistic
        glrt = 2 * (new_log_lik_h0 - new_log_lik_h1_total)

        return glrt, new_state

    def _compute_log_likelihood(self, X: Array, Sigma: Array, tau: Array) -> Array:
        """Compute negative log-likelihood for scaled Gaussian model.

        Parameters
        ----------
        X : Array of shape (..., n_samples, n_features)
        Sigma : Array of shape (..., n_features, n_features)
        tau : Array of shape (..., n_samples, 1)

        Returns
        -------
        Array of shape (...)
            Negative log-likelihood, summed over samples, normalized by n*p
        """
        n_samples = X.shape[-2]
        n_features = X.shape[-1]

        # Mahalanobis distances
        i_Sigma = self.be.linalg.inv(Sigma)
        q = self.be.real(
            self.be.einsum("...ni,...ij,...nj->...n", X.conj(), i_Sigma, X)
        )  # (..., n)

        # Negative log-likelihood (det(Sigma)=1 already)
        tau_flat = tau[..., 0]  # (..., n)
        L = n_features * self.be.log(tau_flat) + q / tau_flat  # (..., n)
        # Sum over samples, return per-batch
        return self.be.sum(L, axis=-1) / (n_samples * n_features)
