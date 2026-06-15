# Online change detection for scaled Gaussian model
# Author: Ammar Mian

import math
from dataclasses import dataclass
from typing import Union, Tuple, Any

from ..core.backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
    batched_trace,
)
from ..core.detection import OnlineDetector
from ..core.estimation import ScaledGaussianNaturalGradientEstimator
from .estimation_online import OnlineScaledGaussianEstimator, OnlineKroneckerEstimator
from .estimation_kronecker import kronecker_mm_h1, _kronecker_quadratic_forms


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
    h0_alpha_0 : float
        Initial Armijo step size for the online H0 estimator. By default 0.1.
    h0_max_backtracks : int
        Maximum number of Armijo backtracking steps. By default 5.
    **h1_kwargs
        Additional keyword arguments for H1 estimator (e.g., iter_max, tol).
    """

    def __init__(
        self,
        backend_name: Union[str, Backend] = "numpy",
        h0_alpha_0: float = 0.1,
        h0_max_backtracks: int = 5,
        iter_max: int = 200,
        tol: float = 1e-8,
        **h1_kwargs,
    ):
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.h0_alpha_0 = h0_alpha_0
        self.h0_max_backtracks = h0_max_backtracks
        self.iter_max = iter_max
        self.tol = tol
        self.state = None
        self._h1_estimator = ScaledGaussianNaturalGradientEstimator(
            backend_name=backend_name, iter_max=iter_max, tol=tol, **h1_kwargs
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

        # H0: initialize and update with both dates
        h0_estimator = OnlineScaledGaussianEstimator(
            n_features, n_samples,
            alpha_0=self.h0_alpha_0,
            armijo_max_backtracks=self.h0_max_backtracks,
            iter_max=self.iter_max,
            tol=self.tol,
            backend_name=self.backend_name,
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

    def _compute_log_likelihood(self, X: Array, Sigma: Array, tau: Array) -> Array:  # noqa: E501
        """Compute negative log-likelihood for scaled Gaussian model.

        Parameters
        ----------
        X : Array of shape (..., n_samples, n_features)
        Sigma : Array of shape (..., n_features, n_features)
        tau : Array of shape (..., n_samples, 1)

        Returns
        -------
        Array of shape (...)
            Negative log-likelihood summed over samples (unnormalised).
            Factor 0.5 is applied so that 2*(L_H0 - L_H1) reproduces the
            offline DeterministicCompoundGaussianGLRT formula exactly when
            both estimators reach their respective MLEs (Mahalanobis q/tau→p
            cancels; log|Sigma| and p*log(tau) terms match the offline).
        """
        n_features = X.shape[-1]

        i_Sigma = self.be.linalg.inv(Sigma)
        q = self.be.real(
            self.be.einsum("...ni,...ij,...nj->...n", X.conj(), i_Sigma, X)
        )  # (..., n)

        # Trace-normalised log-det: log|Σ_trace| where Σ_trace = Σ·p/Tr(Σ).
        # The online estimator enforces det(Σ)≈1, so log|Σ_det|≈0; using
        # log|Σ_trace| = p·log(p/Tr(Σ)) instead makes the scale match the
        # offline DeterministicCompoundGaussianGLRT which uses Tyler (Tr=p).
        trace_sigma = self.be.real(batched_trace(self.backend_name, Sigma))  # (...)
        log_det_sigma_trace = n_features * math.log(n_features) - n_features * self.be.log(trace_sigma)
        tau_flat = tau[..., 0]  # (..., n)
        L = n_features * self.be.log(tau_flat) + log_det_sigma_trace[..., None] + q / tau_flat
        return 0.5 * self.be.sum(L, axis=-1)


# -----------------------------------------------------------------------
# Online Kronecker Detector
# -----------------------------------------------------------------------
@dataclass
class OnlineKroneckerDetectorState:
    """State for OnlineKroneckerDetector.

    Attributes
    ----------
    h0_estimator : OnlineKroneckerEstimator
        Online estimator for H0 (pooled across all dates)
    log_likelihood_h0 : Array
        Accumulated log-likelihood under H0
    log_likelihood_h1_total : Array
        Sum of per-date H1 log-likelihoods
    n_times : int
        Number of time steps processed
    """

    h0_estimator: OnlineKroneckerEstimator
    log_likelihood_h0: Array
    log_likelihood_h1_total: Array
    n_times: int


class OnlineKroneckerDetector(OnlineDetector):
    """Online Kronecker structured scaled Gaussian change point detector.

    Uses a GLRT to detect changes in the Kronecker covariance structure
    kron(A, B) and textures tau over time.

    H0: Single (A, B) shared across all dates, updated via stochastic natural
        gradient on SHPD(a) x SHPD(b) x SPV(N) with Fisher Information Metric.
    H1: Separate (A_t, B_t) per date estimated via the Kronecker MM algorithm.

    Test statistic: GLRT = 2 * [log L(data | H0_pooled) - sum_t log L(data_t | H1_t)]

    Parameters
    ----------
    a, b : int
        Sizes of the Kronecker factors (p = a*b).
    backend_name : str or Backend
        Backend specification. By default 'numpy'.
    h0_alpha_0 : float
        Initial step size for the H0 online estimator. By default 0.1.
    h0_max_backtracks : int
        Max Armijo backtracking steps for H0. By default 5.
    iter_max : int
        Max MM iterations for warm-start and H1 per-date estimates. By default 30.
    tol : float
        Convergence tolerance for MM algorithms. By default 1e-4.
    """

    def __init__(
        self,
        a: int,
        b: int,
        backend_name: Union[str, "Backend"] = "numpy",
        h0_alpha_0: float = 0.1,
        h0_max_backtracks: int = 5,
        iter_max: int = 30,
        tol: float = 1e-4,
    ):
        self.a = a
        self.b = b
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.h0_alpha_0 = h0_alpha_0
        self.h0_max_backtracks = h0_max_backtracks
        self.iter_max = iter_max
        self.tol = tol
        self.state = None

    def initialize(self, X: Array) -> Array:
        """Initialize detector with first two time steps.

        Parameters
        ----------
        X : Array of shape (..., 2, n_samples, p) where p = a*b

        Returns
        -------
        Array of shape (...)
            Initial Kronecker GLRT statistic.
        """
        X = get_data_on_device(X, self.backend_name)
        assert X.shape[-3] == 2, "Initialize expects exactly 2 time steps"

        n_samples = X.shape[-2]

        h0_estimator = OnlineKroneckerEstimator(
            self.a, self.b, n_samples,
            alpha_0=self.h0_alpha_0,
            armijo_max_backtracks=self.h0_max_backtracks,
            iter_max=self.iter_max,
            tol=self.tol,
            backend_name=self.backend_name,
        )

        X_0 = X[..., 0, :, :]  # (..., N, p)
        A_h0, B_h0, tau_h0 = h0_estimator.update(X_0)
        log_lik_h0 = self._compute_log_likelihood(X_0, A_h0, B_h0, tau_h0)

        X_1 = X[..., 1, :, :]
        A_h0, B_h0, tau_h0 = h0_estimator.update(X_1)
        log_lik_h0 = log_lik_h0 + self._compute_log_likelihood(X_1, A_h0, B_h0, tau_h0)

        # H1: per-date Kronecker MM (T=1 for each date)
        log_lik_h1_0 = self._compute_h1_log_likelihood(X_0)
        log_lik_h1_1 = self._compute_h1_log_likelihood(X_1)
        log_lik_h1_total = log_lik_h1_0 + log_lik_h1_1

        self.state = OnlineKroneckerDetectorState(
            h0_estimator=h0_estimator,
            log_likelihood_h0=log_lik_h0,
            log_likelihood_h1_total=log_lik_h1_total,
            n_times=2,
        )

        return 2 * (log_lik_h0 - log_lik_h1_total)

    def compute(
        self, past_value: Array, X: Array, state: Any, *args, **kwargs
    ) -> Tuple[Array, Any]:
        """Update detector with a new time step.

        Parameters
        ----------
        past_value : Array of shape (...)
            Previous GLRT statistic.
        X : Array of shape (..., n_samples, p)
            Data for the current time step.
        state : OnlineKroneckerDetectorState

        Returns
        -------
        Tuple[Array, OnlineKroneckerDetectorState]
            (Updated GLRT statistic, updated state)
        """
        X = get_data_on_device(X, self.backend_name)

        # H0: one natural gradient step on pooled estimate
        A_h0, B_h0, tau_h0 = state.h0_estimator.update(X)
        log_lik_h0_batch = self._compute_log_likelihood(X, A_h0, B_h0, tau_h0)

        # H1: per-date MM for this time step
        log_lik_h1 = self._compute_h1_log_likelihood(X)

        new_log_lik_h0 = state.log_likelihood_h0 + log_lik_h0_batch
        new_log_lik_h1_total = state.log_likelihood_h1_total + log_lik_h1

        new_state = OnlineKroneckerDetectorState(
            h0_estimator=state.h0_estimator,
            log_likelihood_h0=new_log_lik_h0,
            log_likelihood_h1_total=new_log_lik_h1_total,
            n_times=state.n_times + 1,
        )

        return 2 * (new_log_lik_h0 - new_log_lik_h1_total), new_state

    def _compute_h1_log_likelihood(self, X: Array) -> Array:
        """Run Kronecker MM on a single date and compute its log-likelihood.

        Parameters
        ----------
        X : Array of shape (..., N, p)

        Returns
        -------
        Array of shape (...)
        """
        X_t = X[..., None, :, :]  # (..., 1, N, p)
        A_t, B_t, tau_t = kronecker_mm_h1(
            X_t, self.a, self.b,
            tol=self.tol, iter_max=self.iter_max,
            backend_name=self.backend_name,
        )
        # Squeeze T=1 dimension: (..., 1, a, a) → (..., a, a)
        A_sq = A_t[..., 0, :, :]
        B_sq = B_t[..., 0, :, :]
        tau_sq = tau_t[..., 0, :, None]  # (..., N, 1)
        return self._compute_log_likelihood(X, A_sq, B_sq, tau_sq)

    def _compute_log_likelihood(
        self, X: Array, A: Array, B: Array, tau: Array
    ) -> Array:
        """Log-likelihood for the Kronecker scaled Gaussian model.

        Uses log|kron(A,B)| = b*log|A| + a*log|B| to avoid forming the full
        p×p matrix.

        Parameters
        ----------
        X : Array of shape (..., N, p)
        A : Array of shape (..., a, a)
        B : Array of shape (..., b, b)
        tau : Array of shape (..., N, 1)

        Returns
        -------
        Array of shape (...)
        """
        a, b = self.a, self.b
        p = a * b

        M_i = X.reshape(*X.shape[:-1], a, b).swapaxes(-1, -2)  # (..., N, b, a)
        iA = self.be.linalg.inv(A)
        iB = self.be.linalg.inv(B)
        Q = _kronecker_quadratic_forms(M_i, iA, iB, self.backend_name)  # (..., N)

        log_det_A = self.be.real(self.be.linalg.slogdet(A)[1])  # (...,)
        log_det_B = self.be.real(self.be.linalg.slogdet(B)[1])  # (...,)
        log_det_kron = b * log_det_A + a * log_det_B  # (...,)

        tau_flat = tau[..., 0]  # (..., N)
        L = p * self.be.log(tau_flat) + log_det_kron[..., None] + Q / tau_flat
        return 0.5 * self.be.sum(L, axis=-1)
