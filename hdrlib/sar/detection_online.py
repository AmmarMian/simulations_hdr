# Online change detection for scaled Gaussian model
# Author: Ammar Mian

from dataclasses import dataclass
from typing import Union, Tuple, Any

from ..core.backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
)
from ..core.detection import OnlineDetector
from ..core.estimation import ScaledGaussianNaturalGradientEstimator
from .estimation_online import OnlineScaledGaussianEstimator, OnlineKroneckerEstimator
from .estimation_kronecker import kronecker_mm_h1, _kronecker_quadratic_forms
from .detectors import _tyler_matandtext_fixed_point


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
    scatter_h0 : Array of shape (..., n_samples, n_features, n_features)
        Sufficient statistic S_n = sum_t x_{t,n} x_{t,n}^H, accumulated one
        date at a time. Its size does not grow with T, so the detector stays
        streaming: no date is ever kept, yet L0 can be re-evaluated in full
        with the current estimate at every read. See _log_lik_h0_pooled.
    log_likelihood_h1_total : Array
        Sum of per-date H1 log-likelihoods. Each term uses its own date's MLE,
        so this one is correct to accumulate.
    n_times : int
        Number of time steps processed
    """

    h0_estimator: OnlineScaledGaussianEstimator
    scatter_h0: Array
    log_likelihood_h1_total: Array
    n_times: int


class OnlineDCGDetector(OnlineDetector):
    r"""Online scaled-Gaussian change point detector.

    Generalized likelihood ratio test between a single set of parameters
    shared by every date and one set per date:

    $$
    \Lambda = 2 \left[
      \log L\!\left(x \,\middle|\, \widehat{\Sigma}_0,
                      \widehat{\tau}_0 \right)
      - \sum_{t=1}^{T} \log L\!\left(x_t \,\middle|\,
                      \widehat{\Sigma}_t, \widehat{\tau}_t \right)
    \right],
    $$

    where under $H_0$ a single $(\Sigma, \tau)$ is pooled across all dates
    and under $H_1$ each date carries its own. What makes the detector
    *online* is that the $H_0$ estimate is refreshed by one Riemannian
    gradient step per new date rather than recomputed from the whole
    time-series.

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
        h0_step_rule: str = "armijo",
        **h1_kwargs,
    ):
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.h0_step_rule = h0_step_rule
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
            step_rule=self.h0_step_rule,
            armijo_max_backtracks=self.h0_max_backtracks,
            iter_max=self.iter_max,
            tol=self.tol,
            backend_name=self.backend_name,
        )
        X_0 = X[..., 0, :, :]  # (..., n, p)
        X_1 = X[..., 1, :, :]  # (..., n, p)

        # H0 warm start on BOTH dates at once, via the MatAndText MLE -- the
        # very estimator the offline DeterministicCompoundGaussianGLRT uses
        # under H0.
        #
        # This used to warm-start on date 0 alone and spend date 1 on a single
        # gradient step. That wastes half of what initialize() is handed, and
        # for the unstructured model it is fatal: a one-date warm start fits
        # p^2 = 144 parameters to n = 13 samples, and the recursion never
        # recovers. Measured on K, nu = 1: the online estimate sits at an
        # affine-invariant distance of 4.4 to 4.6 from the offline MatAndText
        # estimate at every T from 2 to 49 -- flat, no convergence, against a
        # sampling scatter of 0.67 between two independent offline estimates --
        # and travels only 1.4 from its warm start over 49 dates. Power at
        # Pfa = 1e-2 then plateaued at 0.82 where the offline reaches 1.000.
        # Pooling the two dates lifts it to 0.99 by T = 5.
        #
        # OnlineKroneckerDetector keeps the one-date warm start below and does
        # not suffer from it: its MM fit estimates a^2 + b^2 = 25 parameters
        # from the same 13 samples.
        Sigma_h0 = _tyler_matandtext_fixed_point(
            X[..., :2, :, :], tol=self.tol, iter_max=max(50, self.iter_max),
            backend_name=self.backend_name,
        )
        # The estimator's manifold is SHPD: rescale to det = 1 and let tau
        # carry the scale. The likelihood is invariant under the joint change.
        log_det = self.be.real(self.be.linalg.slogdet(Sigma_h0)[1])
        Sigma_h0 = Sigma_h0 * self.be.exp(-log_det / n_features)[..., None, None]
        i_Sigma = self.be.linalg.inv(Sigma_h0)
        q = self.be.real(self.be.einsum(
            "...tni,...ij,...tnj->...tn", X[..., :2, :, :].conj(), i_Sigma,
            X[..., :2, :, :]))
        tau_h0 = (self.be.sum(q, axis=-2) / (2 * n_features))[..., None]
        h0_estimator.set_state(Sigma_h0, tau_h0, 2)

        scatter_h0 = self._scatter(X_0) + self._scatter(X_1)

        # H1: separate estimates for each date
        Sigma_h1_0, tau_h1_0 = self._h1_estimator.compute(X_0)
        log_lik_h1_0 = self._compute_log_likelihood(X_0, Sigma_h1_0, tau_h1_0)

        Sigma_h1_1, tau_h1_1 = self._h1_estimator.compute(X_1)
        log_lik_h1_1 = self._compute_log_likelihood(X_1, Sigma_h1_1, tau_h1_1)

        log_lik_h1_total = log_lik_h1_0 + log_lik_h1_1

        # Initialize state
        self.state = OnlineDCGDetectorState(
            h0_estimator=h0_estimator,
            scatter_h0=scatter_h0,
            log_likelihood_h1_total=log_lik_h1_total,
            n_times=2,
        )

        # Initial GLRT (should be small)
        log_lik_h0 = self._log_lik_h0_pooled(scatter_h0, 2, Sigma_h0, tau_h0)
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

        # H0: one step of the online estimator, then fold the new date into
        # the sufficient statistic. L0 is NOT accumulated -- it is recomputed
        # below from S and the estimate as it stands now.
        Sigma_h0, tau_h0 = state.h0_estimator.update(X)
        new_scatter = state.scatter_h0 + self._scatter(X)
        new_n_times = state.n_times + 1

        # H1: per-date estimate for this time step
        Sigma_h1, tau_h1 = self._h1_estimator.compute(X)
        log_lik_h1 = self._compute_log_likelihood(X, Sigma_h1, tau_h1)
        new_log_lik_h1_total = state.log_likelihood_h1_total + log_lik_h1

        new_state = OnlineDCGDetectorState(
            h0_estimator=state.h0_estimator,
            scatter_h0=new_scatter,
            log_likelihood_h1_total=new_log_lik_h1_total,
            n_times=new_n_times,
        )

        new_log_lik_h0 = self._log_lik_h0_pooled(
            new_scatter, new_n_times, Sigma_h0, tau_h0)
        glrt = 2 * (new_log_lik_h0 - new_log_lik_h1_total)

        return glrt, new_state

    def _scatter(self, X: Array) -> Array:
        """Per-sample outer products ``S_n = x_n x_n^H`` of one date.

        Indexed so that ``tr(Sigma^-1 S_n)`` recovers ``x_n^H Sigma^-1 x_n``.
        """
        return self.be.einsum("...ni,...nj->...nij", X, X.conj())

    def _log_lik_h0_pooled(
        self, scatter: Array, n_times: int, Sigma: Array, tau: Array
    ) -> Array:
        """H0 log-likelihood over every date seen so far, at the current estimate.

        Same quantity as summing _compute_log_likelihood over the dates with a
        fixed (Sigma, tau), but reached from the sufficient statistic rather
        than from the dates themselves:

            sum_t sum_n q_{t,n} / tau_n = sum_n tr(Sigma^-1 S_n) / tau_n

        so only S and the date count are carried, never the data. This is what
        makes the H0 term match the offline statistic while the detector still
        sees one date at a time.
        """
        n_features = Sigma.shape[-1]
        n_samples = tau.shape[-2]
        i_Sigma = self.be.linalg.inv(Sigma)
        quad = self.be.real(
            self.be.einsum("...ij,...nji->...n", i_Sigma, scatter))  # (..., n)
        log_det = self.be.real(self.be.linalg.slogdet(Sigma)[1])  # (...)
        tau_flat = tau[..., 0]
        total = (n_times * n_samples * log_det
                 + n_times * n_features * self.be.sum(self.be.log(tau_flat), axis=-1)
                 + self.be.sum(quad / tau_flat, axis=-1))
        return 0.5 * total

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

        # Honest log-det, as OnlineKroneckerDetector._compute_log_likelihood
        # does with b·log|A| + a·log|B|.
        #
        # This used to substitute log|Σ_trace| = p·log(p/Tr Σ), on the grounds
        # that the online estimator enforces det(Σ)=1 (it does: log|Σ| is 4e-16
        # here) and that the offline DeterministicCompoundGaussianGLRT works on
        # trace-normalised Tyler estimates. But the scaled Gaussian likelihood
        # is invariant only under the JOINT rescaling (Σ, τ) → (cΣ, τ/c):
        # moving the log-det alone, while q and τ still come from the
        # det-normalised Σ, adds a spurious N·p·log(p/Tr Σ) per date — about
        # -264 per date at p=12, N=13, Tr Σ ≈ 65.
        log_det_sigma = self.be.real(self.be.linalg.slogdet(Sigma)[1])  # (...)
        tau_flat = tau[..., 0]  # (..., n)
        L = n_features * self.be.log(tau_flat) + log_det_sigma[..., None] + q / tau_flat
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
    scatter_h0 : Array of shape (..., n_samples, a, a, b, b)
        Sufficient statistic of the Kronecker quadratic form, accumulated one
        date at a time and of a size that does not grow with T. Written in the
        factor indices rather than as a p x p scatter so that the full
        Kronecker product is never materialised, as everywhere else in this
        module. See _log_lik_h0_pooled.
    log_likelihood_h1_total : Array
        Sum of per-date H1 log-likelihoods. Each term uses its own date's MM
        estimate, so this one is correct to accumulate.
    n_times : int
        Number of time steps processed
    """

    h0_estimator: OnlineKroneckerEstimator
    scatter_h0: Array
    log_likelihood_h1_total: Array
    n_times: int


class OnlineKroneckerDetector(OnlineDetector):
    r"""Online Kronecker-structured scaled Gaussian change point detector.

    The test of :class:`OnlineDCGDetector`, with the covariance constrained to
    $\Sigma = A \otimes B$:

    $$
    \Lambda = 2 \left[
      \log L\!\left(x \,\middle|\, \widehat{A}_0, \widehat{B}_0,
                      \widehat{\tau}_0 \right)
      - \sum_{t=1}^{T} \log L\!\left(x_t \,\middle|\,
                      \widehat{A}_t, \widehat{B}_t, \widehat{\tau}_t
                      \right)
    \right].
    $$

    Under $H_0$ a single $(A, B)$ is shared across all dates and updated by a
    stochastic natural gradient step on
    $\mathcal{SH}^{++}(a) \times \mathcal{SH}^{++}(b) \times
    (\mathbb{R}^{+})^{N}$ with the Fisher information metric; under $H_1$
    each date is estimated on its own by the Kronecker MM algorithm.

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
        h0_step_rule: str = "armijo",
        h0_init_mode: str = "mm",
    ):
        self.a = a
        self.b = b
        self.backend_name = backend_name
        self.be = get_backend_module(backend_name)
        self.h0_step_rule = h0_step_rule
        self.h0_init_mode = h0_init_mode
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
            step_rule=self.h0_step_rule,
            init_mode=self.h0_init_mode,
            armijo_max_backtracks=self.h0_max_backtracks,
            iter_max=self.iter_max,
            tol=self.tol,
            backend_name=self.backend_name,
        )

        X_0 = X[..., 0, :, :]  # (..., N, p)
        A_h0, B_h0, tau_h0 = h0_estimator.update(X_0)

        X_1 = X[..., 1, :, :]
        A_h0, B_h0, tau_h0 = h0_estimator.update(X_1)

        scatter_h0 = self._scatter(X_0) + self._scatter(X_1)

        # H1: per-date Kronecker MM (T=1 for each date)
        log_lik_h1_0 = self._compute_h1_log_likelihood(X_0)
        log_lik_h1_1 = self._compute_h1_log_likelihood(X_1)
        log_lik_h1_total = log_lik_h1_0 + log_lik_h1_1

        self.state = OnlineKroneckerDetectorState(
            h0_estimator=h0_estimator,
            scatter_h0=scatter_h0,
            log_likelihood_h1_total=log_lik_h1_total,
            n_times=2,
        )

        log_lik_h0 = self._log_lik_h0_pooled(scatter_h0, 2, A_h0, B_h0, tau_h0)
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

        # H0: one natural gradient step, then fold the new date into the
        # sufficient statistic. L0 is recomputed below, not accumulated.
        A_h0, B_h0, tau_h0 = state.h0_estimator.update(X)
        new_scatter = state.scatter_h0 + self._scatter(X)
        new_n_times = state.n_times + 1

        # H1: per-date MM for this time step
        log_lik_h1 = self._compute_h1_log_likelihood(X)
        new_log_lik_h1_total = state.log_likelihood_h1_total + log_lik_h1

        new_state = OnlineKroneckerDetectorState(
            h0_estimator=state.h0_estimator,
            scatter_h0=new_scatter,
            log_likelihood_h1_total=new_log_lik_h1_total,
            n_times=new_n_times,
        )

        new_log_lik_h0 = self._log_lik_h0_pooled(
            new_scatter, new_n_times, A_h0, B_h0, tau_h0)
        return 2 * (new_log_lik_h0 - new_log_lik_h1_total), new_state

    def _scatter(self, X: Array) -> Array:
        """Sufficient statistic of one date, in the Kronecker factor indices.

        With M_n the (b, a) reshape of x_n used throughout this module and
        q_n = tr(A^-1 M_n^H B^-1 M_n), expanding the traces gives

            q_n = sum_{i,i',j,j'} (A^-1)_{i i'} (B^-1)_{j j'}
                                  conj(M_{j i'}) M_{j' i}

        so C[n, i, i', j', j] = sum_t M_{t,n,j',i} conj(M_{t,n,j,i'}) carries
        everything the H0 term needs about the past, in a a*a*b*b = p^2 array
        per sample that never grows with T.
        """
        a, b = self.a, self.b
        M = X.reshape(*X.shape[:-1], a, b).swapaxes(-1, -2)  # (..., N, b, a)
        return self.be.einsum("...nbi,...ncj->...nijbc", M, M.conj())

    def _log_lik_h0_pooled(
        self, scatter: Array, n_times: int, A: Array, B: Array, tau: Array
    ) -> Array:
        """H0 log-likelihood over every date seen so far, at the current estimate."""
        a, b = self.a, self.b
        p = a * b
        n_samples = tau.shape[-2]
        iA = self.be.linalg.inv(A)
        iB = self.be.linalg.inv(B)
        quad = self.be.real(
            self.be.einsum("...ik,...jl,...niklj->...n", iA, iB, scatter))
        log_det_A = self.be.real(self.be.linalg.slogdet(A)[1])
        log_det_B = self.be.real(self.be.linalg.slogdet(B)[1])
        log_det_kron = b * log_det_A + a * log_det_B
        tau_flat = tau[..., 0]
        total = (n_times * n_samples * log_det_kron
                 + n_times * p * self.be.sum(self.be.log(tau_flat), axis=-1)
                 + self.be.sum(quad / tau_flat, axis=-1))
        return 0.5 * total

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
