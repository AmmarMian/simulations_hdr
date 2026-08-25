# Online scaled Gaussian estimation
# Author: Ammar Mian

import logging
from typing import List, Tuple, Union

from ..core.backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
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
def _cap_tau_step(u_tau, tau, max_exp, be):
    """Scale a texture tangent vector so the retraction exponent stays bounded.

    The retraction on R++^n is tau * exp(u / tau); an unbounded exponent
    overflows in the first iterations, where the step is largest and tau can
    still be far from the mode. Scaling the whole vector keeps the direction.
    """
    tau_safe = be.where(tau > 1e-12, tau, be.ones_like(tau) * 1e-12)
    max_ratio = be.max(be.abs(u_tau / tau_safe))
    scale = be.where(max_ratio > max_exp, max_exp / max_ratio, be.ones_like(max_ratio))
    return u_tau * scale



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
        step_rule: str = "armijo",
    ):
        if step_rule not in ("armijo", "fixed"):
            raise ValueError(f"step_rule must be 'armijo' or 'fixed', got {step_rule!r}")
        self.step_rule = step_rule
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
        if self.step_rule == "fixed":
            # Step alpha_0 / t, the stochastic natural gradient schedule, with
            # no line search -- the counterpart of OnlineKroneckerEstimator's
            # "fixed" rule for the unstructured model, and the step of the
            # released code: it calls rgrad_scaledgaussian, whose gradient is
            # p * n_samples times the one returned here, with lr = 1/(p*n), so
            # its effective step is exactly alpha_0 / t in this convention.
            #
            # DO NOT slow this step down on its own. It was once divided by
            # p * n_samples because SG-O had no power with alpha_0 = 1.0 (AUC
            # 0.49, an H0 right tail of q99/median = 4.3). The tail was not
            # caused by the step but by OnlineDCGDetector accumulating frozen
            # log-likelihood values: a fast-moving estimator makes the terms of
            # old dates genuinely stale. Slowing the step hid the tail by
            # freezing the estimator instead -- it then stayed at an
            # affine-invariant distance of 4.6 from its target for every T, and
            # power plateaued at 0.82 where the offline reaches 1.000.
            #
            # The detector now re-evaluates L0 from accumulated sufficient
            # statistics, so staleness is gone and this step is the right one.
            # The two changes only work together: reverting either one alone
            # brings back a blind detector. Measured on the Gaussian regime,
            # AUC at T = 7 / 19 / 49, offline reference 0.666 / 0.763 / 0.865:
            #   accumulated + slowed step   0.733 / 0.645 / 0.658  (flat)
            #   sufficient stats + slowed   0.733 / 0.649 / 0.663  (flat)
            #   sufficient stats + this     0.697 / 0.744 / 0.872  (tracks)
            alpha_t = self.alpha_0 / self._t
            tau_v = self.tau[..., 0]
            u_tau = _cap_tau_step(-alpha_t * r_tau[..., 0], tau_v, 3.0, self.be)
            Sigma_new, tau_new = self._manifold.retr(
                [self.Sigma, tau_v], [-alpha_t * r_Sigma, u_tau],
            )
            self.Sigma, self.tau = Sigma_new, tau_new[..., None]
            self._t += 1
            return self.Sigma, self.tau

        _, self.Sigma, self.tau = _armijo_backtracking_scaled_gaussian(
            X, self.Sigma, self.tau, r_Sigma, r_tau, self._manifold, self.be,
            alpha_0=self.alpha_0, c=self.armijo_c, rho=self.armijo_rho,
            max_backtracks=self.armijo_max_backtracks,
            backend_name=self.backend_name,
        )
        self._t += 1
        return self.Sigma, self.tau

    def set_state(self, Sigma: Array, tau: Array, n_updates: int):
        """Seed the running estimate, bypassing the warm-start on the next call.

        Lets a caller that holds a better initial estimate than a single batch
        can give -- OnlineDCGDetector pools the two dates it is handed at
        initialisation -- install it and have update() carry on with gradient
        steps from there. ``n_updates`` is the number of batches the seed
        already accounts for; it sets the 1/t schedule.

        Parameters
        ----------
        Sigma : Array of shape (..., n_features, n_features), SHPD (det = 1)
        tau : Array of shape (..., n_samples, 1), strictly positive
        n_updates : int
            Number of batches already folded into the seed, at least 1.
        """
        if n_updates < 1:
            raise ValueError(f"n_updates must be >= 1, got {n_updates}")
        self.Sigma, self.tau = Sigma, tau
        self._t = n_updates

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
    step_rule : {"armijo", "fixed"}
        "armijo" — backtracking line search at each update (default, and what
        the online detectors have always used).
        "fixed"  — no line search: step alpha_0 / t, the schedule of equation
        (19) of Mian et al. (2024), for which statistical efficiency is proved.
        Beware the normalisation: _rgrad_kronecker_scaled_gaussian returns the
        gradient of the paper divided by p * n_samples, so the paper's optimal
        alpha_0 = 1 / (p * n) corresponds to alpha_0 = 1.0 here. That is the
        default when step_rule is "fixed" and alpha_0 is left to None.
    init_mode : {"mm", "identity"}
        "mm"       — warm-start from the Kronecker MM estimate on the first
                     batch (default).
        "identity" — start from (I_a, I_b, 1_n), as the released code of the
                     paper does. Use it when comparing against that reference.
    alpha_0 : float, optional
        Initial step size. Default 0.1 for "armijo", 1.0 for "fixed".
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
        alpha_0: "float | None" = None,
        armijo_c: float = 1e-4,
        armijo_rho: float = 0.5,
        armijo_max_backtracks: int = 5,
        iter_max: int = 30,
        tol: float = 1e-4,
        backend_name: Union[str, Backend] = "numpy",
        max_exp_tau: float = 3.0,
        step_rule: str = "armijo",
        init_mode: str = "mm",
    ):
        if step_rule not in ("armijo", "fixed"):
            raise ValueError(f"step_rule must be 'armijo' or 'fixed', got {step_rule!r}")
        if init_mode not in ("mm", "identity"):
            raise ValueError(f"init_mode must be 'mm' or 'identity', got {init_mode!r}")
        self.a = a
        self.b = b
        self.n_samples = n_samples
        self.step_rule = step_rule
        self.init_mode = init_mode
        if alpha_0 is None:
            alpha_0 = 1.0 if step_rule == "fixed" else 0.1
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
            if self.init_mode == "identity":
                batch_shape = X.shape[:-2]
                eye_a = self.be.eye(self.a, dtype=X.dtype)
                eye_b = self.be.eye(self.b, dtype=X.dtype)
                self.A = get_data_on_device(
                    self.be.broadcast_to(eye_a, batch_shape + (self.a, self.a)) * 1,
                    self.backend_name)
                self.B = get_data_on_device(
                    self.be.broadcast_to(eye_b, batch_shape + (self.b, self.b)) * 1,
                    self.backend_name)
                self.tau = get_data_on_device(
                    self.be.ones(batch_shape + (self.n_samples, 1), dtype=self.be.real(X[..., :1, :1]).dtype),
                    self.backend_name)
                self._t = 1
                # The identity start consumes no data, so this first batch is
                # still used for a gradient step below.
            else:
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
        if self.step_rule == "fixed":
            tau_v = self.tau[..., 0]
            u_tau = _cap_tau_step(-alpha_t * r_tau, tau_v, self.max_exp_tau, self.be)
            A_new, B_new, tau_v_new = self._manifold.retr(
                [self.A, self.B, tau_v],
                [-alpha_t * r_A, -alpha_t * r_B, u_tau],
            )
            self.A, self.B, self.tau = A_new, B_new, tau_v_new[..., None]
            self._t += 1
            return self.A, self.B, self.tau

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
