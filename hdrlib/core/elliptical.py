# Real-valued elliptical distributions: backend-agnostic sampling and geometry.
#
# Everything here follows the stochastic representation
#
#     x = mu + sqrt(R) * S^{1/2} u,     u ~ Uniform(S^{d-1}),
#
# where the modular variate R carries the whole dependence on the density
# generator g, with density f_R(r) ∝ r^{d/2-1} g(r).  A distribution
# therefore only has to say how to draw R; the sampler and the isodensity
# geometry are shared.
#
# Backend policy, matching the rest of hdrlib:
#   * everything that produces *arrays* (draws) goes through the primitives of
#     hdrlib.core.backend, so it runs on numpy / torch / cupy / jax and lands
#     on the requested device.  The only randomness primitives assumed are
#     standard normal and uniform draws, from which chi-squared and gamma
#     variates are built here.
#   * everything that produces *scalars* (quantiles, moments) is host-side and
#     uses scipy.  These are O(1) quantities used to place isodensity contours
#     and to normalise the scale, never in an inner loop, so keeping them on
#     the CPU costs nothing and avoids reimplementing special functions per
#     backend.
#
# Note that hdrlib.core.simulation provides the *complex* circular Gaussian and
# DCG generators used by the detection experiments.  This module is the
# real-valued counterpart, matching the elliptical model of the context chapter.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma as gamma_function
from scipy.stats import chi2, f as fisher_dist, gamma as gamma_dist, invgamma

from .backend import (
    Array,
    Backend,
    cast_like,
    get_backend_module,
    sample_standard_normal,
    sample_uniform,
)


# ---------------------------------------------------------------------------
# Random primitives built on top of the backend's normal/uniform draws
# ---------------------------------------------------------------------------

def _spawn_seeds(seed: Optional[int], n: int) -> list[Optional[int]]:
    """Derive ``n`` independent seeds from a single one.

    Returning ``None`` when no seed was given preserves the backend's own
    "unseeded" semantics.
    """
    if seed is None:
        return [None] * n
    sequence = np.random.SeedSequence(seed)
    return [int(s.generate_state(1)[0]) for s in sequence.spawn(n)]


def sample_gamma(
    shape_param: float,
    n_samples: int,
    backend: Union[str, "Backend"],
    seed: Optional[int] = None,
    scale: float = 1.0,
    max_rounds: int = 40,
) -> Array:
    """Draw gamma variates on any backend, via Marsaglia-Tsang.

    The algorithm needs only standard normal and uniform draws, which every
    backend provides, so no per-backend special-function support is required.
    Rejection is vectorised: a full batch is drawn per round and only the
    still-missing entries are filled in, which converges in a couple of rounds
    since the acceptance rate exceeds 95 percent.

    Parameters
    ----------
    shape_param : float
        Shape ``a`` of the gamma law. Values below 1 are handled by the
        standard boosting trick ``Gamma(a) = Gamma(a+1) * U^(1/a)``.
    n_samples : int
    backend : str or Backend
    seed : int, optional
    scale : float
        Scale ``theta``; the mean of the result is ``a * theta``.
    max_rounds : int
        Safety bound on the number of rejection rounds.

    Returns
    -------
    Array of shape (n_samples,) on the requested backend.
    """
    xp = get_backend_module(backend)

    if shape_param < 1.0:
        boost_seed, gamma_seed = _spawn_seeds(seed, 2)
        boosted = sample_gamma(
            shape_param + 1.0, n_samples, backend, seed=gamma_seed, scale=scale
        )
        uniforms = sample_uniform(n_samples, [], backend, seed=boost_seed)
        return boosted * uniforms ** (1.0 / shape_param)

    d = shape_param - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)

    result = None
    filled = None
    seeds = _spawn_seeds(seed, 2 * max_rounds)
    for round_index in range(max_rounds):
        normals = sample_standard_normal(
            n_samples, [], backend, seed=seeds[2 * round_index]
        )
        uniforms = sample_uniform(
            n_samples, [], backend, seed=seeds[2 * round_index + 1]
        )
        v = (1.0 + c * normals) ** 3
        positive = v > 0
        # Guard the logarithm on the rejected entries; they are masked out below.
        safe_v = xp.where(positive, v, xp.ones_like(v))
        accepted = positive & (
            xp.log(uniforms)
            < 0.5 * normals**2 + d - d * safe_v + d * xp.log(safe_v)
        )
        candidate = d * safe_v

        if result is None:
            result = xp.where(accepted, candidate, xp.zeros_like(candidate))
            filled = accepted
        else:
            take = accepted & (~filled)
            result = xp.where(take, candidate, result)
            filled = filled | accepted

        if bool(xp.all(filled)):
            break
    else:
        raise RuntimeError(
            f"Gamma sampling did not converge in {max_rounds} rounds "
            f"(shape={shape_param})."
        )

    return scale * result


def sample_chi2(
    df: float,
    n_samples: int,
    backend: Union[str, "Backend"],
    seed: Optional[int] = None,
) -> Array:
    """Draw chi-squared variates on any backend.

    Integer degrees of freedom use the exact sum-of-squared-normals definition,
    which avoids rejection entirely; non-integer ones fall back to the gamma
    sampler through ``chi2_k = 2 * Gamma(k/2)``.
    """
    xp = get_backend_module(backend)
    if float(df).is_integer():
        normals = sample_standard_normal(n_samples, [int(df)], backend, seed=seed)
        return xp.sum(normals**2, axis=-1)
    return sample_gamma(df / 2.0, n_samples, backend, seed=seed, scale=2.0)


def sample_uniform_sphere(
    n_samples: int,
    n_features: int,
    backend: Union[str, "Backend"],
    seed: Optional[int] = None,
) -> Array:
    """Draw uniformly on the unit sphere ``S^{d-1}``, shape (n_samples, d)."""
    xp = get_backend_module(backend)
    directions = sample_standard_normal(n_samples, [n_features], backend, seed=seed)
    norms = xp.sqrt(xp.sum(directions**2, axis=-1))
    return directions / norms[:, None]


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

class EllipticalDistribution(ABC):
    r"""Base class for a real elliptical distribution in dimension $d$.

    Every such distribution admits the stochastic representation

    $$
    \boldsymbol{x} = \boldsymbol{\mu}
      + \sqrt{\mathcal{R}}\, \boldsymbol{S}^{1/2} \boldsymbol{u},
    \qquad \boldsymbol{u} \sim
      \mathcal{U}\!\left(\mathbb{S}^{d-1}\right),
    $$

    with $\boldsymbol{S}$ the scatter matrix and
    $\mathcal{R} \perp \boldsymbol{u}$ the modular variate, whose density is
    $f_{\mathcal{R}}(r) \propto r^{d/2 - 1} g(r)$ for the density generator
    $g$. Subclasses therefore only describe the law of $\mathcal{R}$: it is
    the only thing distinguishing two elliptical distributions that share a
    scatter matrix.

    Parameters
    ----------
    n_features : int
        Dimension ``d``.
    normalize : bool
        When True, rescale so that ``E{Q} = d``, i.e. so that the covariance
        matrix equals the scatter matrix.  This is what makes distributions
        comparable at fixed scatter matrix; it needs a finite second-order
        moment.
    backend_name : str or Backend
        Backend on which draws are produced.
    """

    label: str = "elliptical"

    def __init__(
        self,
        n_features: int,
        normalize: bool = True,
        backend_name: Union[str, "Backend"] = "numpy",
    ) -> None:
        self.n_features = n_features
        self.normalize = normalize
        self.backend_name = backend_name
        self.backend_module = get_backend_module(backend_name)
        self._scale = self._normalisation_scale() if normalize else 1.0

    # -- to be provided by subclasses ---------------------------------------

    @abstractmethod
    def density_generator(self, t):
        """Density generator ``g(t)``, up to a multiplicative constant."""

    def weight_function(self, x, **kwargs):
        """Maximum-likelihood weight ``u(t) = -2 g'(t)/g(t)``, real convention.

        Signature matches the ``m_estimator_function`` hook of
        ``hdrlib.core.estimation.fixed_point_m_estimation_centered``, so the
        fixed-point engine can be reused directly::

            fixed_point_m_estimation_centered(
                X, m_estimator_function=distribution.weight_function,
            )
        """
        raise NotImplementedError(
            f"No closed-form weight function for {type(self).__name__}."
        )

    @abstractmethod
    def _sample_standard_modular(self, n_samples: int, seed: Optional[int]) -> Array:
        """Draw the modular variate before the ``E{Q} = d`` rescaling."""

    @abstractmethod
    def _standard_modular_quantile(self, probability: float) -> float:
        """Host-side quantile of the modular variate before rescaling."""

    @abstractmethod
    def _standard_modular_mean(self) -> float:
        """``E{Q}`` before rescaling; may be infinite."""

    # -- shared -------------------------------------------------------------

    def _normalisation_scale(self) -> float:
        mean = self._standard_modular_mean()
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError(
                f"{type(self).__name__} has no finite second-order moment: "
                "the covariance matrix is undefined, use normalize=False."
            )
        return self.n_features / mean

    def sample_modular_variate(self, n_samples: int, seed: Optional[int] = None) -> Array:
        """Draw ``n_samples`` realisations of ``Q`` on the configured backend."""
        return self._scale * self._sample_standard_modular(n_samples, seed)

    def modular_quantile(self, probability: float) -> float:
        """Host-side ``q`` such that ``P(Q <= q) = probability``."""
        return self._scale * self._standard_modular_quantile(probability)


class CompoundGaussian(EllipticalDistribution):
    r"""Elliptical distribution written as ``x = mu + sqrt(tau) z``.

    Subclasses give the texture ``tau`` twice: as a backend-side sampler for
    draws, and as a frozen ``scipy.stats`` law for the host-side quantiles.
    The modular variate factorises as $\mathcal{R} = \tau \, \chi^2_d$.
    """

    @abstractmethod
    def sample_texture(self, n_samples: int, seed: Optional[int]) -> Array:
        """Backend-side draw of the texture."""

    @abstractmethod
    def texture_law(self):
        """Frozen ``scipy.stats`` law of the texture, for host-side scalars."""

    def _sample_standard_modular(self, n_samples, seed):
        texture_seed, chi2_seed = _spawn_seeds(seed, 2)
        texture = self.sample_texture(n_samples, texture_seed)
        return texture * sample_chi2(
            self.n_features, n_samples, self.backend_name, seed=chi2_seed
        )

    def _standard_modular_mean(self):
        return self.n_features * float(self.texture_law().mean())

    def _standard_modular_cdf(self, q: float) -> float:
        """``P(Q <= q) = E_tau{ F_{chi2_d}(q / tau) }``, by quadrature."""
        law = self.texture_law()
        lower, upper = law.ppf(1e-10), law.ppf(1 - 1e-10)
        value, _ = quad(
            lambda tau: chi2.cdf(q / tau, df=self.n_features) * law.pdf(tau),
            lower, upper, limit=200,
        )
        return value

    def _standard_modular_quantile(self, probability):
        low, high = 1e-8, float(chi2.ppf(probability, df=self.n_features))
        while self._standard_modular_cdf(high) < probability:
            high *= 4.0
        return brentq(
            lambda q: self._standard_modular_cdf(q) - probability, low, high
        )


class GaussianDistribution(EllipticalDistribution):
    r"""Gaussian: $g(t) = e^{-t/2}$, modular variate
    $\mathcal{R} \sim \chi^2_d$."""

    label = "gaussienne"

    def density_generator(self, t):
        return np.exp(-np.asarray(t) / 2)

    def weight_function(self, x, **kwargs):
        return gaussian_weight(x)

    def _sample_standard_modular(self, n_samples, seed):
        return sample_chi2(self.n_features, n_samples, self.backend_name, seed=seed)

    def _standard_modular_quantile(self, probability):
        return float(chi2.ppf(probability, df=self.n_features))

    def _standard_modular_mean(self):
        return float(self.n_features)


class StudentTDistribution(CompoundGaussian):
    r"""Student t with $\nu$ = ``dof`` degrees of freedom.

    $$
    g(t) = \left(1 + \frac{t}{\nu}\right)^{-(d + \nu)/2},
    $$

    obtained for the inverse-gamma texture $\tau = \nu / w$ with
    $w \sim \chi^2_{\nu}$. The second-order moment exists only for
    $\nu > 2$.
    """

    label = "t de Student"

    def __init__(self, n_features, dof=3.0, normalize=True, backend_name="numpy"):
        self.dof = float(dof)
        super().__init__(n_features, normalize, backend_name)

    def density_generator(self, t):
        return (1 + np.asarray(t) / self.dof) ** (-(self.n_features + self.dof) / 2)

    def weight_function(self, x, **kwargs):
        return student_t_weight_real(x, df=self.dof, n_features=self.n_features)

    def sample_texture(self, n_samples, seed):
        # tau = nu / w with w ~ chi2_nu, built from the backend chi-squared.
        return self.dof / sample_chi2(
            self.dof, n_samples, self.backend_name, seed=seed
        )

    def texture_law(self):
        return invgamma(a=self.dof / 2, scale=self.dof / 2)

    def _standard_modular_quantile(self, probability):
        # Closed form: Q = d * F(d, nu), no quadrature needed.
        return float(
            self.n_features * fisher_dist.ppf(probability, self.n_features, self.dof)
        )

    def _standard_modular_mean(self):
        if self.dof <= 2:
            return np.inf
        return self.n_features * self.dof / (self.dof - 2)


class KDistribution(CompoundGaussian):
    r"""K-distribution with texture shape $\nu$ = ``dof``.

    Gamma texture of unit mean, $\tau \sim \mathcal{G}(\nu, 1/\nu)$, for
    which the density generator involves a modified Bessel function of the
    second kind:

    $$
    g(t) = t^{a/2} K_a\!\left(\sqrt{2 \nu t}\right),
    \qquad a = \nu - \frac{d}{2} .
    $$
    """

    label = "K"

    def __init__(self, n_features, dof=2.0, normalize=True, backend_name="numpy"):
        self.dof = float(dof)
        super().__init__(n_features, normalize, backend_name)

    def density_generator(self, t):
        from scipy.special import kv

        order = self.dof - self.n_features / 2
        t = np.asarray(t, dtype=float)
        return t ** (order / 2) * kv(order, np.sqrt(2 * self.dof * t))

    def weight_function(self, x, **kwargs):
        return k_distribution_weight_real(
            x, df=self.dof, n_features=self.n_features
        )

    def sample_texture(self, n_samples, seed):
        return sample_gamma(
            self.dof, n_samples, self.backend_name, seed=seed, scale=1.0 / self.dof
        )

    def texture_law(self):
        return gamma_dist(a=self.dof, scale=1.0 / self.dof)


class GeneralizedGaussianDistribution(EllipticalDistribution):
    r"""Generalized Gaussian: $g(t) = \exp\!\left(-t^s / (2b)\right)$.

    Not written as a compound-Gaussian here: the modular variate is available
    in closed form, since $\mathcal{R} = w^{1/s}$ with
    $w \sim \mathcal{G}\!\left(d / (2s),\, 2b\right)$. Taking $s = 1$,
    $b = 1$ recovers the Gaussian; $s < 1$ gives heavier tails.
    """

    label = "gaussienne généralisée"

    def __init__(
        self, n_features, shape=0.5, scale=None, normalize=True, backend_name="numpy"
    ):
        self.shape = float(shape)
        # Default scale already gives E{Q} = d, so normalize=False stays
        # comparable to the Gaussian.
        self.scale = float(scale) if scale is not None else self._unit_scale(n_features)
        super().__init__(n_features, normalize, backend_name)

    def _unit_scale(self, n_features) -> float:
        s = self.shape
        ratio = gamma_function(n_features / (2 * s)) / gamma_function(
            (n_features + 2) / (2 * s)
        )
        return 0.5 * (n_features * ratio) ** s

    def density_generator(self, t):
        return np.exp(-np.asarray(t) ** self.shape / (2 * self.scale))

    @property
    def _gamma_shape(self) -> float:
        return self.n_features / (2 * self.shape)

    def weight_function(self, x, **kwargs):
        return generalized_gaussian_weight_real(
            x, shape=self.shape, scale=self.scale
        )

    def _gamma_law(self):
        return gamma_dist(a=self._gamma_shape, scale=2 * self.scale)

    def _sample_standard_modular(self, n_samples, seed):
        gammas = sample_gamma(
            self._gamma_shape,
            n_samples,
            self.backend_name,
            seed=seed,
            scale=2 * self.scale,
        )
        return gammas ** (1.0 / self.shape)

    def _standard_modular_quantile(self, probability):
        return float(self._gamma_law().ppf(probability) ** (1 / self.shape))

    def _standard_modular_mean(self):
        return float(
            (2 * self.scale) ** (1 / self.shape)
            * gamma_function(self._gamma_shape + 1 / self.shape)
            / gamma_function(self._gamma_shape)
        )


# ---------------------------------------------------------------------------
# Sampling and geometry
# ---------------------------------------------------------------------------

def sample_elliptical(
    n_samples: int,
    mean: Array,
    scatter: Array,
    distribution: EllipticalDistribution,
    seed: Optional[int] = None,
) -> Array:
    """Draw from an elliptical distribution via its stochastic representation.

    ``mean`` and ``scatter`` must already live on the distribution's backend;
    use ``hdrlib.core.backend.get_data_on_device`` if needed.

    Returns
    -------
    Array of shape (n_samples, d) on the distribution's backend.
    """
    xp = distribution.backend_module
    backend = distribution.backend_name
    direction_seed, modular_seed = _spawn_seeds(seed, 2)

    cholesky = xp.linalg.cholesky(scatter)
    directions = sample_uniform_sphere(
        n_samples, distribution.n_features, backend, seed=direction_seed,
    )
    modular = distribution.sample_modular_variate(n_samples, seed=modular_seed)
    # Backends do not agree on a default float width — torch draws float32
    # while a scatter matrix coming from numpy is float64 — so align the draws
    # on the scatter matrix rather than assuming either.
    directions = cast_like(directions, scatter, backend)
    modular = cast_like(modular, scatter, backend)
    return mean + xp.sqrt(modular)[:, None] * (
        directions @ xp.swapaxes(cholesky, -1, -2)
    )


def isodensity_ellipse(
    scatter: np.ndarray,
    distribution: EllipticalDistribution,
    probability: float,
    n_points: int = 300,
) -> np.ndarray:
    r"""Centered isodensity curve enclosing a given probability mass.

    Host-side plotting helper: the curve is a small numpy array regardless of
    the distribution's backend.

    Because the density depends on the data only through the quadratic form,
    the curve is the ellipse

    $$
    \left\{ \boldsymbol{x} \;:\; \boldsymbol{x}^{\mathrm{T}}
    \boldsymbol{S}^{-1} \boldsymbol{x} = r \right\},
    $$

    $r$ being the corresponding quantile of the modular variate
    $\mathcal{R}$.

    Returns
    -------
    np.ndarray of shape (2, n_points), for a 2-dimensional scatter matrix.
    """
    radius = np.sqrt(distribution.modular_quantile(probability))
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    return np.linalg.cholesky(np.asarray(scatter)) @ circle


# ---------------------------------------------------------------------------
# M-estimation weight functions, real convention
# ---------------------------------------------------------------------------
#
# hdrlib.core.estimation holds the complex-circular weight functions used by
# the detection experiments, and its fixed-point engine takes the weight as a
# parameter, so the real counterparts live here rather than duplicating or
# modifying anything there:
#
#     from hdrlib.core.estimation import fixed_point_m_estimation_centered
#     Sigma = fixed_point_m_estimation_centered(
#         X, m_estimator_function=student_t_weight_real, df=3, n_features=d,
#     )
#
# Tyler's weight u(t) = d/t is deliberately absent: it is identical in the real
# and complex conventions, so estimation.TylerEstimator applies unchanged.

def student_t_weight_real(x, df: float = 3, n_features: int = 1, **kwargs):
    r"""Real Student-t maximum-likelihood weight,
    $u(t) = (d + \nu) / (t + \nu)$.

    Derived from $u(t) = -2 g'(t) / g(t)$ with the real generator
    $g(t) = (1 + t/\nu)^{-(d+\nu)/2}$. The complex counterpart in
    ``hdrlib.core.estimation`` reads $(d + \nu/2)/(t + \nu/2)$ instead,
    because
    both the generator and the factor relating ``u`` to ``g'/g`` differ; the
    two are genuinely distinct functions, not a reparametrisation.

    Signature matches what ``fixed_point_m_estimation_centered`` passes to its
    ``m_estimator_function``.

    Parameters
    ----------
    x : Array
        Quadratic forms, shape (..., n_samples).
    df : float
        Degrees of freedom ``nu``.
    n_features : int
        Dimension ``d``.

    Returns
    -------
    Array
        Weights, same shape as ``x``.
    """
    return (n_features + df) / (x + df)


def gaussian_weight(x, **kwargs):
    r"""Gaussian weight, $u(t) = 1$ — the fixed point is then the SCM."""
    return np.ones_like(np.asarray(x))


def generalized_gaussian_weight_real(x, shape: float = 0.5, scale: float = 1.0, **kwargs):
    r"""Generalized Gaussian weight, $u(t) = s\, t^{s-1} / b$.

    From $u(t) = -2 g'(t) / g(t)$ with
    $g(t) = \exp\!\left(-t^s / (2b)\right)$.
    """
    return shape * np.asarray(x) ** (shape - 1) / scale


def k_distribution_weight_real(x, df: float = 2, n_features: int = 1, **kwargs):
    r"""K-distribution weight,

    $$
    u(t) = \frac{c\, K_{a-1}\!\left(c \sqrt{t}\right)}
                 {\sqrt{t}\, K_a\!\left(c \sqrt{t}\right)},
    \qquad a = \nu - \frac{d}{2},
    \qquad c = \sqrt{2 \nu} .
    $$

    Obtained from $u(t) = -2 g'(t) / g(t)$ using
    $K_a'(z) = -K_{a-1}(z) - (a/z) K_a(z)$.

    Unlike the other weights this one needs scipy's Bessel function, so it is
    numpy-only; the fixed-point engine still runs on any backend when given one
    of the closed-form weights above.
    """
    from scipy.special import kv

    order = df - n_features / 2
    c = np.sqrt(2 * df)
    root = np.sqrt(np.asarray(x))
    return c * kv(order - 1, c * root) / (root * kv(order, c * root))
