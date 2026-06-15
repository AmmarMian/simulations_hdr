"""Sonar-specific MC helpers: CLI args, aggregation, and plot templates."""

from __future__ import annotations

import argparse
import logging
from string import Template
from typing import Sequence, Union

import numpy as np

from ..core.mc import add_mc_base_args  # re-export for convenience
from ..core.backend import (
    Backend,
    Array,
    get_backend_module,
    get_data_on_device,
    concatenate,
    batched_trace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sonar-specific CLI arguments
# ---------------------------------------------------------------------------

def add_mc_args(parser: argparse.ArgumentParser) -> None:
    """Add all sonar MC CLI arguments (base + sonar-specific) to *parser*."""
    add_mc_base_args(parser)

    g = parser.add_argument_group("sonar model")
    g.add_argument("--m", type=int, default=64,
        help="Per-array dimension (total = 2m, default 64).")
    g.add_argument("--beta", type=float, default=3e-4,
        help="Covariance scale factor beta (default 3e-4).")
    g.add_argument("--rho1", type=float, default=0.4,
        help="Array-1 correlation coefficient rho1 (default 0.4).")
    g.add_argument("--rho2", type=float, default=0.9,
        help="Array-2 correlation coefficient rho2 (default 0.9).")
    g.add_argument("--K", dest="K_secondary", type=int, default=None,
        help="Secondary data size K (default = 2*2m = 4m).")
    g.add_argument("--theta1", type=float, default=45.0,
        help="Array-1 steering angle in degrees (default 45).")
    g.add_argument("--theta2", type=float, default=45.0,
        help="Array-2 steering angle in degrees (default 45).")

    g2 = parser.add_argument_group("clutter model")
    g2.add_argument("--gaussian", dest="clutter", action="store_const", const="gaussian",
        default="gaussian", help="Gaussian clutter (default).")
    g2.add_argument("--k-dist", dest="clutter", action="store_const", const="k",
        help="K-distributed clutter.")
    g2.add_argument("--nu", type=float, default=0.5,
        help="K-distribution shape parameter nu (default 0.5).")

    g3 = parser.add_argument_group("SNR sweep")
    g3.add_argument("--snr-min", type=float, default=-25.0,
        help="Minimum SNR in dB (default -25).")
    g3.add_argument("--snr-max", type=float, default=5.0,
        help="Maximum SNR in dB (default 5).")
    g3.add_argument("--n-snr", type=int, default=150,
        help="Number of SNR values (default 150).")
    g3.add_argument("--pfa", type=float, default=1e-2,
        help="Nominal PFA for PD curves (default 1e-2).")


def add_angle_args(parser: argparse.ArgumentParser) -> None:
    """Add angular-grid arguments for PD-vs-angle experiments."""
    g = parser.add_argument_group("angle grid")
    g.add_argument("--theta-min", type=float, default=-75.0,
        help="Minimum steering angle for grid (default -75).")
    g.add_argument("--theta-max", type=float, default=75.0,
        help="Maximum steering angle for grid (default 75).")
    g.add_argument("--n-theta", type=int, default=51,
        help="Number of angles per dimension (default 51).")
    g.add_argument("--snr-db", type=float, default=-12.0,
        help="Fixed SNR in dB for PD-vs-angle map (default -12).")


def add_tyler_conv_args(parser: argparse.ArgumentParser) -> None:
    """Add 2TYL convergence tracking arguments."""
    g = parser.add_argument_group("Tyler convergence")
    g.add_argument("--iter-max", type=int, default=500,
        help="Number of fixed-point iterations to track (default 500).")
    g.add_argument("--K-conv", dest="K_conv", type=int, default=None,
        help="Secondary samples for convergence experiment (default = K_secondary).")


def resolve_K(args) -> int:
    """Return K_secondary, defaulting to 4*m (= 2*2m) if not set."""
    if args.K_secondary is not None:
        return args.K_secondary
    return 4 * args.m


def clutter_params(args) -> tuple[float, float]:
    """Return (tau_shape, tau_scale) for the chosen clutter model."""
    if args.clutter == "gaussian":
        return 1.0, 1.0
    # K-distributed: tau ~ Gamma(nu, 1/nu)
    return args.nu, 1.0 / args.nu


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def empirical_pfa(stats_h0: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Empirical PFA curve P(stat > threshold) for each threshold value.

    Parameters
    ----------
    stats_h0 : (n_trials,) H0 test statistics.
    thresholds : (n_thresh,) threshold values.

    Returns
    -------
    pfa : (n_thresh,) empirical PFA.
    """
    return np.mean(stats_h0[:, None] > thresholds[None, :], axis=0)


def threshold_at_pfa(stats_h0: np.ndarray, pfa: float) -> float:
    """Return the (1-pfa) empirical quantile of H0 statistics."""
    return float(np.quantile(stats_h0, 1.0 - pfa))


def empirical_pd(stats_h1: np.ndarray, threshold: float) -> float:
    """Empirical PD P(stat > threshold) for H1 statistics."""
    return float(np.mean(stats_h1 > threshold))


def aggregate_pd_snr(
    stats_h0: np.ndarray,
    stats_h1_per_snr: Sequence[np.ndarray],
    pfa: float,
) -> np.ndarray:
    """Compute PD vs SNR at a fixed empirical PFA.

    Parameters
    ----------
    stats_h0 : (n_trials_h0,) H0 test statistics.
    stats_h1_per_snr : sequence of (n_trials_h1,) arrays, one per SNR point.
    pfa : nominal false alarm rate.

    Returns
    -------
    pd : (n_snr,) detection probabilities.
    """
    eta = threshold_at_pfa(stats_h0, pfa)
    return np.array([empirical_pd(s, eta) for s in stats_h1_per_snr])


def aggregate_pfa_threshold(
    stats_h0: np.ndarray,
    n_thresh: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PFA-threshold curve from H0 statistics.

    Returns (thresholds, pfa) sorted ascending in threshold.
    """
    lo, hi = float(np.min(stats_h0)), float(np.max(stats_h0))
    thresholds = np.linspace(lo, hi, n_thresh)
    pfa = empirical_pfa(stats_h0, thresholds)
    return thresholds, pfa


def tyler_relative_deviations(
    X_secondary: Array,
    m: int,
    iter_max: int = 500,
    tol: float = 0.0,  # 0 → run all iter_max steps
    backend_name: Union[str, Backend] = "numpy",
) -> np.ndarray:
    """Track 2TYL relative Frobenius deviation ||M^(k) - M^(k-1)||_F / ||M^(k-1)||_F.

    Records the per-iteration convergence metric averaged across trials.

    Parameters
    ----------
    X_secondary : (K, 2m) or (n_trials, K, 2m)
        When batched, the **mean** relative deviation across trials is returned.
    m : int
    iter_max : int
    tol : float
        Set to 0 to always run all iter_max steps.
    backend_name : str or Backend

    Returns
    -------
    deviations : (iter_max,) numpy array
    """
    be = get_backend_module(backend_name)
    X = get_data_on_device(X_secondary, backend_name)

    p = 2 * m
    K = X.shape[-2]

    if X.ndim == 2:
        X = X[None, :, :]   # (1, K, 2m) — add dummy batch

    n = X.shape[0]
    x1 = X[:, :, :m]
    x2 = X[:, :, m:]

    M_eye = np.broadcast_to(np.eye(p, dtype=np.complex128), (n, p, p)).copy()
    M_hat = get_data_on_device(M_eye, backend_name)
    deviations = np.full(iter_max, np.nan)

    eps = 1e-30
    for it in range(iter_max):
        M_inv = be.linalg.inv(M_hat)
        iM11 = M_inv[:, :m, :m]
        iM12 = M_inv[:, :m, m:]
        iM22 = M_inv[:, m:, m:]

        vx1  = be.swapaxes(iM11 @ be.swapaxes(x1, -1, -2), -1, -2)
        vx2  = be.swapaxes(iM22 @ be.swapaxes(x2, -1, -2), -1, -2)
        vx12 = be.swapaxes(iM12 @ be.swapaxes(x2, -1, -2), -1, -2)

        t1  = be.real((x1.conj() * vx1).sum(-1)) / m    # (n, K)
        t2  = be.real((x2.conj() * vx2).sum(-1)) / m
        t12 = be.real((x1.conj() * vx12).sum(-1)) / m

        tau1 = be.abs(t1 + be.sqrt(be.abs(t1) / (be.abs(t2) + eps)) * t12) + eps
        tau2 = be.abs(t2 + be.sqrt(be.abs(t2) / (be.abs(t1) + eps)) * t12) + eps

        x1s = x1 / be.sqrt(tau1[:, :, None])
        x2s = x2 / be.sqrt(tau2[:, :, None])
        xs  = concatenate(backend_name, [x1s, x2s], axis=-1)

        M_new = be.swapaxes(xs, -1, -2).conj() @ xs / K
        tr = be.real(batched_trace(backend_name, M_new))     # (n,)
        M_new = M_new * (p / tr[:, None, None])

        diff = M_new - M_hat
        fd = be.sqrt(be.sum(be.abs(diff.reshape(n, -1)) ** 2, axis=-1))
        fm = be.sqrt(be.sum(be.abs(M_hat.reshape(n, -1)) ** 2, axis=-1))
        rel = fd / (fm + eps)
        deviations[it] = float(be.mean(rel))

        M_hat = M_new
        if tol > 0 and float(be.max(rel)) < tol:
            deviations[it + 1:] = 0.0
            break

    return deviations


# ---------------------------------------------------------------------------
# Detector batch runner helper
# ---------------------------------------------------------------------------

def run_detectors(
    x: Array,
    detectors: dict,
    X_secondary: "Array | None" = None,
) -> dict[str, np.ndarray]:
    """Run a dict of detectors on primary data and return statistics.

    Parameters
    ----------
    x : (n_trials, 2m)  primary data
    detectors : {name: detector_instance}
        Known-M detectors are called as ``det.compute(x)``.
        Adaptive detectors are called as ``det.compute(x, X_secondary=X_sec)``.
    X_secondary : (n_trials, K, 2m)  or None

    Returns
    -------
    dict {name: (n_trials,) np.ndarray}
    """
    from ..core.backend import to_numpy
    results = {}
    for name, det in detectors.items():
        from .detectors import AdaptiveSonarDetector
        if isinstance(det, AdaptiveSonarDetector):
            if X_secondary is None:
                raise ValueError(f"Detector {name!r} requires X_secondary.")
            stat = det.compute(x, X_secondary=X_secondary)
        else:
            stat = det.compute(x)
        results[name] = np.asarray(to_numpy(stat), dtype=np.float64)
    return results


# ---------------------------------------------------------------------------
# Standalone plot-script templates
# ---------------------------------------------------------------------------


_MC_PFA_THRESHOLD_TEMPLATE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

$style_code
parser = argparse.ArgumentParser()
parser.add_argument("--tikz", action="store_true")
parser.add_argument("--no-save", action="store_true")
parser.add_argument("--use-latex", action="store_true")
args = parser.parse_args()
if args.use_latex:
    import matplotlib as _mpl2
    _mpl2.rcParams.update({"text.usetex": True})

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr
data = np.load(here / (stem + ".npz"), allow_pickle=True)
detector_names = data["detector_names"].tolist()
colors = ["#5ca8d3", "#e06b6b", "#a57bc5", "#6bbf6b", "#e0a050", "#50c8c0"]

fig, ax = plt.subplots(figsize=(7, 5))
for i, name in enumerate(detector_names):
    ax.semilogy(data[f"thresh_{name}"], data[f"pfa_{name}"],
                color=colors[i % len(colors)], label=name)
ax.set_xlabel("Threshold")
ax.set_ylabel("PFA")
ax.set_title(title)
ax.legend()
fig.tight_layout()
if not args.no_save:
    out = here / (stem + "_pfa_threshold.pdf")
    fig.savefig(out); print(f"Saved {out}")
    if args.tikz:
        import matplot2tikz; matplot2tikz.save(str(here / (stem + "_pfa_threshold.tex")))
plt.show()
""")

_MC_PD_SNR_TEMPLATE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

$style_code
parser = argparse.ArgumentParser()
parser.add_argument("--tikz", action="store_true")
parser.add_argument("--no-save", action="store_true")
parser.add_argument("--use-latex", action="store_true")
args = parser.parse_args()
if args.use_latex:
    import matplotlib as _mpl2
    _mpl2.rcParams.update({"text.usetex": True})

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr
data = np.load(here / (stem + ".npz"), allow_pickle=True)
detector_names = data["detector_names"].tolist()
snr_db = data["snr_db"]
colors = ["#5ca8d3", "#e06b6b", "#a57bc5", "#6bbf6b", "#e0a050", "#50c8c0"]
styles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]

fig, ax = plt.subplots(figsize=(8, 5))
for i, name in enumerate(detector_names):
    ax.plot(snr_db, data[f"pd_{name}"],
            color=colors[i % len(colors)],
            linestyle=styles[i % len(styles)],
            label=name)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("PD")
ax.set_ylim(-0.02, 1.02)
ax.set_title(title)
ax.legend()
fig.tight_layout()
if not args.no_save:
    out = here / (stem + "_pd_snr.pdf")
    fig.savefig(out); print(f"Saved {out}")
    if args.tikz:
        import matplot2tikz; matplot2tikz.save(str(here / (stem + "_pd_snr.tex")))
plt.show()
""")

_MC_PD_ANGLE_TEMPLATE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

$style_code
parser = argparse.ArgumentParser()
parser.add_argument("--tikz", action="store_true")
parser.add_argument("--no-save", action="store_true")
parser.add_argument("--detector", default=None)
args = parser.parse_args()

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr
data = np.load(here / (stem + ".npz"), allow_pickle=True)
detector_names = data["detector_names"].tolist()
theta_grid = data["theta_grid"]

to_plot = [args.detector] if args.detector else detector_names
for name in to_plot:
    if f"pd_map_{name}" not in data:
        print(f"Detector {name!r} not found. Available: {detector_names}"); continue
    pd_map = data[f"pd_map_{name}"]  # (n_theta, n_theta)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(theta_grid, theta_grid, pd_map,
                        vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, ax=ax, label="PD")
    ax.set_xlabel(r"$$\\theta_1$$ (deg)")
    ax.set_ylabel(r"$$\\theta_2$$ (deg)")
    ax.set_title(title + f" — {name}")
    fig.tight_layout()
    if not args.no_save:
        out = here / (stem + f"_pd_angle_{name}.pdf")
        fig.savefig(out); print(f"Saved {out}")
plt.show()
""")

_MC_TYLER_CONV_TEMPLATE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

$style_code
parser = argparse.ArgumentParser()
parser.add_argument("--tikz", action="store_true")
parser.add_argument("--no-save", action="store_true")
args = parser.parse_args()

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr
data = np.load(here / (stem + ".npz"), allow_pickle=True)
iterations = data["iterations"]
deviations = data["deviations_mean"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(iterations, deviations, color="#5ca8d3")
ax.axhline(1e-6, color="#e06b6b", linestyle="--", linewidth=1.2, label=r"$$10^{-6}$$")
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Relative deviation $$\\|\\hat{M}^{(k)}-\\hat{M}^{(k-1)}\\| / \\|\\hat{M}^{(k-1)}\\|$$")
ax.set_title(title)
ax.legend()
fig.tight_layout()
if not args.no_save:
    out = here / (stem + "_convergence.pdf")
    fig.savefig(out); print(f"Saved {out}")
    if args.tikz:
        import matplot2tikz; matplot2tikz.save(str(here / (stem + "_convergence.tex")))
plt.show()
""")
