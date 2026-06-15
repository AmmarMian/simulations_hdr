"""Sonar-specific MC helpers: CLI args, aggregation, and plot templates."""

from __future__ import annotations

import argparse
import logging
from string import Template
from typing import Sequence

import numpy as np

from ..core.mc import add_mc_base_args, MCResultExporter  # re-export for convenience

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
    X_secondary: np.ndarray,
    m: int,
    iter_max: int = 500,
    tol: float = 0.0,  # 0 → run all iter_max steps
) -> np.ndarray:
    """Track 2TYL relative Frobenius deviation ||M^(k) - M^(k-1)||_F / ||M^(k-1)||_F.

    Runs a single trial (X_secondary shape: (K, 2m)) and records the
    per-iteration convergence metric.

    Parameters
    ----------
    X_secondary : (K, 2m) or (n_trials, K, 2m)
        When batched, the **mean** relative deviation across trials is returned.
    m : int
    iter_max : int
    tol : float
        Set to 0 to always run all iter_max steps.

    Returns
    -------
    deviations : (iter_max,) array
    """
    from .estimation import two_array_tyler  # local import avoids circular deps

    p = 2 * m
    K = X_secondary.shape[-2]

    if X_secondary.ndim == 2:
        X = X_secondary[None, :, :]   # (1, K, 2m) — add dummy batch
    else:
        X = X_secondary

    n = X.shape[0]
    x1 = X[:, :, :m]
    x2 = X[:, :, m:]

    M_hat = np.broadcast_to(np.eye(p, dtype=np.complex128), (n, p, p)).copy()
    deviations = np.full(iter_max, np.nan)

    eps = 1e-30
    for it in range(iter_max):
        M_inv = np.linalg.inv(M_hat)
        iM11 = M_inv[:, :m, :m]
        iM12 = M_inv[:, :m, m:]
        iM22 = M_inv[:, m:, m:]

        # Apply iMij to each sample: (n, m, m) @ (n, m, K) → (n, m, K)
        vx1  = np.swapaxes(iM11 @ np.swapaxes(x1, -1, -2), -1, -2)
        vx2  = np.swapaxes(iM22 @ np.swapaxes(x2, -1, -2), -1, -2)
        vx12 = np.swapaxes(iM12 @ np.swapaxes(x2, -1, -2), -1, -2)

        t1  = np.real((x1.conj() * vx1).sum(-1)) / m    # (n, K)
        t2  = np.real((x2.conj() * vx2).sum(-1)) / m
        t12 = np.real((x1.conj() * vx12).sum(-1)) / m

        tau1 = t1 + np.sqrt(t1 / (t2 + eps)) * t12 + eps
        tau2 = t2 + np.sqrt(t2 / (t1 + eps)) * t12 + eps

        x1s = x1 / np.sqrt(tau1[:, :, None])
        x2s = x2 / np.sqrt(tau2[:, :, None])
        xs  = np.concatenate([x1s, x2s], axis=-1)

        M_new = np.swapaxes(xs, -1, -2).conj() @ xs / K
        tr = np.real(np.diagonal(M_new, axis1=-2, axis2=-1).sum(-1))
        M_new = M_new * (p / tr[:, None, None])

        diff = M_new - M_hat
        fd = np.sqrt(np.sum(np.abs(diff.reshape(n, -1)) ** 2, axis=-1))
        fm = np.sqrt(np.sum(np.abs(M_hat.reshape(n, -1)) ** 2, axis=-1))
        rel = fd / (fm + eps)
        deviations[it] = float(rel.mean())

        M_hat = M_new
        if tol > 0 and float(rel.max()) < tol:
            deviations[it + 1:] = 0.0
            break

    return deviations


# ---------------------------------------------------------------------------
# Detector batch runner helper
# ---------------------------------------------------------------------------

def run_detectors(
    x: np.ndarray,
    detectors: dict,
    X_secondary: "np.ndarray | None" = None,
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

_RCPARAMS = """\
import matplotlib as _mpl
_mpl.rcParams.update({
    "figure.facecolor": "#14141a", "axes.facecolor": "#14141a",
    "savefig.facecolor": "#14141a",
    "text.color": "#dde3f0", "axes.labelcolor": "#dde3f0", "axes.titlecolor": "#dde3f0",
    "xtick.color": "#48485e", "ytick.color": "#48485e",
    "xtick.labelcolor": "#dde3f0", "ytick.labelcolor": "#dde3f0",
    "axes.edgecolor": "#48485e", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.which": "both",
    "grid.color": "#272733", "grid.linewidth": 0.6, "grid.linestyle": "--",
    "font.family": "serif",
    "font.serif": ["STIXTwoText", "STIX Two Text", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 1.8, "lines.markersize": 5,
    "legend.facecolor": "#1c1c27", "legend.edgecolor": "#48485e",
    "legend.framealpha": 0.9, "savefig.dpi": 200, "savefig.bbox": "tight",
})
"""

_MC_PFA_THRESHOLD_TEMPLATE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
""" + _RCPARAMS + """
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
""" + _RCPARAMS + """
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
""" + _RCPARAMS + """
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
""" + _RCPARAMS + """
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
