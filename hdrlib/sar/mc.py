# SAR-specific MC simulation utilities.

from __future__ import annotations

import argparse
import logging

import numpy as np

from ..core.mc import (
    add_mc_base_args,
)
from ..core.plot_style import apply_style

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Online detector helpers
# ---------------------------------------------------------------------------

def online_single_pass(data, T_vec, detector):
    """Stream data through detector, recording the statistic at each T checkpoint."""
    T_set = set(T_vec)
    T_max = data.shape[-3]
    results = {}
    detector.reset_state()
    stat = detector.initialize(data[..., :2, :, :])
    if 2 in T_set:
        results[2] = stat
    for t in range(2, T_max):
        T_current = t + 1
        stat = detector.update(stat, data[..., t, :, :])
        if T_current in T_set:
            results[T_current] = stat
    return results


def online_run(data, detector):
    """Run online detector through all T dates, return final statistic."""
    T = data.shape[-3]
    detector.reset_state()
    stat = detector.initialize(data[..., :2, :, :])
    for t in range(2, T):
        stat = detector.update(stat, data[..., t, :, :])
    return stat


# ---------------------------------------------------------------------------
# SAR-specific CLI arguments
# ---------------------------------------------------------------------------

def add_mc_args(parser: argparse.ArgumentParser) -> None:
    """Add all MC simulation CLI arguments (base + SAR-specific) to *parser*."""
    add_mc_base_args(parser)
    parser.add_argument("--n-features", type=int, default=8, metavar="P",
        help="Feature dimension p; n_samples is fixed to 2*p+1 (default 8).")
    parser.add_argument("--T-max", type=int, default=1000,
        help="Maximum number of time steps (default 1000).")
    parser.add_argument("--T-min", type=int, default=5,
        help="Minimum number of time steps (default 5).")
    parser.add_argument("--n-T", type=int, default=30,
        help="Number of T values in log scale (default 30).")
    parser.add_argument("--sigma-seed", type=int, default=0,
        help="Seed for Sigma_true generation, independent from --seed (default 0).")


# ---------------------------------------------------------------------------
# H1-specific CLI arguments (call after add_mc_args)
# ---------------------------------------------------------------------------

def add_mc_h1_args(parser: argparse.ArgumentParser) -> None:
    """Add H1-specific CLI arguments: second Sigma, change point, PFA."""
    parser.add_argument("--sigma2-seed", type=int, default=1,
        help="Seed for Sigma_2 (H1 distribution, default 1 — different from --sigma-seed).")
    parser.add_argument("--change-fraction", type=float, default=0.5,
        help="Change point as a fraction of T, so n_change_dates = max(2, int(T * change_fraction)). "
             "Default 0.5 — change at midpoint, ensuring equal pre/post evidence at every T.")
    parser.add_argument("--pfa", type=float, default=1e-3,
        help="Target false alarm probability for power estimation (default 1e-3). "
             "Reliable threshold estimation requires at least 10/PFA H0 trials.")


# ---------------------------------------------------------------------------
# Aggregation (shared between mc_gaussian and mc_dcg)
# ---------------------------------------------------------------------------

def aggregate(
    online_dict: dict[int, "np.ndarray | float"],
    offline_dict: dict[int, "np.ndarray | float"],
    T_vec: list[int],
) -> dict:
    """Stack per-T results over trials and compute mean ± std.

    Both dicts map T -> scalar (Pool path) or (n_trials,) array (batched path).
    """
    T_arr = np.asarray(T_vec)
    online_arr = np.stack([np.asarray(online_dict[T]) for T in T_vec], axis=-1)   # (n_trials, n_T)
    offline_arr = np.stack([np.asarray(offline_dict[T]) for T in T_vec], axis=-1)
    diff = np.abs(online_arr - offline_arr)
    return {
        "T": T_arr,
        "S_online_mean": online_arr.mean(0),
        "S_online_std": online_arr.std(0),
        "S_offline_mean": offline_arr.mean(0),
        "S_offline_std": offline_arr.std(0),
        "diff_mean": diff.mean(0),
        "diff_std": diff.std(0),
    }


# ---------------------------------------------------------------------------
# H1 aggregation: empirical power curves
# ---------------------------------------------------------------------------

def aggregate_power(
    h0_stats: dict,
    h1_stats: dict,
    T_vec: list[int],
    pfa: float,
) -> dict:
    """Compute empirical power curves for online and offline detectors.

    Parameters
    ----------
    h0_stats : {"online": {T: (n_trials,)}, "offline": {T: (n_trials,)}}
        Statistics under H0 — used to set detector-specific thresholds at *pfa*.
    h1_stats : same structure but under H1.
    T_vec : list of T values.
    pfa : target false alarm probability.

    Returns
    -------
    dict with keys: T, power_online, power_offline, pfa
    """
    T_arr = np.asarray(T_vec)
    power_online = np.zeros(len(T_vec))
    power_offline = np.zeros(len(T_vec))
    first_T = T_vec[0]
    n_trials = len(np.asarray(h0_stats["online"][first_T]))

    for i, T in enumerate(T_vec):
        h0_on = np.asarray(h0_stats["online"][T])
        h0_off = np.asarray(h0_stats["offline"][T])
        h1_on = np.asarray(h1_stats["online"][T])
        h1_off = np.asarray(h1_stats["offline"][T])

        thresh_on = np.percentile(h0_on, 100.0 * (1.0 - pfa))
        thresh_off = np.percentile(h0_off, 100.0 * (1.0 - pfa))

        power_online[i] = np.mean(h1_on > thresh_on)
        power_offline[i] = np.mean(h1_off > thresh_off)

    return {
        "T": T_arr,
        "power_online": power_online,
        "power_offline": power_offline,
        "pfa": np.array(pfa),
        "n_trials": np.array(n_trials),
    }


# ---------------------------------------------------------------------------
# H1 power-curve plot template
# ---------------------------------------------------------------------------

from string import Template

_MC_PLOT_TEMPLATE_H1 = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
import argparse
from pathlib import Path

import numpy as np
import matplotlib as _mpl
import matplotlib.pyplot as plt

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

parser = argparse.ArgumentParser("Plot MC power curve results.")
parser.add_argument("--tikz", action="store_true")
parser.add_argument("--no-save", action="store_true")
parser.add_argument("--use-latex", action="store_true", help="Enable LaTeX rendering (requires a LaTeX install).")
args = parser.parse_args()

if args.use_latex:
    try:
        _mpl.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\\\\usepackage{amsmath}"})
    except Exception:
        pass

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr
stats = np.load(here / (stem + ".npz"), allow_pickle=True)
T = stats["T"]
pfa = float(stats["pfa"])
n_trials = int(stats["n_trials"]) if "n_trials" in stats else None

_BLUE  = "#5ca8d3"
_CORAL = "#e06b6b"

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(T, stats["power_offline"], color=_BLUE,  marker="o", markersize=3, label="offline")
ax.semilogx(T, stats["power_online"],  color=_CORAL, linestyle="--", marker="s", markersize=3, label="online")
if n_trials is not None:
    se_off = np.sqrt(stats["power_offline"] * (1 - stats["power_offline"]) / n_trials)
    ax.fill_between(T, np.clip(stats["power_offline"] - se_off, 0, 1),
                       np.clip(stats["power_offline"] + se_off, 0, 1),
                    alpha=0.18, color=_BLUE)
    se_on = np.sqrt(stats["power_online"] * (1 - stats["power_online"]) / n_trials)
    ax.fill_between(T, np.clip(stats["power_online"] - se_on, 0, 1),
                       np.clip(stats["power_online"] + se_on, 0, 1),
                    alpha=0.18, color=_CORAL)
ax.axhline(pfa, color="#6b7280", linestyle=":", linewidth=1.2, label=f"PFA = {pfa:.2e}")
ax.set_xlabel("T (number of dates)")
ax.set_ylabel("Empirical power")
ax.set_ylim(-0.02, 1.05)
ax.set_title(title if title else "Detection power vs T")
ax.legend()
fig.tight_layout()

if not args.no_save:
    out = here / (stem + "_power.pdf")
    fig.savefig(out)
    print(f"Saved {out}")
    if args.tikz:
        import matplot2tikz
        matplot2tikz.save(str(here / (stem + "_power.tex")))

plt.show()
""")


# ---------------------------------------------------------------------------
# Interactive plotting
# ---------------------------------------------------------------------------

def plot_mc_stats(stats: dict, title: str = "", use_latex: bool = False) -> None:
    """Display the two standard MC convergence figures interactively.

    Calls matplotlib.pyplot.show() — meant to be called at end of a script
    when --show-interactive is passed.
    """
    import matplotlib.pyplot as plt

    apply_style(use_latex=use_latex)

    T = stats["T"]
    _BLUE   = "#5ca8d3"
    _CORAL  = "#e06b6b"
    _VIOLET = "#a57bc5"

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(T, stats["S_offline_mean"], color=_BLUE, label="S_offline")
    ax1.fill_between(T,
        stats["S_offline_mean"] - stats["S_offline_std"],
        stats["S_offline_mean"] + stats["S_offline_std"],
        alpha=0.18, color=_BLUE)
    ax1.plot(T, stats["S_online_mean"], color=_CORAL, linestyle="--", label="S_online")
    ax1.fill_between(T,
        stats["S_online_mean"] - stats["S_online_std"],
        stats["S_online_mean"] + stats["S_online_std"],
        alpha=0.18, color=_CORAL)
    ax1.set_xlabel("T (number of dates)")
    ax1.set_ylabel("GLRT statistic")
    ax1.set_title(f"{title} — online vs offline" if title else "online vs offline")
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    lo = np.maximum(stats["diff_mean"] - stats["diff_std"], 1e-15)
    ax2.loglog(T, stats["diff_mean"], color=_VIOLET)
    ax2.fill_between(T, lo, stats["diff_mean"] + stats["diff_std"], alpha=0.18, color=_VIOLET)
    ax2.set_xlabel("T (number of dates)")
    ax2.set_ylabel("|S_online - S_offline|")
    ax2.set_title(f"{title} — difference (log scale)" if title else "difference (log scale)")
    fig2.tight_layout()

    plt.show()


def plot_mc_power(stats: dict, title: str = "", use_latex: bool = False) -> None:
    """Display the power-vs-T figure interactively."""
    import matplotlib.pyplot as plt

    apply_style(use_latex=use_latex)

    T = stats["T"]
    pfa = float(stats["pfa"])
    n_trials = int(stats["n_trials"]) if "n_trials" in stats else None

    _BLUE  = "#5ca8d3"
    _CORAL = "#e06b6b"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(T, stats["power_offline"], color=_BLUE,  marker="o", markersize=3, label="offline")
    ax.semilogx(T, stats["power_online"],  color=_CORAL, linestyle="--", marker="s", markersize=3, label="online")
    if n_trials is not None:
        se_off = np.sqrt(stats["power_offline"] * (1 - stats["power_offline"]) / n_trials)
        ax.fill_between(T, np.clip(stats["power_offline"] - se_off, 0, 1),
                           np.clip(stats["power_offline"] + se_off, 0, 1),
                        alpha=0.18, color=_BLUE)
        se_on = np.sqrt(stats["power_online"] * (1 - stats["power_online"]) / n_trials)
        ax.fill_between(T, np.clip(stats["power_online"] - se_on, 0, 1),
                           np.clip(stats["power_online"] + se_on, 0, 1),
                        alpha=0.18, color=_CORAL)
    ax.axhline(pfa, color="#6b7280", linestyle=":", linewidth=1.2, label=f"PFA = {pfa:.2e}")
    ax.set_xlabel("T (number of dates)")
    ax.set_ylabel("Empirical power")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{title} — power curve" if title else "power curve")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main function helpers
# ---------------------------------------------------------------------------

def finish_h0(args, exporter, online_dict, offline_dict, T_vec, stem, title, elapsed):
    """Aggregate H0 convergence stats, log summary, export, and optionally plot."""
    stats = aggregate(online_dict, offline_dict, T_vec)
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  S_offline mean range: [{stats['S_offline_mean'].min():.3g}, {stats['S_offline_mean'].max():.3g}]")
    logger.info(f"  |S_online - S_offline| mean range: [{stats['diff_mean'].min():.3g}, {stats['diff_mean'].max():.3g}]")
    exporter.save(stats, stem, elapsed, title=title)
    if args.show_interactive:
        plot_mc_stats(stats, title=title)


def finish_h1(args, exporter, h0_stats, h1_stats, T_vec, stem, title, elapsed):
    """Aggregate H1 power stats, log summary, export, and optionally plot."""
    stats = aggregate_power(h0_stats, h1_stats, T_vec, args.pfa)
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  Power offline @ T_max: {stats['power_offline'][-1]:.3f}")
    logger.info(f"  Power online  @ T_max: {stats['power_online'][-1]:.3f}")
    exporter.save(stats, stem, elapsed, title=title)
    if args.show_interactive:
        plot_mc_power(stats, title=title)
