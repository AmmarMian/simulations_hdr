# SAR-specific MC simulation utilities.

from __future__ import annotations

import argparse
import logging
from string import Template

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

_MC_PLOT_TEMPLATE_H1 = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

$style_code
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


# ---------------------------------------------------------------------------
# MSE / ICRB aggregation and plotting (Kronecker scaled Gaussian estimation)
# ---------------------------------------------------------------------------

def aggregate_mse(
    online_err: dict,
    offline_err: dict,
    icrb: dict,
    T_vec: list[int],
) -> dict:
    """Average per-component squared Riemannian errors over trials.

    Parameters
    ----------
    online_err, offline_err : {component: {T: (n_trials,) array}}
        Components are "A", "B", "tau", "total".
    icrb : {component: array over T_vec}
    T_vec : list of T values.

    Returns
    -------
    dict of flat arrays, keys "T", "<comp>_online", "<comp>_offline",
    "<comp>_online_se", "<comp>_offline_se", "<comp>_icrb".
    """
    out = {"T": np.asarray(T_vec)}
    for comp in ("A", "B", "tau", "total"):
        for label, src in (("online", online_err), ("offline", offline_err)):
            arr = np.stack([np.asarray(src[comp][T]).ravel() for T in T_vec], axis=-1)
            out[f"{comp}_{label}"] = arr.mean(0)
            out[f"{comp}_{label}_se"] = arr.std(0) / np.sqrt(arr.shape[0])
        out[f"{comp}_icrb"] = np.asarray(icrb[comp])
    return out


_MC_PLOT_TEMPLATE_MSE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
# To regenerate: re-run the simulation script with --export
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

$style_dict

parser = argparse.ArgumentParser("Plot MSE vs T against the ICRB.")
parser.add_argument("--tikz", action="store_true",
    help="Export PGFPlots .tex for the dissertation (light background).")
parser.add_argument("--no-save", action="store_true", help="Show only, do not save.")
parser.add_argument("--use-latex", action="store_true", help="LaTeX text rendering.")
args = parser.parse_args()

# The dark theme is for reading on screen. Exported to PGFPlots it would paint a
# black background into a manuscript printed on white, so it is applied on the
# viewing path only -- never on the one that writes the .tex.
if not args.tikz:
    import matplotlib as _mpl
    _mpl.rcParams.update(_DARK_STYLE)
    if args.use_latex:
        _mpl.rcParams.update({"text.usetex": True})

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr

# Legends sit outside the axes: inside, the curves run under the entries and
# neither is readable at the size a figure takes in the manuscript.
_LEGEND = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

data = np.load(here / (stem + ".npz"))
T = data["T"]

_BLUE  = "#5ca8d3"
_CORAL = "#e06b6b"
_GREY  = "#6b7280"
_LABELS = {"A": r"$$\\delta^2(\\widehat{A}, A)$$",
           "B": r"$$\\delta^2(\\widehat{B}, B)$$",
           "tau": r"$$\\delta^2(\\widehat{\\tau}, \\tau)$$"}

fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
for i, (ax, comp) in enumerate(zip(axes, ("A", "B", "tau"))):
    # Labelled on the first panel only. matplot2tikz emits an \\addlegendentry
    # for every labelled curve and gathers them into one legend, so labelling
    # the three panels would print the same three entries three times.
    lbl = (lambda name: name) if i == 0 else (lambda name: None)
    ax.loglog(T, data[comp + "_offline"], color=_BLUE, marker="o", markersize=3,
              label=lbl("hors ligne"))
    ax.loglog(T, data[comp + "_online"], color=_CORAL, marker="s", markersize=3,
              linestyle="--", label=lbl("en ligne"))
    ax.loglog(T, data[comp + "_icrb"], color=_GREY, linestyle=":", label=lbl("ICRB"))
    ax.set_ylabel(_LABELS[comp])
    if i == 0:
        ax.legend(**_LEGEND)
axes[-1].set_xlabel(r"$$T$$")
if not args.tikz:
    fig.suptitle(title)
fig.tight_layout()


def _export(fig, suffix):
    if args.no_save:
        return
    out = here / (stem + suffix + ".pdf")
    fig.savefig(out)
    print(f"Saved {out}")
    if args.tikz:
        from hdrlib.core.exporter import save_tikz
        # Panels are stacked, each as wide as the text block: side by side at
        # 0.45\\textwidth they end up too small to read once printed.
        save_tikz(str(here / (stem + suffix + ".tex")),
                  axis_width=r"\\textwidth", axis_height="5cm")

_export(fig, "_mse")
plt.show()
""")


def finish_mse(args, exporter, online_err, offline_err, icrb, T_vec, stem, title, elapsed):
    """Aggregate MSE/ICRB stats, log a summary, and export."""
    stats = aggregate_mse(online_err, offline_err, icrb, T_vec)
    logger.info(f"Done in {elapsed:.1f}s")
    for comp in ("A", "B", "tau"):
        logger.info(
            f"  {comp:>3}  @T={T_vec[-1]}: offline {stats[comp + '_offline'][-1]:.3g} | "
            f"online {stats[comp + '_online'][-1]:.3g} | ICRB {stats[comp + '_icrb'][-1]:.3g}"
        )
    exporter.save(stats, stem, elapsed, title=title)


# ---------------------------------------------------------------------------
# Structured vs unstructured estimation as a function of N
# ---------------------------------------------------------------------------

def aggregate_struct(kron_err: dict, full_err: dict, icrb_kron, icrb_full, N_vec) -> dict:
    """Average total squared errors over trials for both parametrisations."""
    N_arr = np.asarray(N_vec)
    kron = np.stack([np.asarray(kron_err[N]).ravel() for N in N_vec], axis=-1)
    full = np.stack([np.asarray(full_err[N]).ravel() for N in N_vec], axis=-1)
    return {
        "N": N_arr,
        "kron_mean": kron.mean(0),
        "kron_se": kron.std(0) / np.sqrt(kron.shape[0]),
        "full_mean": full.mean(0),
        "full_se": full.std(0) / np.sqrt(full.shape[0]),
        "kron_icrb": np.asarray(icrb_kron),
        "full_icrb": np.asarray(icrb_full),
    }


_MC_PLOT_TEMPLATE_STRUCT = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
# To regenerate: re-run the simulation script with --export
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

$style_dict

parser = argparse.ArgumentParser("Plot structured vs unstructured error against N.")
parser.add_argument("--tikz", action="store_true",
    help="Export PGFPlots .tex for the dissertation (light background).")
parser.add_argument("--no-save", action="store_true", help="Show only, do not save.")
parser.add_argument("--use-latex", action="store_true", help="LaTeX text rendering.")
args = parser.parse_args()

# The dark theme is for reading on screen. Exported to PGFPlots it would paint a
# black background into a manuscript printed on white, so it is applied on the
# viewing path only -- never on the one that writes the .tex.
if not args.tikz:
    import matplotlib as _mpl
    _mpl.rcParams.update(_DARK_STYLE)
    if args.use_latex:
        _mpl.rcParams.update({"text.usetex": True})

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr

# Legends sit outside the axes: inside, the curves run under the entries and
# neither is readable at the size a figure takes in the manuscript.
_LEGEND = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

data = np.load(here / (stem + ".npz"))
N = data["N"]

_BLUE  = "#5ca8d3"
_CORAL = "#e06b6b"

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.loglog(N, data["kron_mean"], color=_BLUE, marker="o", markersize=3, label="Kronecker")
ax.loglog(N, data["kron_icrb"], color=_BLUE, linestyle=":", label="ICRB Kronecker")
ax.loglog(N, data["full_mean"], color=_CORAL, marker="s", markersize=3, label="non structuré")
ax.loglog(N, data["full_icrb"], color=_CORAL, linestyle=":", label="ICRB non structuré")
ax.set_xlabel(r"$$N$$")
ax.set_ylabel(r"$$\\delta^2_{\\mathcal{M}}$$")
ax.legend(**_LEGEND)
if not args.tikz:
    ax.set_title(title)
fig.tight_layout()


def _export(fig, suffix):
    if args.no_save:
        return
    out = here / (stem + suffix + ".pdf")
    fig.savefig(out)
    print(f"Saved {out}")
    if args.tikz:
        from hdrlib.core.exporter import save_tikz
        save_tikz(str(here / (stem + suffix + ".tex")),
                  axis_width=r"\\textwidth", axis_height="5cm")

_export(fig, "_struct")
plt.show()
""")


def finish_struct(args, exporter, kron_err, full_err, icrb_kron, icrb_full, N_vec, stem, title, elapsed):
    """Aggregate structured/unstructured stats, log a summary, and export."""
    stats = aggregate_struct(kron_err, full_err, icrb_kron, icrb_full, N_vec)
    logger.info(f"Done in {elapsed:.1f}s")
    for i, N in enumerate(N_vec):
        logger.info(
            f"  N={N:>3}: kron {stats['kron_mean'][i]:.3g} (ICRB {stats['kron_icrb'][i]:.3g}) | "
            f"full {stats['full_mean'][i]:.3g} (ICRB {stats['full_icrb'][i]:.3g}) | "
            f"gain x{stats['full_mean'][i] / stats['kron_mean'][i]:.2f}"
        )
    exporter.save(stats, stem, elapsed, title=title)


# ---------------------------------------------------------------------------
# Power curves for several detectors at once
# ---------------------------------------------------------------------------

def aggregate_power_multi(h0: dict, h1: dict, T_vec: list[int], pfa: float) -> dict:
    """Empirical power of several detectors, thresholds set per detector and per T.

    Parameters
    ----------
    h0, h1 : {detector name: {T: (n_trials,) array of statistics}}
    T_vec : list of T values.
    pfa : target false alarm probability.

    Returns
    -------
    dict with "T", "pfa", "detectors" (list of names) and "<name>_power".
    """
    out = {"T": np.asarray(T_vec), "pfa": np.asarray(pfa),
           "detectors": np.asarray(sorted(h0.keys()))}
    for name in h0:
        power = np.zeros(len(T_vec))
        for i, T in enumerate(T_vec):
            s0 = np.asarray(h0[name][T]).ravel()
            s1 = np.asarray(h1[name][T]).ravel()
            finite = s0[np.isfinite(s0)]
            threshold = np.quantile(finite, 1 - pfa) if finite.size else np.inf
            power[i] = float(np.mean(s1 > threshold))
        out[f"{name}_power"] = power
    return out


_MC_PLOT_TEMPLATE_POWER_MULTI = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
# To regenerate: re-run the simulation script with --export
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

$style_dict

parser = argparse.ArgumentParser("Plot power vs T for several detectors.")
parser.add_argument("--tikz", action="store_true",
    help="Export PGFPlots .tex for the dissertation (light background).")
parser.add_argument("--no-save", action="store_true", help="Show only, do not save.")
parser.add_argument("--use-latex", action="store_true", help="LaTeX text rendering.")
args = parser.parse_args()

# The dark theme is for reading on screen. Exported to PGFPlots it would paint a
# black background into a manuscript printed on white, so it is applied on the
# viewing path only -- never on the one that writes the .tex.
if not args.tikz:
    import matplotlib as _mpl
    _mpl.rcParams.update(_DARK_STYLE)
    if args.use_latex:
        _mpl.rcParams.update({"text.usetex": True})

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr

# Legends sit outside the axes: inside, the curves run under the entries and
# neither is readable at the size a figure takes in the manuscript.
_LEGEND = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

data = np.load(here / (stem + ".npz"), allow_pickle=True)
T = data["T"]
pfa = float(data["pfa"])

_STYLE = {
    "K-SG":   ("#5ca8d3", "-",  "o"),
    "K-SG-O": ("#5ca8d3", "--", "s"),
    "SG":     ("#e06b6b", "-",  "o"),
    "SG-O":   ("#e06b6b", "--", "s"),
    "G":      ("#a57bc5", "-.", "^"),
}

fig, ax = plt.subplots(figsize=(7, 4.4))
for name in [str(d) for d in data["detectors"]]:
    color, ls, marker = _STYLE.get(name, ("#6b7280", "-", "x"))
    ax.semilogx(T, data[name + "_power"], color=color, linestyle=ls, marker=marker,
                markersize=3, label=name)
ax.axhline(pfa, color="#6b7280", linestyle=":", linewidth=1.0)
ax.set_xlabel(r"$$T$$")
ax.set_ylabel("puissance")
ax.set_ylim(-0.02, 1.05)
ax.legend(**_LEGEND)
if not args.tikz:
    ax.set_title(title)
fig.tight_layout()


def _export(fig, suffix):
    if args.no_save:
        return
    out = here / (stem + suffix + ".pdf")
    fig.savefig(out)
    print(f"Saved {out}")
    if args.tikz:
        from hdrlib.core.exporter import save_tikz
        save_tikz(str(here / (stem + suffix + ".tex")),
                  axis_width=r"\\textwidth", axis_height="5cm")

_export(fig, "_power")
plt.show()
""")


def finish_power_multi(args, exporter, h0, h1, T_vec, stem, title, elapsed):
    """Aggregate multi-detector power stats, log a summary, and export."""
    stats = aggregate_power_multi(h0, h1, T_vec, args.pfa)
    logger.info(f"Done in {elapsed:.1f}s")
    for name in sorted(h0.keys()):
        logger.info(f"  {name:>6}: power @T={T_vec[-1]} = {stats[name + '_power'][-1]:.3f}")
    exporter.save(stats, stem, elapsed, title=title)


def aggregate_roc_multi(h0: dict, h1: dict, T_vec: list[int],
                        n_points: int = 200) -> dict:
    """Empirical ROC of several detectors, one curve per detector and per T.

    The Pfa grid stops at 10 / n_trials: below that the threshold rests on
    fewer than ten H0 exceedances and the curve is drawing sampling noise.

    Parameters
    ----------
    h0, h1 : {detector name: {T: (n_trials,) array of statistics}}
    T_vec : list of T values.
    n_points : number of Pfa samples, log-spaced.

    Returns
    -------
    dict with "T", "pfa_grid", "detectors" and "<name>_pd" of shape
    (len(T_vec), n_points).
    """
    n_min = min(np.asarray(h0[n][T]).size for n in h0 for T in T_vec)
    pfa_min = max(10.0 / n_min, 1e-6)
    pfa_grid = np.logspace(np.log10(pfa_min), 0.0, n_points)
    out = {"T": np.asarray(T_vec), "pfa_grid": pfa_grid,
           "detectors": np.asarray(sorted(h0.keys()))}
    for name in h0:
        pd = np.zeros((len(T_vec), n_points))
        for i, T in enumerate(T_vec):
            s0 = np.asarray(h0[name][T]).ravel()
            s1 = np.asarray(h1[name][T]).ravel()
            s0, s1 = s0[np.isfinite(s0)], s1[np.isfinite(s1)]
            if s0.size == 0 or s1.size == 0:
                pd[i] = np.nan
                continue
            thr = np.quantile(s0, 1.0 - pfa_grid)
            pd[i] = (s1[:, None] > thr[None, :]).mean(axis=0)
        out[f"{name}_pd"] = pd
    return out


_MC_PLOT_TEMPLATE_ROC_MULTI = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
# To regenerate: re-run the simulation script with --export
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

$style_dict

parser = argparse.ArgumentParser("Plot ROC curves for several detectors and several T.")
parser.add_argument("--tikz", action="store_true",
    help="Export PGFPlots .tex for the dissertation (light background).")
parser.add_argument("--no-save", action="store_true", help="Show only, do not save.")
parser.add_argument("--use-latex", action="store_true", help="LaTeX text rendering.")
args = parser.parse_args()

if not args.tikz:
    import matplotlib as _mpl
    _mpl.rcParams.update(_DARK_STYLE)
    if args.use_latex:
        _mpl.rcParams.update({"text.usetex": True})

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr

data = np.load(here / (stem + ".npz"), allow_pickle=True)
T = data["T"]
pfa = data["pfa_grid"]
names = [str(d) for d in data["detectors"]]

# Two panels rather than one per T: the comparison the chapter makes is offline
# against recursive, within a model. Three T on twelve curves in one panel is
# unreadable, and a three-column grid overflows the text block once exported.
_PANELS = [("non structuré", ["SG", "SG-O"]), ("Kronecker", ["K-SG", "K-SG-O"])]
_SHADE = ["#9ecae1", "#4292c6", "#08519c"]          # T croissant
_SHADE_R = ["#fcae91", "#fb6a4a", "#a50f15"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=True)
for ax, (panel, wanted) in zip(axes, _PANELS):
    for name in [n for n in wanted if n in names]:
        recursive = name.endswith("-O")
        shades = _SHADE_R if recursive else _SHADE
        for i, t in enumerate(T):
            ax.semilogx(pfa, data[name + "_pd"][i],
                        color=shades[i % len(shades)],
                        linestyle="--" if recursive else "-",
                        linewidth=1.2,
                        label=f"{name}, $$T={t}$$")
    ax.set_xlabel(r"$$P_{fa}$$")
    ax.set_ylim(-0.02, 1.05)
    if not args.tikz:
        ax.set_title(panel)
axes[0].set_ylabel(r"$$P_d$$")
axes[1].legend(loc="lower right", frameon=False, fontsize=7)
axes[0].legend(loc="lower right", frameon=False, fontsize=7)
if not args.tikz:
    fig.suptitle(title)
fig.tight_layout()


def _export(fig, suffix):
    if args.no_save:
        return
    out = here / (stem + suffix + ".pdf")
    fig.savefig(out)
    print(f"Saved {out}")
    if args.tikz:
        from hdrlib.core.exporter import save_tikz
        save_tikz(str(here / (stem + suffix + ".tex")),
                  axis_width=r"0.45\textwidth", axis_height="4.6cm")

_export(fig, "_roc")
plt.show()
""")


def finish_roc_multi(args, exporter, h0, h1, T_vec, stem, title, elapsed):
    """Aggregate multi-detector ROC stats, log a summary, and export."""
    stats = aggregate_roc_multi(h0, h1, T_vec)
    logger.info(f"Done in {elapsed:.1f}s")
    pfa = stats["pfa_grid"]
    i_ref = int(np.argmin(np.abs(pfa - 1e-2)))
    for name in sorted(h0.keys()):
        row = ", ".join(f"T={t}: {stats[name + '_pd'][j, i_ref]:.3f}"
                        for j, t in enumerate(T_vec))
        logger.info(f"  {name:>6}: Pd @Pfa={pfa[i_ref]:.1e} — {row}")
    exporter.save(stats, stem, elapsed, title=title)
