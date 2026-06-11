# MC simulation utilities: result export, aggregation, and common CLI arguments.

from __future__ import annotations

import json
import sys
import argparse
import logging
from pathlib import Path
from string import Template

import numpy as np

_ROOT = str(Path(__file__).parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.exporter import _git_sha

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone plot script template
# ---------------------------------------------------------------------------

_MC_PLOT_TEMPLATE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
# To regenerate: re-run the simulation script with --export
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\\usepackage{amsmath}"
except Exception:
    pass

parser = argparse.ArgumentParser("Plot MC convergence results.")
parser.add_argument("--tikz", action="store_true", help="Also export as PGFPlots .tex via matplot2tikz.")
parser.add_argument("--no-save", action="store_true", help="Show only, do not save PDFs.")
args = parser.parse_args()

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr
stats = np.load(here / (stem + ".npz"))
T = stats["T"]

# --- Figure 1: S_online and S_offline vs T (linear scale) ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(T, stats["S_offline_mean"], color="steelblue", label="S_offline")
ax1.fill_between(T,
    stats["S_offline_mean"] - stats["S_offline_std"],
    stats["S_offline_mean"] + stats["S_offline_std"],
    alpha=0.25, color="steelblue")
ax1.plot(T, stats["S_online_mean"], color="tomato", linestyle="--", label="S_online")
ax1.fill_between(T,
    stats["S_online_mean"] - stats["S_online_std"],
    stats["S_online_mean"] + stats["S_online_std"],
    alpha=0.25, color="tomato")
ax1.set_xlabel("T (number of dates)")
ax1.set_ylabel("GLRT statistic")
ax1.set_title(title + " -- online vs offline")
ax1.legend()
ax1.grid(True, alpha=0.4)
fig1.tight_layout()
if not args.no_save:
    out = here / (stem + "_convergence.pdf")
    fig1.savefig(out)
    print(f"Saved {out}")
    if args.tikz:
        import matplot2tikz
        matplot2tikz.save(str(here / (stem + "_convergence.tex")))

# --- Figure 2: |S_online - S_offline| log-log ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
lo = np.maximum(stats["diff_mean"] - stats["diff_std"], 1e-15)
ax2.loglog(T, stats["diff_mean"], color="purple")
ax2.fill_between(T, lo, stats["diff_mean"] + stats["diff_std"], alpha=0.25, color="purple")
ax2.set_xlabel("T (number of dates)")
ax2.set_ylabel("|S_online - S_offline|")
ax2.set_title(title + " -- difference (log scale)")
ax2.grid(True, which="both", alpha=0.4)
fig2.tight_layout()
if not args.no_save:
    out = here / (stem + "_difference.pdf")
    fig2.savefig(out)
    print(f"Saved {out}")
    if args.tikz:
        import matplot2tikz
        matplot2tikz.save(str(here / (stem + "_difference.tex")))

plt.show()
""")


# ---------------------------------------------------------------------------
# Result exporter
# ---------------------------------------------------------------------------

class MCResultExporter:
    """Save MC stats with provenance sidecar and standalone plot script.

    Saves three files per result:
      {stem}.npz       — stats arrays
      {stem}.json      — provenance (git SHA, script path, args, timing)
      {stem}_plot.py   — standalone plot script; supports --tikz for PGFPlots export

    Pass plot_template to override the default H0 convergence template (e.g. for H1 power curves).
    """

    def __init__(self, args, export_path: Path, stem_suffix: str,
                 plot_template: "Template | None" = None) -> None:
        self._args = args
        self._export_path = Path(export_path)
        self._stem_suffix = stem_suffix
        self._plot_template = plot_template if plot_template is not None else _MC_PLOT_TEMPLATE
        self._provenance_base = {
            "git_sha": _git_sha(),
            "script": str(Path(sys.argv[0]).resolve()),
            "args": vars(args),
        }

    @property
    def active(self) -> bool:
        return getattr(self._args, "export", False)

    def save(self, stats: dict, stem: str, elapsed: float, title: str = "") -> "Path | None":
        if not self.active:
            return None

        self._export_path.mkdir(parents=True, exist_ok=True)
        full_stem = f"{stem}_{self._stem_suffix}"
        out = self._export_path / full_stem

        np.savez(str(out) + ".npz", **stats)

        provenance = {
            **self._provenance_base,
            "elapsed_s": round(elapsed, 3),
            "T_min": int(stats["T"].min()),
            "T_max": int(stats["T"].max()),
            "n_T": int(len(stats["T"])),
            "title": title,
        }
        Path(str(out) + ".json").write_text(json.dumps(provenance, indent=2))
        Path(str(out) + "_plot.py").write_text(
            self._plot_template.substitute(stem_repr=repr(full_stem), title_repr=repr(title))
        )

        logger.info(f"Exported to {self._export_path}/")
        logger.info(f"  {full_stem}.npz      — stats arrays")
        logger.info(f"  {full_stem}.json     — provenance sidecar")
        logger.info(f"  {full_stem}_plot.py  — standalone plot script")
        return out


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
# Common CLI arguments
# ---------------------------------------------------------------------------

def add_mc_args(parser: argparse.ArgumentParser) -> None:
    """Add common MC simulation CLI arguments to *parser*."""
    parser.add_argument("--n-features", type=int, default=8, metavar="P",
        help="Feature dimension p; n_samples is fixed to 2*p+1 (default 8).")
    parser.add_argument("--n-trials", type=int, default=10000,
        help="Number of Monte-Carlo trials (default 10000).")
    parser.add_argument("--T-max", type=int, default=1000,
        help="Maximum number of time steps (default 1000).")
    parser.add_argument("--T-min", type=int, default=5,
        help="Minimum number of time steps (default 5).")
    parser.add_argument("--n-T", type=int, default=30,
        help="Number of T values in log scale (default 30).")
    parser.add_argument("--seed", type=int, default=42,
        help="RNG seed for data generation (default 42).")
    parser.add_argument("--sigma-seed", type=int, default=0,
        help="Seed for Sigma_true generation, independent from --seed (default 0).")
    parser.add_argument(
        "--backend", type=str, default="numpy",
        choices=["numpy", "torch-cpu", "torch-cuda", "torch-mps",
                 "jax-cpu", "jax-cuda", "jax-metal", "cupy"],
        help="Compute backend. numpy → multiprocessing.Pool (one worker per trial); "
             "all others → trials stacked in leading batch dimension (default numpy).")
    parser.add_argument("--n-workers", type=int, default=None,
        help="Pool workers for numpy backend (default: os.cpu_count()).")
    parser.add_argument("--export", action="store_true",
        help="Save .npz results + provenance sidecar + plot script.")
    parser.add_argument("--export-path", type=str, default="./exports",
        help="Directory for exported results (default: ./exports).")
    parser.add_argument("--show-interactive", action="store_true",
        help="Display figures interactively at the end of the simulation.")


# ---------------------------------------------------------------------------
# Interactive plotting
# ---------------------------------------------------------------------------

def plot_mc_stats(stats: dict, title: str = "") -> None:
    """Display the two standard MC convergence figures interactively.

    Calls matplotlib.pyplot.show() — meant to be called at end of a script
    when --show-interactive is passed.
    """
    import matplotlib.pyplot as plt

    T = stats["T"]

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(T, stats["S_offline_mean"], color="steelblue", label="S_offline")
    ax1.fill_between(T,
        stats["S_offline_mean"] - stats["S_offline_std"],
        stats["S_offline_mean"] + stats["S_offline_std"],
        alpha=0.25, color="steelblue")
    ax1.plot(T, stats["S_online_mean"], color="tomato", linestyle="--", label="S_online")
    ax1.fill_between(T,
        stats["S_online_mean"] - stats["S_online_std"],
        stats["S_online_mean"] + stats["S_online_std"],
        alpha=0.25, color="tomato")
    ax1.set_xlabel("T (number of dates)")
    ax1.set_ylabel("GLRT statistic")
    ax1.set_title(f"{title} — online vs offline" if title else "online vs offline")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    lo = np.maximum(stats["diff_mean"] - stats["diff_std"], 1e-15)
    ax2.loglog(T, stats["diff_mean"], color="purple")
    ax2.fill_between(T, lo, stats["diff_mean"] + stats["diff_std"], alpha=0.25, color="purple")
    ax2.set_xlabel("T (number of dates)")
    ax2.set_ylabel("|S_online - S_offline|")
    ax2.set_title(f"{title} — difference (log scale)" if title else "difference (log scale)")
    ax2.grid(True, which="both", alpha=0.4)
    fig2.tight_layout()

    plt.show()


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

try:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\\usepackage{amsmath}"
except Exception:
    pass

parser = argparse.ArgumentParser("Plot MC power curve results.")
parser.add_argument("--tikz", action="store_true")
parser.add_argument("--no-save", action="store_true")
args = parser.parse_args()

here = Path(__file__).parent
stem = $stem_repr
title = $title_repr
stats = np.load(here / (stem + ".npz"), allow_pickle=True)
T = stats["T"]
pfa = float(stats["pfa"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(T, stats["power_offline"], color="steelblue", marker="o", markersize=3, label="offline")
ax.semilogx(T, stats["power_online"],  color="tomato", linestyle="--", marker="s", markersize=3, label="online")
ax.axhline(pfa, color="gray", linestyle=":", linewidth=1, label=f"PFA = {pfa:.2f}")
ax.set_xlabel("T (number of dates)")
ax.set_ylabel("Empirical power")
ax.set_ylim(-0.02, 1.05)
ax.set_title(title if title else "Detection power vs T")
ax.legend()
ax.grid(True, which="both", alpha=0.4)
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
    }


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
# H1 interactive plot
# ---------------------------------------------------------------------------

def plot_mc_power(stats: dict, title: str = "") -> None:
    """Display the power-vs-T figure interactively."""
    import matplotlib.pyplot as plt

    T = stats["T"]
    pfa = float(stats["pfa"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(T, stats["power_offline"], color="steelblue", marker="o", markersize=3, label="offline")
    ax.semilogx(T, stats["power_online"], color="tomato", linestyle="--", marker="s", markersize=3, label="online")
    ax.axhline(pfa, color="gray", linestyle=":", linewidth=1, label=f"PFA = {pfa:.2f}")
    ax.set_xlabel("T (number of dates)")
    ax.set_ylabel("Empirical power")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{title} — power curve" if title else "power curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    plt.show()
