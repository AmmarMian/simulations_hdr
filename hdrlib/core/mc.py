# MC simulation utilities: result export, common CLI arguments, and helpers.
# Generic functions shared across modalities.

from __future__ import annotations

import json
import sys
import argparse
import logging
import time
from pathlib import Path
from string import Template

import numpy as np

from .exporter import _git_sha
from .plot_style import EMBEDDED_STYLE_CODE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Online detector helpers (shared between pool workers and batched paths)
# ---------------------------------------------------------------------------

def chunk_trial_ranges(
    n_trials: int, chunk_trials: "int | None"
) -> "tuple[list[int], int, int]":
    """Return (c_starts, chunk_size, n_chunks) for iterating over trial chunks."""
    chunk = min(chunk_trials if chunk_trials is not None else n_trials, n_trials)
    c_starts = list(range(0, n_trials, chunk))
    return c_starts, chunk, len(c_starts)


def warn_gpu_not_optimized(backend: str) -> None:
    """Warn when a GPU/accelerator backend is selected.

    The batched GPU path is functionally correct but not memory- or
    kernel-optimized: the MM loop iterates sequentially over T checkpoints
    rather than exploiting GPU parallelism across T, so utilization will be
    low.  For large n_trials, --backend numpy + Pool is often faster.
    """
    if backend != "numpy":
        logger.warning(
            f"Backend '{backend}': GPU/accelerator path is NOT optimized for these MC "
            "simulations. Operations run sequentially per T checkpoint; GPU utilization "
            "will be low. For large n_trials, --backend numpy (Pool) may be faster."
        )


def maybe_empty_cache(backend: str) -> None:
    """Release PyTorch's cached (but unused) GPU memory after each T-loop step.

    PyTorch's caching allocator retains freed tensors to amortise system calls.
    Across many iterations of a T-loop each allocating hundreds of MB of
    intermediates this cache can exhaust GPU memory even though live tensors are
    small.  Calling this after every offline.compute() prevents that build-up.
    No-op for non-CUDA backends.
    """
    if "cuda" in backend:
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Standalone plot script template (H0 convergence)
# ---------------------------------------------------------------------------

_MC_PLOT_TEMPLATE = Template("""\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
# To regenerate: re-run the simulation script with --export
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

$style_code
parser = argparse.ArgumentParser("Plot MC convergence results.")
parser.add_argument("--tikz", action="store_true", help="Also export as PGFPlots .tex via matplot2tikz.")
parser.add_argument("--no-save", action="store_true", help="Show only, do not save PDFs.")
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
stats = np.load(here / (stem + ".npz"))
T = stats["T"]

_BLUE   = "#5ca8d3"
_CORAL  = "#e06b6b"
_VIOLET = "#a57bc5"

# --- Figure 1: S_online and S_offline vs T (linear scale) ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(T, stats["S_offline_mean"], color=_BLUE,  label="S_offline")
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
ax1.set_title(title + " — online vs offline")
ax1.legend()
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
ax2.loglog(T, stats["diff_mean"], color=_VIOLET)
ax2.fill_between(T, lo, stats["diff_mean"] + stats["diff_std"], alpha=0.18, color=_VIOLET)
ax2.set_xlabel("T (number of dates)")
ax2.set_ylabel("|S_online - S_offline|")
ax2.set_title(title + " — difference (log scale)")
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

        provenance: dict = {**self._provenance_base, "elapsed_s": round(elapsed, 3), "title": title}
        if "T" in stats:
            provenance.update({
                "T_min": int(stats["T"].min()),
                "T_max": int(stats["T"].max()),
                "n_T": int(len(stats["T"])),
            })
        Path(str(out) + ".json").write_text(json.dumps(provenance, indent=2))
        Path(str(out) + "_plot.py").write_text(
            self._plot_template.substitute(
                stem_repr=repr(full_stem),
                title_repr=repr(title),
                style_code=EMBEDDED_STYLE_CODE,
            )
        )

        logger.info(f"Exported to {self._export_path}/")
        logger.info(f"  {full_stem}.npz      — stats arrays")
        logger.info(f"  {full_stem}.json     — provenance sidecar")
        logger.info(f"  {full_stem}_plot.py  — standalone plot script")
        return out


# ---------------------------------------------------------------------------
# Common base CLI arguments
# ---------------------------------------------------------------------------

def add_mc_base_args(parser: argparse.ArgumentParser) -> None:
    """Add common base MC simulation CLI arguments to *parser*."""
    parser.add_argument("--n-trials", type=int, default=10000,
        help="Number of Monte-Carlo trials (default 10000).")
    parser.add_argument("--seed", type=int, default=42,
        help="RNG seed for data generation (default 42).")
    parser.add_argument(
        "--backend", type=str, default="numpy",
        choices=["numpy", "torch-cpu", "torch-cuda", "torch-mps",
                 "jax-cpu", "jax-cuda", "jax-metal", "cupy"],
        help="Compute backend. numpy → multiprocessing.Pool (one worker per trial); "
             "all others → trials stacked in leading batch dimension (default numpy).")
    parser.add_argument("--n-workers", type=int, default=None,
        help="Pool workers for numpy backend (default: os.cpu_count()).")
    parser.add_argument("--export", action=argparse.BooleanOptionalAction, default=True,
        help="Save .npz results + provenance sidecar + plot script (default: True).")
    parser.add_argument("--storage-path", "--storage_path", "--export-path", dest="export_path",
        type=str, default="./exports",
        help="Directory for exported results; --storage-path is the qanat alias (default: ./exports).")
    parser.add_argument("--show-interactive", action="store_true",
        help="Display figures interactively at the end of the simulation.")


# ---------------------------------------------------------------------------
# Main function helpers
# ---------------------------------------------------------------------------

def make_mc_parser(description: str = "") -> argparse.ArgumentParser:
    """Return an ArgumentParser with the standard MC docstring formatter."""
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def init_logging(backend: str) -> None:
    """Configure root logging and emit the GPU disclaimer when needed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warn_gpu_not_optimized(backend)


def timed_run(args, pool_fn, batched_fn):
    """Dispatch to pool or batched runner; return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = pool_fn() if args.backend == "numpy" else batched_fn()
    return result, time.perf_counter() - t0
