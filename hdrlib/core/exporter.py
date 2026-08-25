# Experiment result exporter — shared infrastructure for all 2-detection experiments.
#
# Saves three sidecar files per result:
#   {stem}.npy        — raw result array
#   {stem}.json       — provenance (git SHA, paths, args, timing)
#   {stem}_plot.py    — self-contained plot script; supports --tikz for PGFPlots export

import json
import logging
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    """Return HEAD short SHA, appending '*' if the working tree is dirty."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{sha}{'*' if dirty else ''}"
    except Exception:
        return "unknown"


def write_prov_sidecar(stem_path, args) -> None:
    """Write a minimal provenance sidecar ``{stem}.json`` next to a saved figure.

    For scripts too simple to warrant :class:`ResultExporter` (e.g. the
    1-context illustration scripts), call this right after saving each
    ``.tex`` figure so ``register_latex.py`` can recover the run's ``seed``
    and the dissertation's ``gen_figures.py`` can print every CLI parameter.

    Parameters
    ----------
    stem_path : str or Path
        Path to the saved figure, with or without extension
        (e.g. ``outputs/mean.tex`` or ``outputs/mean``).
    args : argparse.Namespace
        Full parsed CLI args — stored verbatim as ``"args"``.
    """
    stem_path = Path(stem_path)
    provenance = {
        "git_sha": _git_sha(),
        "script": str(Path(sys.argv[0]).resolve()),
        "args": vars(args),
    }
    stem_path.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, default=str)
    )


class ResultExporter(ABC):
    """Base class for saving experiment results with full provenance.

    Subclass and implement :meth:`_plot_script` for each result modality.

    Parameters
    ----------
    args : argparse.Namespace
        Full parsed CLI args — stored verbatim in the provenance sidecar.
    export_path : Path
        Directory where all files are written.
    stem_suffix : str
        Appended to every stem to make filenames unique across runs,
        e.g. ``"Scene4_cropped_20260609_085500"``.
    """

    def __init__(self, args, export_path: Path, stem_suffix: str) -> None:
        self._args = args
        self._export_path = export_path
        self._stem_suffix = stem_suffix
        self._provenance_base = {
            "git_sha": _git_sha(),
            "script": str(Path(sys.argv[0]).resolve()),
            "data_path": str(Path(args.data_path).resolve()),
            "args": vars(args),
        }

    @property
    def active(self) -> bool:
        """True when ``--export`` was passed."""
        return self._args.export

    def save(
        self, data: np.ndarray, stem: str, elapsed: float, **plot_kwargs
    ) -> "Path | None":
        """Persist *data* and write provenance sidecar + plot script.

        Parameters
        ----------
        data : np.ndarray
            Result array to persist.
        stem : str
            Base filename component, e.g. ``"gaussian_offline"``.
        elapsed : float
            Wall-clock seconds for the computation.
        **plot_kwargs
            Forwarded to :meth:`_plot_script` (e.g. ``title``, ``cmap``).

        Returns
        -------
        Path or None
            Stem path (no extension) if exported, else ``None``.
        """
        if not self.active:
            return None

        self._export_path.mkdir(parents=True, exist_ok=True)
        full_stem = f"{stem}_{self._stem_suffix}"
        out = self._export_path / full_stem

        np.save(f"{out}.npy", data)

        provenance = {
            **self._provenance_base,
            "elapsed_s": round(elapsed, 3),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }
        Path(f"{out}.json").write_text(json.dumps(provenance, indent=2))
        Path(f"{out}_plot.py").write_text(self._plot_script(full_stem, **plot_kwargs))

        logger.info(f"Exported results to {self._export_path}/")
        logger.info(f"  {out.name}.npy  — result array {list(data.shape)} {data.dtype}")
        logger.info(f"  {out.name}.json — provenance sidecar")
        logger.info(f"  {out.name}_plot.py — standalone plot script")

        return out

    @abstractmethod
    def _plot_script(self, stem: str, **kwargs) -> str:
        """Return source code of a self-contained standalone plot script."""
        ...


# ---------------------------------------------------------------------------
# TikZ export
# ---------------------------------------------------------------------------

_MATH_GROUP = re.compile(r"\\\(\\displaystyle(.*?)\\\)", re.DOTALL)


def _repair_math(code: str) -> str:
    r"""Undo matplot2tikz's escaping of ``_`` and ``^`` inside math groups.

    ``matplot2tikz`` escapes every underscore of a label *before* it splits
    text mode from math mode, so a perfectly ordinary ``$x_1$`` comes out as
    ``\(\displaystyle x\_1\)`` and typesets as a literal underscore. Text mode
    still needs the escape, so the repair is applied inside math groups only.
    """
    def unescape(match: "re.Match[str]") -> str:
        body = match.group(1).replace(r"\_", "_").replace(r"\^", "^")
        return rf"\(\displaystyle{body}\)"

    return _MATH_GROUP.sub(unescape, code)


def save_tikz(
    filepath,
    axis_width: str = r"0.45\textwidth",
    axis_height: str = "4.6cm",
    horizontal_sep: str = "1.4cm",
    vertical_sep: str = "1.5cm",
    legend_font: str = r"\footnotesize",
    extra_axis_parameters=None,
    **kwargs,
) -> None:
    r"""Save the current figure as PGFPlots code, with this repo's defaults.

    Wraps :func:`matplot2tikz.save` with what every figure of the dissertation
    needs and nothing more:

    * a panel size given here rather than patched into the ``.tex`` afterwards,
      so that a re-sync into the dissertation does not undo it;
    * enough separation between the panels of a grid for a title not to land on
      the axis label of the panel above it;
    * a legend without a frame, opaque, and in the same size as the caption,
      since the exported one otherwise keeps matplotlib's proportions and lets
      the curves run through its entries;
    * the math repair of :func:`_repair_math`.

    Parameters
    ----------
    filepath : str or Path
        Destination ``.tex`` file.
    axis_width, axis_height : str
        Size of a single panel, as LaTeX lengths.
    horizontal_sep, vertical_sep : str
        Separation between the panels of a grid.
    legend_font : str
        Font size command used inside the legend.
    extra_axis_parameters : iterable of str, optional
        Appended to the per-axis options.
    **kwargs
        Passed through to :func:`matplot2tikz.get_tikz_code`.
    """
    from matplot2tikz import get_tikz_code

    axis_parameters = [
        # An opaque legend: the exported one is transparent by default and
        # the curves run straight through the entries.
        "legend style={draw=none, fill=white, fill opacity=0.9, "
        f"text opacity=1, font={legend_font}}}",
        r"label style={font=\footnotesize}",
        r"tick label style={font=\scriptsize}",
        r"title style={font=\footnotesize, yshift=-2pt}",
    ]
    axis_parameters += list(extra_axis_parameters or [])

    code = get_tikz_code(
        filepath=filepath,
        axis_width=axis_width,
        axis_height=axis_height,
        extra_axis_parameters=axis_parameters,
        extra_groupstyle_parameters=[
            f"horizontal sep={horizontal_sep}",
            f"vertical sep={vertical_sep}",
        ],
        **kwargs,
    )
    # matplot2tikz emits "minor xticklabels={}" / "minor yticklabels={}" and the
    # matching "scaled minor ? ticks=manual:..." for log axes and for the shared
    # axes of a grid. This pgfplots release does not know those keys, and a
    # single unknown key aborts the externalised build of the WHOLE document --
    # with an error pointing at \end{axis}, far from the cause. They carry no
    # information, so they are dropped here rather than in each experiment.
    _UNSUPPORTED = (
        "minor xticklabels=", "minor yticklabels=",
        "scaled minor x ticks=", "scaled minor y ticks=",
    )
    kept = [ln for ln in _repair_math(code).splitlines()
            if not ln.strip().startswith(_UNSUPPORTED)]
    Path(filepath).write_text("\n".join(kept) + "\n")
