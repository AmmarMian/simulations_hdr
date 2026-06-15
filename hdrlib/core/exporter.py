# Experiment result exporter — shared infrastructure for all 2-detection experiments.
#
# Saves three sidecar files per result:
#   {stem}.npy        — raw result array
#   {stem}.json       — provenance (git SHA, paths, args, timing)
#   {stem}_plot.py    — self-contained plot script; supports --tikz for PGFPlots export

import json
import logging
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
