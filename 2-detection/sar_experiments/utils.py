# Utilities for SAR experiments

import argparse
import ast
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import torch

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.backend import Array, Backend
import matplot2tikz
from sar_experiments.wavelets import apply_wavelet_to_sits


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_splitting(s: str) -> tuple:
    """Parse a splitting string like '(5,5)' into a (rows, cols) tuple.

    Uses ``ast.literal_eval`` so no arbitrary code is executed.

    Parameters
    ----------
    s : str
        String of the form ``'(r,c)'``.

    Returns
    -------
    tuple[int, int]
    """
    try:
        result = ast.literal_eval(s)
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and all(isinstance(x, int) for x in result)
        ):
            return result
    except (ValueError, SyntaxError):
        pass
    raise argparse.ArgumentTypeError(
        f"Invalid splitting '{s}'. Expected '(r,c)', e.g. '(5,5)'."
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all compute_detection scripts.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add arguments to.
    """
    parser.add_argument(
        "data_path", type=str, help="Path to the numpy data file (.npy)."
    )
    parser.add_argument("window_size", type=int, help="Sliding window size.")
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch-cpu", "torch-cuda"],
        help="Computation backend (default: numpy).",
    )
    parser.add_argument(
        "--show-interactive",
        action="store_true",
        help="Show plots interactively with matplotlib.",
    )
    parser.add_argument("--export", action="store_true", help="Save PNG plots.")
    parser.add_argument(
        "--export-tikz", action="store_true", help="Also save TikZ (.tex) files."
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="./exports",
        help="Directory for exported plots (default: ./exports).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Crop data to 100×100 for fast debugging."
    )
    parser.add_argument(
        "--splitting",
        type=str,
        default=None,
        help="Grid splitting '(r,c)'. Default: (1,1) CPU, (5,5) GPU.",
    )
    parser.add_argument(
        "--wavelet",
        action="store_true",
        help="Apply wavelet decomposition before detection.",
    )
    parser.add_argument(
        "--wavelet-R", type=int, default=2, help="Range sub-bands (default 2)."
    )
    parser.add_argument(
        "--wavelet-L", type=int, default=2, help="Azimuth sub-bands (default 2)."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    parser.add_argument(
        "--report-memory",
        action="store_true",
        help="Print peak GPU memory at the end (torch-cuda only).",
    )


@dataclass
class RunConfig:
    """Resolved runtime configuration, built from parsed CLI args."""

    backend: "Backend"
    is_gpu: bool
    splitting: tuple
    export_path: Path
    data_stem: str
    run_ts: str


def setup_run(args) -> RunConfig:
    """Validate *args* and return a :class:`RunConfig`.

    Performs GPU availability check, resolves splitting defaults, and
    creates the export directory when needed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (produced by a parser that used
        :func:`add_common_args`).

    Returns
    -------
    RunConfig
    """
    backend = Backend.from_str(args.backend)
    is_gpu = args.backend == "torch-cuda"

    if is_gpu and not torch.cuda.is_available():
        print("ERROR: torch-cuda backend requested but no GPU available.")
        sys.exit(1)

    splitting_str = getattr(args, "splitting", None)
    if splitting_str is None:
        splitting = (5, 5) if is_gpu else (1, 1)
    else:
        splitting = parse_splitting(splitting_str)

    export_path = Path(args.export_path)
    if args.export or args.export_tikz:
        export_path.mkdir(parents=True, exist_ok=True)

    return RunConfig(
        backend=backend,
        is_gpu=is_gpu,
        splitting=splitting,
        export_path=export_path,
        data_stem=Path(args.data_path).stem,
        run_ts=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


def load_sits(args) -> np.ndarray:
    """Load SITS data, applying optional debug crop and wavelet decomposition.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    np.ndarray, shape (n_times, n_rows, n_cols, n_features)
        Always time-first, feature-last — the format expected by
        ``OnlineImageResourceManager``.
    """
    time_first_path = require_time_first(args.data_path)
    if not args.quiet:
        print("Loading data...")
    sits = np.load(time_first_path, mmap_mode="r")  # (T, rows, cols, features)

    if args.debug:
        sits = np.ascontiguousarray(sits[:, :100, :100, :])

    if args.wavelet:
        if not args.quiet:
            print(
                f"Applying wavelet decomposition "
                f"(R={args.wavelet_R}, L={args.wavelet_L})..."
            )
        # apply_wavelet_to_sits expects (rows, cols, features, times)
        sits_wavelet = apply_wavelet_to_sits(
            np.asarray(sits).transpose(1, 2, 3, 0),
            R=args.wavelet_R,
            L=args.wavelet_L,
        )  # (rows, cols, p*R*L, times)
        sits = np.ascontiguousarray(sits_wavelet.transpose(3, 0, 1, 2))
        if not args.quiet:
            print(f"  Shape after wavelet: {sits.shape}")

    return sits


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


class FigureExporter:
    """Save matplotlib figures to PNG and/or TikZ based on CLI export flags.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``export``, ``export_tikz`` attributes.
    cfg : RunConfig
    """

    def __init__(self, args, cfg: RunConfig) -> None:
        self._do_png = args.export
        self._do_tikz = args.export_tikz
        self._path = cfg.export_path
        self._stem_suffix = f"{cfg.data_stem}_{cfg.run_ts}"

    @property
    def active(self) -> bool:
        """True when at least one export format is enabled."""
        return self._do_png or self._do_tikz

    def save(
        self, fig: plt.Figure, stem: str, elapsed: float, *, close: bool = True
    ) -> None:
        """Append elapsed time to the figure title, then write to disk.

        Parameters
        ----------
        fig : plt.Figure
        stem : str
            Base filename (without extension or timestamp).
        elapsed : float
            Wall-clock seconds to append to the title.
        close : bool
            Close the figure after saving (default True).
        """
        if not self.active:
            if close:
                plt.close(fig)
            return
        full_stem = f"{stem}_{self._stem_suffix}"
        fig.axes[0].set_title(fig.axes[0].get_title() + f" ({elapsed:.2f}s)")
        if self._do_png:
            fig.savefig(self._path / f"{full_stem}.png", dpi=150)
        if self._do_tikz:
            matplot2tikz.save(str(self._path / f"{full_stem}.tex"), figure=fig)
        if close:
            plt.close(fig)


def plot_glrt_map(
    data: np.ndarray,
    title: str,
    *,
    cmap: str = "viridis",
    colorbar_label: str = "GLRT",
) -> plt.Figure:
    """Create a standard GLRT detection-map figure.

    Parameters
    ----------
    data : np.ndarray, shape (rows, cols)
    title : str
    cmap : str
        Matplotlib colormap name (default 'viridis').
    colorbar_label : str

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    return fig


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def require_time_first(data_path: str) -> str:
    """Return the time-first version of a SITS .npy file, or exit with instructions.

    Looks for a ``<stem>_time_first.npy`` file next to *data_path*.
    If found, returns its path as a string.
    If not found, prints a message telling the user how to create it and exits.

    Parameters
    ----------
    data_path : str
        Path to the original .npy file (rows, cols, features, times).

    Returns
    -------
    str
        Path to the time-first .npy file.
    """
    if "_time_first" in data_path:
        return data_path

    p = Path(data_path)
    time_first = p.with_stem(p.stem + "_time_first")
    if time_first.exists():
        print(f"Note: using time-first dataset {time_first} instead of {p.name}")
        return str(time_first)
    print(
        f"ERROR: time-first dataset not found: {time_first}\n"
        f"Please convert your data first by running:\n\n"
        f"  uv run sar_experiments/prepare_data.py {data_path}\n"
    )
    sys.exit(1)


def plot_pauli(sar_data: Array) -> AxesImage:
    """Plot Pauli representation of an SLC SAR image by mapping:
        * Red <- |HH-VV|
        * Green <- |HV|
        * Blue <- |HH+VV|

    Parameters
    ----------
    sar_data : Array
        data of shape (n_rows, n_cols, 3), with third dimension in order HH/HV/VV

    Returns
    -------
    AxesImage
        The plotted image object thanks to matplotlib
    """
    R = np.abs(sar_data[:, :, 0] - sar_data[:, :, 2])
    G = np.abs(sar_data[:, :, 1])
    B = np.abs(sar_data[:, :, 0] + sar_data[:, :, 2])

    return plt.imshow(np.dstack([R, G, B]), aspect="auto")
