# Utilities for SAR experiments

import argparse
import ast
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import torch
import logging

# Add project root to path so src module is accessible
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.backend import Array, Backend
from src.exporter import ResultExporter
from sar_experiments.wavelets import apply_wavelet_to_sits

try:
    import matplot2tikz as _matplot2tikz
except ImportError:
    _matplot2tikz = None

logger = logging.getLogger(__name__)

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
        choices=[
            "numpy",
            "torch-cpu",
            "torch-cuda",
            "cupy",
            "cupy-cuda",
            "jax-cpu",
            "jax-cuda",
        ],
        help="Computation backend (default: numpy).",
    )
    parser.add_argument(
        "--show-interactive",
        action="store_true",
        help="Show plots interactively with matplotlib.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Save result (.npy), provenance sidecar (.json), and plot script (_plot.py).",
    )
    parser.add_argument(
        "--export-tikz",
        action="store_true",
        help="Also save a TikZ/PGFPlots figure (.tex) alongside the exported data.",
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
        "--log-debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--report-memory",
        action="store_true",
        help="Print peak GPU memory at the end (torch-cuda only).",
    )
    parser.add_argument(
        "--repeat-times",
        type=int,
        default=1,
        help=(
            "Repeat the time axis N times using a palindrome bounce "
            "(e.g. T=68, repeat=2 → 136 frames: 0..67, 66..1, 0..1, ...). "
            "Materialises the full repeated array in RAM."
        ),
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
    is_gpu = backend.is_gpu

    # Availability checks per backend
    if args.backend in ("torch-cuda",) and not torch.cuda.is_available():
        logger.error("torch-cuda backend requested but no CUDA GPU is available.")
        sys.exit(1)
    if args.backend in ("torch-mps",) and not torch.backends.mps.is_available():
        logger.error("torch-mps backend requested but MPS is not available.")
        sys.exit(1)
    if args.backend in ("cupy", "cupy-cuda"):
        try:
            import cupy

            if not cupy.cuda.is_available():
                raise RuntimeError
        except (ImportError, RuntimeError):
            logger.error(
                "cupy backend requested but CuPy/CUDA is not available. "
                "Install it with: uv sync --extra cupy"
            )
            sys.exit(1)
    if args.backend.startswith("jax"):
        try:
            import jax  # noqa: F401
        except ImportError:
            logger.error(
                f"{args.backend} backend requested but JAX is not installed. "
                "Install it with: uv sync --extra jax  (CPU/CUDA) or "
                "uv sync --extra jax-metal  (Apple Silicon)"
            )
            sys.exit(1)
        if args.backend == "jax-cuda":
            import jax as _jax

            gpu_devices = (
                _jax.devices("gpu")
                if any(d.platform == "gpu" for d in _jax.devices())
                else []
            )
            if not gpu_devices:
                logger.error(
                    "jax-cuda backend requested but no CUDA GPU is available to JAX. "
                    "A CUDA-enabled jaxlib is required. "
                    "Install it with: uv sync --extra jax-cuda"
                )
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


def repeat_times_palindrome(sits: np.ndarray, repeat: int) -> np.ndarray:
    """Extend the time axis by bouncing the sequence back and forth.

    The bounce pattern avoids duplicate frames at turn-around points:
    one cycle is [0, 1, ..., T-1, T-2, ..., 1] (length 2*(T-1)).
    The cycle is tiled and sliced to produce exactly repeat*T frames,
    so the result always materialises as a contiguous in-RAM array.

    Parameters
    ----------
    sits : np.ndarray, shape (T, ...)
    repeat : int
        Multiplier for the time axis (repeat=1 returns sits unchanged).

    Returns
    -------
    np.ndarray, shape (repeat*T, ...)
    """
    if repeat <= 1:
        return sits
    T = sits.shape[0]
    if T < 2:
        return np.concatenate([sits] * repeat, axis=0)
    # One bounce cycle: forward then back (no duplicate endpoints)
    cycle = list(range(T)) + list(range(T - 2, 0, -1))  # length 2*(T-1)
    target = repeat * T
    n_tiles = math.ceil(target / len(cycle))
    indices = (cycle * n_tiles)[:target]
    return np.ascontiguousarray(sits[indices])


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
    logger.info("Loading data...")
    sits = np.load(time_first_path, mmap_mode="r")  # (T, rows, cols, features)

    if args.debug:
        sits = np.ascontiguousarray(sits[:, :100, :100, :])

    if args.wavelet:
        logger.info(
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
        logger.info(f"  Shape after wavelet: {sits.shape}")

    repeat = getattr(args, "repeat_times", 1)
    if repeat > 1:
        T_orig = sits.shape[0]
        sits = repeat_times_palindrome(np.asarray(sits), repeat)
        logger.info(
            f"Time axis repeated x{repeat} (palindrome bounce): "
            f"{T_orig} → {sits.shape[0]} frames"
        )

    return sits


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

# Template for the self-contained _plot.py script generated by DetectionMapExporter.
# Uses str.format() — double braces produce literal braces in the output.
_DETECTION_MAP_PLOT_TEMPLATE = """\
#!/usr/bin/env python
# Auto-generated — edit freely to restyle.
# To regenerate: re-run the detection script with --export
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Regenerate GLRT result figure.")
parser.add_argument("--tikz", action="store_true", help="Export as PGFPlots (.tex) via matplot2tikz.")
parser.add_argument("--no-save", action="store_true", help="Show only, do not save PNG.")
args = parser.parse_args()

here = Path(__file__).parent
stem = {stem_repr}
data = np.load(here / (stem + ".npy"))

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
im = ax.imshow(data, cmap={cmap_repr}, aspect="auto")
ax.set_title({title_repr})
ax.set_xlabel("Column")
ax.set_ylabel("Row")
fig.colorbar(im, ax=ax, label={colorbar_label_repr})
fig.tight_layout()

if not args.no_save:
    out = here / (stem + ".png")
    fig.savefig(out, dpi=150)
    print(f"Saved {{out}}")

if args.tikz:
    import matplot2tikz
    out = here / (stem + ".tex")
    matplot2tikz.save(str(out), figure=fig)
    print(f"Saved {{out}}")

plt.show()
"""


class DetectionMapExporter(ResultExporter):
    """Export GLRT detection maps as .npy with provenance sidecar and plot script.

    Pass ``--export-tikz`` to also render and save a PGFPlots ``.tex`` file
    at export time (requires ``matplot2tikz``).

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``export``, ``export_tikz``, ``data_path``, ``export_path``.
    cfg : RunConfig
    """

    def __init__(self, args, cfg: "RunConfig") -> None:
        super().__init__(
            args,
            export_path=cfg.export_path,
            stem_suffix=f"{cfg.data_stem}_{cfg.run_ts}",
        )

    @property
    def active(self) -> bool:
        return self._args.export or self._args.export_tikz

    def save(
        self,
        data: Array,
        stem: str,
        elapsed: float,
        *,
        title: str = "GLRT",
        cmap: str = "viridis",
        colorbar_label: str = "GLRT",
    ) -> "Path | None":
        out = super().save(
            data, stem, elapsed, title=title, cmap=cmap, colorbar_label=colorbar_label
        )

        if self._args.export_tikz:
            if _matplot2tikz is None:
                raise ImportError(
                    "--export-tikz requires matplot2tikz. Install it with: uv add matplot2tikz"
                )
            self._export_path.mkdir(parents=True, exist_ok=True)
            full_stem = f"{stem}_{self._stem_suffix}"
            fig = plot_glrt_map(data, title, cmap=cmap, colorbar_label=colorbar_label)
            _matplot2tikz.save(str(self._export_path / full_stem) + ".tex", figure=fig)
            plt.close(fig)

        return out

    def _plot_script(
        self,
        stem: str,
        *,
        title: str = "GLRT",
        cmap: str = "viridis",
        colorbar_label: str = "GLRT",
    ) -> str:
        return _DETECTION_MAP_PLOT_TEMPLATE.format(
            stem_repr=repr(stem),
            cmap_repr=repr(cmap),
            title_repr=repr(title),
            colorbar_label_repr=repr(colorbar_label),
        )


def plot_glrt_map(
    data: Array,
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
        logger.info(f"Note: using time-first dataset {time_first} instead of {p.name}")
        return str(time_first)
    logger.error(
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
