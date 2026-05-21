# Change detection using Kronecker-structured GLRT on CPU or GPU

import sys
from pathlib import Path
import os
import torch
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from detection import ScaleAndShapeKroneckerGLRT
from wavelets import apply_wavelet_to_sits
from utils import require_time_first

import argparse
from datetime import datetime
from src.backend import Backend, get_data_on_device
from src.hardware_ressources import ImageCPURessourceManager, ImageGPURessourceManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Change Detection using Kronecker-structured GLRT (scale and shape)."
    )
    parser.add_argument(
        "data_path", type=str, help="Path to the numpy stored data (.npy file)"
    )
    parser.add_argument("window_size", type=int, help="Sliding window size.")
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch-cpu", "torch-cuda"],
        help="Which backend to use for computations.",
    )
    parser.add_argument(
        "--show_interactive",
        action="store_true",
        help="Whether to show plots with matplotlib.",
    )
    parser.add_argument("--export", action="store_true", help="Save plots of CD maps.")
    parser.add_argument(
        "--export-tikz",
        action="store_true",
        help="Also save plots as TikZ (.tex) files.",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="./exports",
        help="Directory where exported plots are saved (default: ./exports).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Crop data to smaller size to debug."
    )
    parser.add_argument(
        "--splitting",
        type=str,
        default="auto",
        help='Grid splitting for processing, format "(r,c)" or "auto" (default: auto for GPU, (1,1) for CPU).',
    )
    parser.add_argument(
        "--wavelet-R",
        type=int,
        default=2,
        help="Number of range sub-bands for wavelet decomposition (default 2).",
    )
    parser.add_argument(
        "--wavelet-L",
        type=int,
        default=2,
        help="Number of azimuth sub-bands for wavelet decomposition (default 2).",
    )
    parser.add_argument(
        "--iter_max",
        type=int,
        default=10,
        help="Maximum MM iterations for Kronecker estimator (default 10).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for MM iterations (default 1e-4).",
    )
    parser.add_argument(
        "--vram",
        type=float,
        default=None,
        help="Available VRAM in MB (for GPU auto-splitting; default: auto-detect).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output (useful for benchmarking).",
    )
    parser.add_argument(
        "--report-memory",
        action="store_true",
        help="Print peak GPU memory usage at the end (torch-cuda only).",
    )
    args = parser.parse_args()

    # Determine backend and set defaults
    backend = Backend.from_str(args.backend)
    is_gpu = args.backend == "torch-cuda"

    if is_gpu and not torch.cuda.is_available():
        print("ERROR: torch-cuda backend requested but no GPU available")
        sys.exit(1)

    device = torch.device("cuda") if is_gpu else None

    # Determine splitting
    if args.splitting == "auto":
        splitting = (5, 5) if is_gpu else (1, 1)
    else:
        splitting = eval(args.splitting)

    export_path = Path(args.export_path)
    if args.export or args.export_tikz:
        export_path.mkdir(parents=True, exist_ok=True)

    if args.export_tikz:
        import matplot2tikz

    data_stem = Path(args.data_path).stem
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save(fig, stem, elapsed):
        full_stem = f"{stem}_{data_stem}_{run_ts}"
        fig.axes[0].set_title(fig.axes[0].get_title() + f" ({elapsed:.2f}s)")
        if args.export:
            fig.savefig(export_path / f"{full_stem}.png", dpi=150)
        if args.export_tikz:
            matplot2tikz.save(str(export_path / f"{full_stem}.tex"), figure=fig)

    # Load data — shape (n_times, n_rows, n_cols, n_features)
    time_first_path = require_time_first(args.data_path)
    if not args.quiet:
        print("Reading data...")
    sits_np = np.load(time_first_path)  # (n_times, n_rows, n_cols, n_features)
    if args.debug:
        sits_np = sits_np[:, :100, :100, :]

    # Extract n_features before wavelet decomposition
    n_features = sits_np.shape[3]

    # Wavelet decomposition (mandatory for Kronecker structure)
    if not args.quiet:
        print(f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})...")
    # apply_wavelet_to_sits expects (rows, cols, features, times)
    sits_np = apply_wavelet_to_sits(
        sits_np.transpose(1, 2, 3, 0), R=args.wavelet_R, L=args.wavelet_L
    ).transpose(3, 0, 1, 2)
    if not args.quiet:
        print(f"  Shape after wavelet: {sits_np.shape}")

    # Kronecker structure is defined by wavelet decomposition
    # a = n_features (original channels), b = R*L (wavelet sub-bands)
    a = n_features
    b = args.wavelet_R * args.wavelet_L

    # Convert to torch tensor: (n_times, n_features, n_rows, n_cols)
    sits_data = torch.from_numpy(sits_np).moveaxis(3, 1)
    sits_np = None  # free memory

    if not args.quiet:
        print(
            f"Image size: {sits_data.shape[-2]}x{sits_data.shape[-1]}, "
            f"Time steps: {sits_data.shape[0]}, "
            f"Channels after wavelet: {sits_data.shape[1]}, "
            f"Kronecker factors: a={a} (features), b={b} (R×L={args.wavelet_R}×{args.wavelet_L}), p={a * b}"
        )

    # Verify Kronecker structure
    p = sits_data.shape[1]
    if a * b != p:
        raise ValueError(
            f"Kronecker structure a*b={a * b} does not match number of channels p={p}"
        )

    # Create detector
    if not args.quiet:
        print(
            f"\nComputing Kronecker GLRT (a={a}, b={b}, backend={args.backend}, device={device})..."
        )
    kronecker_detector = ScaleAndShapeKroneckerGLRT(
        a=a,
        b=b,
        backend_name=args.backend,
        tol=args.tol,
        iter_max=args.iter_max,
        verbosity=False,
    )

    # Create resource manager and process
    if is_gpu:
        manager = ImageGPURessourceManager(
            sits_data,
            args.window_size,
            1,
            kronecker_detector.compute,
            device=device,
            splitting=splitting,
            vram=args.vram,
            verbose=0 if args.quiet else 1,
        )
    else:
        manager = ImageCPURessourceManager(
            sits_data,
            args.window_size,
            1,
            kronecker_detector.compute,
            backend=args.backend,
            splitting=splitting,
            verbose=0 if args.quiet else 1,
        )

    if is_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    cd_results = manager.process_all_data()
    elapsed = perf_counter() - start
    if not args.quiet:
        print(f"Took {elapsed:.2f} seconds.")

    # Export / display results
    if args.show_interactive or args.export or args.export_tikz:
        fig = plt.figure(dpi=150)
        plt.imshow(
            get_data_on_device(cd_results, "numpy"), aspect="auto", cmap="viridis"
        )
        plt.colorbar(label="Log-likelihood ratio")
        plt.title(f"Kronecker GLRT (a={a}, b={b})")
        if args.export or args.export_tikz:
            _save(fig, f"kronecker_a{a}b{b}_{args.backend}", elapsed)
            if not args.quiet:
                print(f"Exported to {export_path}/")

    if args.report_memory and is_gpu:
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"PEAK_GPU_MEMORY_BYTES={peak_bytes}")

    if args.show_interactive:
        plt.show()

    if not args.quiet:
        print("\nDone.")
