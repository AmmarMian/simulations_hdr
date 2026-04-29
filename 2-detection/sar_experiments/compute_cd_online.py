# Online change detection computation on CPU or GPU
# Processes all dates sequentially while maintaining state per spatial split.
# Memory-efficient for large images (even when 2+ dates don't fit in RAM).

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

sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from detection import OnlineGaussianGLRT
from wavelets import apply_wavelet_to_sits

import argparse
from datetime import datetime
from src.backend import Backend, get_data_on_device
from src.hardware_ressources import OnlineImageResourceManager, OnlineImageGPURessourceManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Online change detection on CPU or GPU backend.")
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
        help='Grid splitting for processing, format "(r,c)". Defaults: (1,1) for CPU, (5,5) for GPU.',
    )
    parser.add_argument(
        "--wavelet",
        action="store_true",
        help="Apply wavelet decomposition before detection.",
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

    # Set default splitting based on backend
    if args.splitting is None:
        default_splitting = (5, 5) if is_gpu else (1, 1)
        args.splitting = f"({default_splitting[0]},{default_splitting[1]})"

    # Validate GPU availability
    if is_gpu and not torch.cuda.is_available():
        print("ERROR: torch-cuda backend requested but no GPU available")
        sys.exit(1)

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
            fig.savefig(export_path / f"{full_stem}.png")
        if args.export_tikz:
            matplot2tikz.save(str(export_path / f"{full_stem}.tex"), figure=fig)
        plt.close(fig)

    # Load data with memory mapping to avoid loading full dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if not args.quiet:
        print("Loading data...")
    sits_np = np.load(args.data_path, mmap_mode="r")  # (n_rows, n_cols, n_features, n_times)
    if args.debug:
        sits_np = sits_np[:100, :100].copy()  # Make writable copy for debug
    else:
        # For torch compatibility, ensure array is writable (mmap is read-only)
        sits_np = np.asarray(sits_np)

    # Optional wavelet decomposition (applied on first 2 dates to minimize reads)
    if args.wavelet:
        if not args.quiet:
            print(
                f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})..."
            )
        # For online processing, we apply wavelet per-date as it streams through
        # For now, apply to a reference (could be optimized further)
        sits_np_wavelet = apply_wavelet_to_sits(
            sits_np[..., :2], R=args.wavelet_R, L=args.wavelet_L
        )
        n_features_wavelet = sits_np_wavelet.shape[2]
        if not args.quiet:
            print(f"  Shape after wavelet: {sits_np_wavelet.shape}")
        # Create a new array with wavelet features for all dates
        sits_np_full = np.zeros(
            (sits_np.shape[0], sits_np.shape[1], n_features_wavelet, sits_np.shape[3]),
            dtype=sits_np.dtype,
        )
        for t in range(sits_np.shape[3]):
            sits_np_full[..., t] = apply_wavelet_to_sits(
                sits_np[..., t : t + 1], R=args.wavelet_R, L=args.wavelet_L
            )[..., 0]
        sits_np = sits_np_full

    # Convert to torch tensor: (n_times, n_channels, height, width)
    sits_data = torch.from_numpy(sits_np).moveaxis((0, 1, 3), (2, 3, 0))
    sits_np = None  # free mmap reference

    splitting = eval(args.splitting)
    if not args.quiet:
        print(
            f"Image size: {sits_data.shape[-2]}x{sits_data.shape[-1]}, "
            f"Time steps: {sits_data.shape[0]}, Splitting: {splitting[0]}x{splitting[1]}"
        )

    if not args.quiet:
        print("\nComputing Online Gaussian GLRT")
    gaussian_detector = OnlineGaussianGLRT(backend)
    manager_kwargs = {
        "image_data": sits_data,
        "window_size": 7,  # Standard window size
        "stride": 1,
        "detector": gaussian_detector,
        "splitting": splitting,
        "verbose": 0 if args.quiet else 1,
    }

    # Choose resource manager based on backend
    if is_gpu:
        manager = OnlineImageGPURessourceManager(**manager_kwargs)
    else:
        manager_kwargs["backend"] = backend
        manager = OnlineImageResourceManager(**manager_kwargs)

    if is_gpu:
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    gaussian_results = manager.process_all_data()
    elapsed = perf_counter() - start

    if not args.quiet:
        print(f"Took {elapsed:.2f} seconds.")

    if args.export or args.export_tikz:
        fig = plt.figure(dpi=150)
        plt.imshow(get_data_on_device(gaussian_results, "numpy"), aspect="auto")
        plt.colorbar()
        plt.title("Gaussian GLRT (Online)")
        _save(fig, f"gaussian_online_{args.backend}", elapsed)

    if args.report_memory and is_gpu:
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"PEAK_GPU_MEMORY_BYTES={peak_bytes}")

    if not args.quiet:
        print("\nDone.")
