# Online change detection computation on CPU or GPU
# Processes all dates sequentially while maintaining state per spatial split.
# Memory-efficient for large images (even when 2+ dates don't fit in RAM).

import sys
from pathlib import Path
import torch
import argparse
from datetime import datetime
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sar_experiments.detection import OnlineGaussianGLRT
from sar_experiments.wavelets import apply_wavelet_to_sits
from sar_experiments.utils import require_time_first

from src.backend import Backend, get_data_on_device
from src.hardware_ressources import (
    OnlineImageResourceManager,
    OnlineImageGPURessourceManager,
)


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
    parser.add_argument(
        "--show_interactive",
        action="store_true",
        help="Wheter to show plot with matplotlib.",
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
        # plt.close(fig)

    # Load data as memmap — shape (n_times, n_rows, n_cols, n_features).
    # Each time step is contiguous on disk so only active slices are paged in.
    time_first_path = require_time_first(args.data_path)
    if not args.quiet:
        print("Loading data...")
    sits_np = np.load(
        time_first_path, mmap_mode="r"
    )  # (n_times, n_rows, n_cols, n_features)
    if args.debug:
        sits_np = np.ascontiguousarray(sits_np[:, :100, :100, :])

    # Optional wavelet decomposition — loads all dates into RAM
    if args.wavelet:
        if not args.quiet:
            print(
                f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})..."
            )
        # apply_wavelet_to_sits expects (rows, cols, features, times)
        sits_for_wavelet = np.asarray(sits_np).transpose(1, 2, 3, 0)
        sits_wavelet = apply_wavelet_to_sits(
            sits_for_wavelet, R=args.wavelet_R, L=args.wavelet_L
        )  # (rows, cols, p*R*L, times)
        if not args.quiet:
            print(f"  Shape after wavelet: {sits_wavelet.shape}")
        sits_np = np.ascontiguousarray(sits_wavelet.transpose(3, 0, 1, 2))

    splitting = eval(args.splitting)
    if not args.quiet:
        print(
            f"Image size: {sits_np.shape[1]}x{sits_np.shape[2]}, "
            f"Time steps: {sits_np.shape[0]}, Splitting: {splitting[0]}x{splitting[1]}"
        )

    if not args.quiet:
        print("\nComputing Online Gaussian GLRT")
    gaussian_detector = OnlineGaussianGLRT(backend)
    manager_kwargs = {
        "image_data": sits_np,
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

    if args.show_interactive or args.export or args.export_tikz:
        fig = plt.figure(dpi=150)
        plt.imshow(get_data_on_device(gaussian_results, "numpy"), aspect="auto")
        plt.colorbar()
        plt.title("Gaussian GLRT (Online)")
        if args.export or args.export_tikz:
            _save(fig, f"gaussian_online_{args.backend}", elapsed)

    if args.report_memory and is_gpu:
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"PEAK_GPU_MEMORY_BYTES={peak_bytes}")

    if args.show_interactive:
        plt.show()

    if not args.quiet:
        print("\nDone.")
