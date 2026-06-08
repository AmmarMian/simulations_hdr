# Online DCG (Date-Class Gaussian) change detection on CPU or GPU
# Processes all dates sequentially maintaining H0 (pooled) and H1 (per-date) estimates.
# Memory-efficient for large images.

import sys
from pathlib import Path
import os
import torch
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import matplot2tikz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backend import Backend
from src.detection_online import OnlineDCGDetector
from src.hardware_ressources import (
    OnlineImageResourceManager,
    OnlineImageGPURessourceManager,
)
from sar_experiments.wavelets import apply_wavelet_to_sits

import argparse
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Online DCG change detection on CPU or GPU backend."
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
    parser.add_argument(
        "--iter-max",
        type=int,
        default=10,
        help="Maximum iterations for H1 natural gradient estimator (default 20).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Convergence tolerance for H1 estimator (default 1e-8).",
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

    # Load data with memory mapping
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if not args.quiet:
        print("Loading data...")
    sits_np = np.load(
        args.data_path, mmap_mode="r"
    )  # (n_rows, n_cols, n_features, n_times)
    if args.debug:
        sits_np = sits_np[:100, :100].copy()
    else:
        sits_np = np.asarray(sits_np)

    # Optional wavelet decomposition
    if args.wavelet:
        if not args.quiet:
            print(
                f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})..."
            )
        sits_np_wavelet = apply_wavelet_to_sits(
            sits_np[..., :2], R=args.wavelet_R, L=args.wavelet_L
        )
        n_features_wavelet = sits_np_wavelet.shape[2]
        if not args.quiet:
            print(f"  Shape after wavelet: {sits_np_wavelet.shape}")
        sits_np_full = np.zeros(
            (sits_np.shape[0], sits_np.shape[1], n_features_wavelet, sits_np.shape[3]),
            dtype=complex,
        )
        for t in range(sits_np.shape[3]):
            sits_np_full[..., t] = apply_wavelet_to_sits(
                sits_np[..., t : t + 1], R=args.wavelet_R, L=args.wavelet_L
            ).squeeze(-1)
        sits_np = sits_np_full

    n_rows, n_cols, n_features, n_times = sits_np.shape
    if not args.quiet:
        print(f"Data shape: ({n_rows}, {n_cols}, {n_features}, {n_times})")
        print(f"Backend: {args.backend}, window_size: {args.window_size}")

    # Reshape data: (n_rows, n_cols, n_features, n_times) -> (n_times, n_features, n_rows, n_cols)
    sits_np = np.transpose(sits_np, (3, 2, 0, 1))  # (times, features, rows, cols)

    # Create resource manager for online processing
    splitting = tuple(map(int, args.splitting.strip("()").split(",")))

    # Initialize DCG detector factory
    def create_detector(backend_name):
        return OnlineDCGDetector(
            backend_name=backend_name,
            h0_lr=1.0,
            iter_max=args.iter_max,
            tol=args.tol,
        )

    t0 = perf_counter()

    # The resource manager will create detectors per split using the correct backend
    if is_gpu:
        detector = create_detector("torch-cuda")
        resource_manager = OnlineImageGPURessourceManager(
            sits_np,
            window_size=args.window_size,
            stride=args.window_size,
            detector=detector,
            splitting=splitting,
            verbose=(0 if args.quiet else 1),
        )
    else:
        detector = create_detector(args.backend)
        resource_manager = OnlineImageResourceManager(
            sits_np,
            window_size=args.window_size,
            stride=args.window_size,
            detector=detector,
            backend=args.backend,
            splitting=splitting,
            verbose=(0 if args.quiet else 1),
        )

    if not args.quiet:
        print(f"Splitting: {splitting}")

    # Process all data
    if not args.quiet:
        print("\nProcessing...")

    glrt_map = resource_manager.process_all_data()

    elapsed = perf_counter() - t0

    if not args.quiet:
        print(f"Total elapsed time: {elapsed:.2f}s")

    # Export results
    if args.export or args.export_tikz:
        # glrt_map shape is (n_rows, n_cols, n_times - 1)
        glrt_map_np = (
            glrt_map.detach().cpu().numpy()
            if isinstance(glrt_map, torch.Tensor)
            else glrt_map
        )
        for t in range(glrt_map_np.shape[2]):
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(glrt_map_np[..., t], cmap="jet")
            ax.set_title(f"Online DCG (Date {t} → {t + 1})")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            plt.colorbar(im, ax=ax, label="GLRT")
            _save(fig, "dcg_online", elapsed)

    if is_gpu and args.report_memory:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
