# Offline change detection computation on CPU or GPU

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

from detection import GaussianGLRT, DeterministicCompoundGaussianGLRT
from wavelets import apply_wavelet_to_sits

import argparse
from datetime import datetime
from src.backend import Backend, get_data_on_device
from src.hardware_ressources import ImageCPURessourceManager, ImageGPURessourceManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Offline change detection on CPU or GPU backend.")
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
        "--iter_max",
        type=int,
        default=10,
        help="Maximum iterations for fixed point.",
    )
    parser.add_argument(
        "--iteration-chunk",
        type=int,
        default=4096,
        help="Chunk size for DCG detector iterations (default 4096).",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=["gaussian", "dcg"],
        default=["gaussian", "dcg"],
        help="Which detectors to run (default: both).",
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

    # Load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if not args.quiet:
        print("Reading data...")
    sits_np = np.load(args.data_path)  # (n_rows, n_cols, n_features, n_times)
    if args.debug:
        sits_np = sits_np[:100, :100]

    # Optional wavelet decomposition (applied on numpy data before torch conversion)
    if args.wavelet:
        if not args.quiet:
            print(
                f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})..."
            )
        sits_np = apply_wavelet_to_sits(sits_np, R=args.wavelet_R, L=args.wavelet_L)
        if not args.quiet:
            print(f"  Shape after wavelet: {sits_np.shape}")

    # Convert to torch tensor: (n_times, n_channels, height, width)
    sits_data = torch.from_numpy(sits_np).moveaxis((0, 1, 3), (2, 3, 0))
    sits_np = None  # free memory

    splitting = eval(args.splitting)
    if not args.quiet:
        print(
            f"Image size: {sits_data.shape[-2]}x{sits_data.shape[-1]}, "
            f"Time steps: {sits_data.shape[0]}, Splitting: {splitting[0]}x{splitting[1]}"
        )

    # Choose resource manager based on backend
    ResourceManager = ImageGPURessourceManager if is_gpu else ImageCPURessourceManager

    if "gaussian" in args.detectors:
        if not args.quiet:
            print("\nComputing Gaussian GLRT")
        gaussian_detector = GaussianGLRT(backend)
        manager_kwargs = {
            "image_data": sits_data,
            "window_size": args.window_size,
            "stride": 1,
            "process_one_split": gaussian_detector.compute,
            "splitting": splitting,
            "verbose": 0 if args.quiet else 1,
        }
        if not is_gpu:
            manager_kwargs["backend"] = backend
        manager = ResourceManager(**manager_kwargs)

        start = perf_counter()
        gaussian_results = manager.process_all_data()
        elapsed = perf_counter() - start
        if not args.quiet:
            print(f"Took {elapsed:.2f} seconds.")
        if args.export or args.export_tikz:
            fig = plt.figure(dpi=150)
            plt.imshow(get_data_on_device(gaussian_results, "numpy"), aspect="auto")
            plt.colorbar()
            plt.title("Gaussian GLRT")
            _save(fig, f"gaussian_{args.backend}", elapsed)

    if "dcg" in args.detectors:
        if not args.quiet:
            print("\nComputing Deterministic Compound Gaussian GLRT")
        dcg_detector = DeterministicCompoundGaussianGLRT(
            backend,
            verbosity=False,
            iteration_chunk_size=args.iteration_chunk,
            iter_max=args.iter_max,
        )
        manager_kwargs = {
            "image_data": sits_data,
            "window_size": args.window_size,
            "stride": 1,
            "process_one_split": dcg_detector.compute,
            "splitting": splitting,
            "verbose": 0 if args.quiet else 1,
        }
        if not is_gpu:
            manager_kwargs["backend"] = backend
        manager = ResourceManager(**manager_kwargs)

        try:
            start = perf_counter()
            dcg_results = manager.process_all_data()
            elapsed = perf_counter() - start
            if not args.quiet:
                print(f"Took {elapsed:.2f} seconds.")
            if args.export or args.export_tikz:
                fig = plt.figure(dpi=150)
                plt.imshow(get_data_on_device(dcg_results, "numpy"), aspect="auto")
                plt.colorbar()
                plt.title("DCG GLRT")
                _save(fig, f"dcg_{args.backend}", elapsed)
        except torch.cuda.OutOfMemoryError:
            print(
                "ERROR: CUDA out of memory for DCG GLRT. "
                "Try increasing --splitting (e.g. --splitting '(8,8)') "
                "or decreasing --iteration-chunk."
            )
            sys.exit(1)

    if args.report_memory and is_gpu:
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"PEAK_GPU_MEMORY_BYTES={peak_bytes}")

    if not args.quiet:
        print("\nDone.")
