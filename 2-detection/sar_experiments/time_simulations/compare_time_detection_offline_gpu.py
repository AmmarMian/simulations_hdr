# Compare time of detection on whole image using Gaussian or CG detectors on GPU

import sys
from pathlib import Path
import os
import torch
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from detection import GaussianGLRT, DeterministicCompoundGaussianGLRT
from wavelets import apply_wavelet_to_sits

import argparse
from datetime import datetime
from src.backend import Backend, get_data_on_device
from src.hardware_ressources import ImageGPURessourceManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Comparison of Change Detection Time between Gaussian and Compound-Gaussian detectors."
    )
    parser.add_argument(
        "data_path", type=str, help="Path to the numpy stored data (.npy file)"
    )
    parser.add_argument("window_size", type=int, help="Sliding window size.")
    parser.add_argument("--export", action="store_true", help="Save plots of CD maps.")
    parser.add_argument("--export-tikz", action="store_true", help="Also save plots as TikZ (.tex) files.")
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
        default="(5,5)",
        help='Grid splitting for sequential GPU processing, format "(r,c)" (default (5,5)).',
    )
    parser.add_argument(
        "--wavelet", action="store_true",
        help="Apply wavelet decomposition before detection.",
    )
    parser.add_argument(
        "--wavelet-R", type=int, default=2,
        help="Number of range sub-bands for wavelet decomposition (default 2).",
    )
    parser.add_argument(
        "--wavelet-L", type=int, default=2,
        help="Number of azimuth sub-bands for wavelet decomposition (default 2).",
    )
    parser.add_argument(
        "--iteration-chunk", type=int, default=4096,
        help="Chunk size for DCG detector iterations (default 4096).",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=["gaussian", "dcg"],
        default=["gaussian", "dcg"],
        help="Which detectors to run (default: both).",
    )
    args = parser.parse_args()

    export_path = Path(args.export_path)
    if args.export or args.export_tikz:
        export_path.mkdir(parents=True, exist_ok=True)

    if args.export_tikz:
        import matplot2tikz

    data_stem = Path(args.data_path).stem
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    assert torch.cuda.is_available(), "Cannot run if no GPU available"
    backend = Backend.torch_cuda()

    # Load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    print("Reading data...")
    sits_np = np.load(args.data_path)  # (n_rows, n_cols, n_features, n_times)

    # Optional wavelet decomposition (applied on numpy data before torch conversion)
    if args.wavelet:
        print(f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})...")
        sits_np = apply_wavelet_to_sits(sits_np, R=args.wavelet_R, L=args.wavelet_L)
        print(f"  Shape after wavelet: {sits_np.shape}")

    sits_data = torch.from_numpy(sits_np).moveaxis((0, 1, 3), (2, 3, 0))
    sits_np = None  # free memory
    if args.debug:
        sits_data = sits_data[:100, :100]

    splitting = eval(args.splitting)
    print(f"Image size: {sits_data.shape[-2]}x{sits_data.shape[-1]}, "
          f"Time steps: {sits_data.shape[0]}, Splitting: {splitting[0]}x{splitting[1]}")

    def _save(results, name, elapsed):
        if args.export or args.export_tikz:
            base = name.lower().replace(' ', '_') + "_gpu"
            full_stem = f"{base}_{data_stem}_{run_ts}"
            fig = plt.figure(dpi=150)
            plt.imshow(results, aspect="auto")
            plt.colorbar()
            plt.title(f"{name} ({elapsed:.2f}s)")
            if args.export:
                fig.savefig(export_path / f"{full_stem}.png")
            if args.export_tikz:
                matplot2tikz.save(str(export_path / f"{full_stem}.tex"), figure=fig)
            plt.close(fig)

    if "gaussian" in args.detectors:
        print("\nComputing Gaussian GLRT")
        gaussian_detector = GaussianGLRT(backend)
        try:
            gpu_manager = ImageGPURessourceManager(
                sits_data, args.window_size, 1, gaussian_detector.compute,
                splitting=splitting, verbose=1,
            )
            start = perf_counter()
            gaussian_results = gpu_manager.process_all_data()
            elapsed = perf_counter() - start
            print(f"Took {elapsed:.2f} seconds.")
            _save(gaussian_results, "Gaussian GLRT", elapsed)
        except torch.cuda.OutOfMemoryError:
            print("ERROR: CUDA out of memory for Gaussian GLRT. "
                  "Try increasing --splitting (e.g. --splitting '(8,8)').")

    if "dcg" in args.detectors:
        print("\nComputing Deterministic Compound Gaussian GLRT")
        dcg_detector = DeterministicCompoundGaussianGLRT(
            backend,
            verbosity=False,
            iter_max=10,
            iteration_chunk_size=args.iteration_chunk,
        )
        try:
            gpu_manager = ImageGPURessourceManager(
                sits_data, args.window_size, 1, dcg_detector.compute,
                splitting=splitting, verbose=1,
            )
            start = perf_counter()
            dcg_results = gpu_manager.process_all_data()
            elapsed = perf_counter() - start
            print(f"Took {elapsed:.2f} seconds.")
            _save(dcg_results, "DCG GLRT", elapsed)
        except torch.cuda.OutOfMemoryError:
            print("ERROR: CUDA out of memory for DCG GLRT. "
                  "Try increasing --splitting (e.g. --splitting '(8,8)') "
                  "or decreasing --iteration-chunk.")
