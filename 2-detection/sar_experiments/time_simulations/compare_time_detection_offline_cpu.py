# Compare time of detection on whole image using Gaussian or DCG detectors on CPU

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
from src.hardware_ressources import ImageCPURessourceManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Comparison of Change Detection Time between Gaussian and Compound-Gaussian detectors."
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
        default="(1,1)",
        help='Grid splitting for sequential CPU processing, format "(r,c)" (default (1,1) = no split).',
    )
    parser.add_argument(
        "--wavelet", action="store_true",
        help="Apply wavelet decomposition before detection."
    )
    parser.add_argument(
        "--wavelet-R", type=int, default=2,
        help="Number of range sub-bands for wavelet decomposition (default 2)."
    )
    parser.add_argument(
        "--wavelet-L", type=int, default=2,
        help="Number of azimuth sub-bands for wavelet decomposition (default 2)."
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
    backend = Backend.from_str(args.backend)

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
    print("Reading data...")
    sits_np = np.load(args.data_path)  # (n_rows, n_cols, n_features, n_times)
    if args.debug:
        sits_np = sits_np[:100, :100]

    # Optional wavelet decomposition (applied on numpy data before torch conversion)
    if args.wavelet:
        print(f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})...")
        sits_np = apply_wavelet_to_sits(sits_np, R=args.wavelet_R, L=args.wavelet_L)
        print(f"  Shape after wavelet: {sits_np.shape}")

    # Convert to torch CPU tensor: (n_times, n_channels, height, width)
    sits_data = torch.from_numpy(sits_np).moveaxis((0, 1, 3), (2, 3, 0))
    sits_np = None  # free memory

    splitting = eval(args.splitting)
    print(f"Image size: {sits_data.shape[-2]}x{sits_data.shape[-1]}, "
          f"Time steps: {sits_data.shape[0]}, Splitting: {splitting[0]}x{splitting[1]}")

    if "gaussian" in args.detectors:
        print("\nComputing Gaussian GLRT")
        gaussian_detector = GaussianGLRT(backend)
        cpu_manager = ImageCPURessourceManager(
            sits_data, args.window_size, 1, gaussian_detector.compute,
            backend=backend, splitting=splitting, verbose=1,
        )
        start = perf_counter()
        gaussian_results = cpu_manager.process_all_data()
        elapsed = perf_counter() - start
        print(f"Took {elapsed:.2f} seconds.")
        if args.export or args.export_tikz:
            fig = plt.figure(dpi=150)
            plt.imshow(get_data_on_device(gaussian_results, "numpy"), aspect="auto")
            plt.colorbar()
            plt.title("Gaussian GLRT")
            _save(fig, f"gaussian_{args.backend}", elapsed)

    if "dcg" in args.detectors:
        print("\nComputing Deterministic Compound Gaussian GLRT")
        dcg_detector = DeterministicCompoundGaussianGLRT(
            backend, verbosity=True, iteration_chunk_size=args.iteration_chunk
        )
        cpu_manager = ImageCPURessourceManager(
            sits_data, args.window_size, 1, dcg_detector.compute,
            backend=backend, splitting=splitting, verbose=1,
        )
        start = perf_counter()
        dcg_results = cpu_manager.process_all_data()
        elapsed = perf_counter() - start
        print(f"Took {elapsed:.2f} seconds.")
        if args.export or args.export_tikz:
            fig = plt.figure(dpi=150)
            plt.imshow(get_data_on_device(dcg_results, "numpy"), aspect="auto")
            plt.colorbar()
            plt.title("DCG GLRT")
            _save(fig, f"dcg_{args.backend}", elapsed)
