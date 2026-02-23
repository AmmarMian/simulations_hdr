# Compare time of detection on whole image using Gaussian or CG detectors

import sys
from pathlib import Path
import os
import numpy as np
from time import perf_counter
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from detection import GaussianGLRT, DeterministicCompoundGaussianGLRT
from wavelets import apply_wavelet_to_sits

import argparse
from src.backend import get_backend_module, get_data_on_device


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
    parser.add_argument(
        "--export-path",
        type=str,
        default=".",
        help="Directory where exported plots are saved (default: current directory).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Crop data to smaller size to debug."
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
    args = parser.parse_args()

    export_path = Path(args.export_path)
    if args.export:
        export_path.mkdir(parents=True, exist_ok=True)

    be = get_backend_module(args.backend)

    # Load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    print("Reading data...")
    sits_data = np.load(args.data_path)
    if args.debug:
        sits_data = sits_data[:100, :100]

    # Optional wavelet decomposition (applied before sliding window)
    if args.wavelet:
        print(f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})...")
        sits_data = apply_wavelet_to_sits(sits_data, R=args.wavelet_R, L=args.wavelet_L)
        print(f"  Shape after wavelet: {sits_data.shape}")

    n_rows, n_cols, n_features, n_times = sits_data.shape
    print(f"Data shape: {n_rows}x{n_cols}, features={n_features}, times={n_times}")

    temp = sliding_window_view(
        np.moveaxis(sits_data, -1, -2),
        (args.window_size, args.window_size),
        (0, 1),
    )
    sits_data = np.swapaxes(
        np.reshape(
            temp,
            (temp.shape[0], temp.shape[1], n_times, n_features, args.window_size**2),
        ),
        -1,
        -2,
    )
    temp = None

    # Compute GaussianGLRT
    print("\nComputing Gaussian GLRT")
    gaussian_detector = GaussianGLRT(args.backend)
    start = perf_counter()
    gaussian_results = gaussian_detector.compute(sits_data)
    end = perf_counter()
    print(f"Took {end - start:.2f} seconds.")
    if args.export:
        plt.figure(dpi=150)
        plt.imshow(get_data_on_device(gaussian_results, "numpy"), aspect="auto")
        plt.colorbar()
        plt.title("Gaussian GLRT")
        plt.savefig(export_path / f"gaussian_{args.backend}.png")
        plt.close()

    # Compute Deterministic Compound Gaussian GLRT
    print("\nComputing Deterministic Compound Gaussian GLRT")
    dcg_detector = DeterministicCompoundGaussianGLRT(
        args.backend, verbosity=True, iteration_chunk_size=args.iteration_chunk
    )
    start = perf_counter()
    dcg_results = dcg_detector.compute(sits_data)
    end = perf_counter()
    print(f"Took {end - start:.2f} seconds.")
    if args.export:
        plt.figure(dpi=150)
        plt.imshow(get_data_on_device(dcg_results, "numpy"), aspect="auto")
        plt.colorbar()
        plt.title("DCG GLRT")
        plt.savefig(export_path / f"dcg_{args.backend}.png")
        plt.close()
