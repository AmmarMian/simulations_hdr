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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "auto"],
        help="Device to use: cpu, gpu, or auto-detect (default: cpu).",
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
        "--iter-max",
        type=int,
        default=5,
        help="Maximum MM iterations for Kronecker estimator (default 30).",
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
    args = parser.parse_args()

    # Wavelet decomposition is mandatory for Kronecker structure
    print("Note: Wavelet decomposition is always applied for Kronecker structure.")

    # Determine device and backend
    if args.device == "auto":
        use_gpu = torch.cuda.is_available()
    else:
        use_gpu = args.device == "gpu"

    if use_gpu:
        backend_name = "torch-cuda"
        device = torch.device("cuda")
    else:
        backend_name = "torch-cpu"
        device = None

    backend = Backend.from_str(backend_name)

    # Determine splitting
    if args.splitting == "auto":
        splitting = (5, 5) if use_gpu else (1, 1)
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
        plt.close(fig)

    # Load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    print("Reading data...")
    sits_np = np.load(args.data_path)  # (n_rows, n_cols, n_features, n_times)
    if args.debug:
        sits_np = sits_np[:100, :100]

    # Extract n_features before wavelet decomposition
    n_features = sits_np.shape[2]

    # Wavelet decomposition (mandatory for Kronecker structure)
    print(f"Applying wavelet decomposition (R={args.wavelet_R}, L={args.wavelet_L})...")
    sits_np = apply_wavelet_to_sits(sits_np, R=args.wavelet_R, L=args.wavelet_L)
    print(f"  Shape after wavelet: {sits_np.shape}")

    # Kronecker structure is defined by wavelet decomposition
    # a = n_features (original channels), b = R*L (wavelet sub-bands)
    a = n_features
    b = args.wavelet_R * args.wavelet_L

    # Convert to torch tensor: (n_times, n_channels, height, width)
    sits_data = torch.from_numpy(sits_np).moveaxis((0, 1, 3), (2, 3, 0))
    sits_np = None  # free memory

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
    print(
        f"\nComputing Kronecker GLRT (a={a}, b={b}, backend={backend_name}, device={device})..."
    )
    kronecker_detector = ScaleAndShapeKroneckerGLRT(
        a=a,
        b=b,
        backend_name=backend_name,
        tol=args.tol,
        iter_max=args.iter_max,
        verbosity=True,
    )

    # Create resource manager and process
    if use_gpu:
        manager = ImageGPURessourceManager(
            sits_data,
            args.window_size,
            1,
            kronecker_detector.compute,
            device=device,
            splitting=splitting,
            vram=args.vram,
            verbose=1,
        )
    else:
        manager = ImageCPURessourceManager(
            sits_data,
            args.window_size,
            1,
            kronecker_detector.compute,
            backend=backend_name,
            splitting=splitting,
            verbose=1,
        )

    start = perf_counter()
    cd_results = manager.process_all_data()
    elapsed = perf_counter() - start
    print(f"Took {elapsed:.2f} seconds.")

    # Export results
    if args.export or args.export_tikz:
        fig = plt.figure(dpi=150)
        plt.imshow(
            get_data_on_device(cd_results, "numpy"), aspect="auto", cmap="viridis"
        )
        plt.colorbar(label="Log-likelihood ratio")
        plt.title(f"Kronecker GLRT (a={a}, b={b})")
        _save(fig, f"kronecker_a{a}b{b}_{backend_name}", elapsed)
        print(f"Exported to {export_path}/")

    print("\nDone.")
