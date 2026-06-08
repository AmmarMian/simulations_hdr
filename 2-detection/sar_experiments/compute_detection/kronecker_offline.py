# Change detection using Kronecker-structured GLRT on CPU or GPU.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from sar_experiments.detection import ScaleAndShapeKroneckerGLRT
from sar_experiments.utils import (
    add_common_args,
    setup_run,
    load_sits,
    require_time_first,
    FigureExporter,
    plot_glrt_map,
)
from src.backend import get_data_on_device
from src.hardware_ressources import ImageCPURessourceManager, ImageGPURessourceManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Change Detection using Kronecker-structured GLRT (scale and shape)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--iter-max",
        type=int,
        default=5,
        help="Maximum MM iterations for Kronecker estimator (default 5).",
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
        help="Available VRAM in MB for GPU auto-splitting (default: auto-detect).",
    )
    args = parser.parse_args()

    # Kronecker detection requires the wavelet decomposition to build Kronecker structure.
    # Warn the user but do not force it — they may have pre-processed data.
    if not args.wavelet and not args.quiet:
        print(
            "Warning: Kronecker GLRT typically requires --wavelet to establish "
            "the Kronecker structure (a = n_features, b = R×L)."
        )

    cfg = setup_run(args)
    exporter = FigureExporter(args, cfg)

    # Read n_features from the raw file before wavelet expands the channel dimension.
    # mmap_mode="r" makes this essentially free — only the header is accessed.
    n_features = np.load(require_time_first(args.data_path), mmap_mode="r").shape[3]

    sits_np = load_sits(args)  # (T, rows, cols, p*R*L or p)
    b = args.wavelet_R * args.wavelet_L if args.wavelet else 1
    a = n_features

    if not args.quiet:
        print(
            f"Image size: {sits_np.shape[1]}×{sits_np.shape[2]}, "
            f"time steps: {sits_np.shape[0]}, "
            f"channels: {sits_np.shape[3]}, "
            f"Kronecker factors: a={a} (features), b={b} (R×L), p={a*b}"
        )

    p = sits_np.shape[3]
    if a * b != p:
        raise ValueError(
            f"Kronecker structure a*b={a*b} does not match channel count p={p}. "
            "Make sure --wavelet-R and --wavelet-L match the data."
        )

    # Convert to (T, p, H, W) for the batch resource managers
    sits_data = torch.from_numpy(np.asarray(sits_np)).moveaxis(3, 1)
    sits_np = None

    if not args.quiet:
        print(
            f"\nComputing Kronecker GLRT "
            f"(a={a}, b={b}, backend={args.backend}, splitting={cfg.splitting})..."
        )

    kronecker_detector = ScaleAndShapeKroneckerGLRT(
        a=a,
        b=b,
        backend_name=args.backend,
        tol=args.tol,
        iter_max=args.iter_max,
        verbosity=False,
    )

    manager_kwargs = dict(
        image_data=sits_data,
        window_size=args.window_size,
        stride=1,
        process_one_split=kronecker_detector.compute,
        splitting=cfg.splitting,
        verbose=0 if args.quiet else 1,
    )
    if cfg.is_gpu:
        torch.cuda.reset_peak_memory_stats()
        manager = ImageGPURessourceManager(
            **manager_kwargs,
            device=torch.device("cuda"),
            vram=args.vram,
        )
    else:
        manager = ImageCPURessourceManager(**manager_kwargs, backend=args.backend)

    t0 = perf_counter()
    cd_results = manager.process_all_data()
    elapsed = perf_counter() - t0

    if not args.quiet:
        print(f"Took {elapsed:.2f}s.")

    if exporter.active or args.show_interactive:
        fig = plot_glrt_map(
            get_data_on_device(cd_results, "numpy"),
            f"Kronecker GLRT (a={a}, b={b})",
            colorbar_label="Log-likelihood ratio",
        )
        exporter.save(
            fig, f"kronecker_a{a}b{b}_{args.backend}", elapsed, close=not args.show_interactive
        )
        if exporter.active and not args.quiet:
            print(f"Exported to {cfg.export_path}/")

    if args.report_memory and cfg.is_gpu:
        print(f"PEAK_GPU_MEMORY_BYTES={torch.cuda.max_memory_allocated()}")

    if args.show_interactive:
        plt.show()

    if not args.quiet:
        print("\nDone.")

