# Offline change detection on CPU or GPU.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from sar_experiments.detection import GaussianGLRT, DeterministicCompoundGaussianGLRT
from sar_experiments.utils import (
    add_common_args,
    setup_run,
    load_sits,
    FigureExporter,
    plot_glrt_map,
)
from src.backend import get_data_on_device
from src.hardware_ressources import ImageCPURessourceManager, ImageGPURessourceManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Offline change detection on CPU or GPU backend.")
    add_common_args(parser)
    parser.add_argument(
        "--iter-max",
        type=int,
        default=10,
        help="Maximum iterations for fixed-point estimator (default 10).",
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
    args = parser.parse_args()
    cfg = setup_run(args)
    exporter = FigureExporter(args, cfg)

    # Offline scripts use (n_times, n_features, n_rows, n_cols) for the batch managers
    sits_np = load_sits(args)  # (n_times, n_rows, n_cols, n_features)
    sits_data = torch.from_numpy(np.asarray(sits_np)).moveaxis(3, 1)  # (T, p, H, W)
    sits_np = None  # free memory

    if not args.quiet:
        print(
            f"Image size: {sits_data.shape[-2]}×{sits_data.shape[-1]}, "
            f"time steps: {sits_data.shape[0]}, splitting: {cfg.splitting}"
        )

    ResourceManager = ImageGPURessourceManager if cfg.is_gpu else ImageCPURessourceManager

    def _run_detector(label, process_fn):
        if not args.quiet:
            print(f"\nComputing {label}...")
        manager_kwargs = dict(
            image_data=sits_data,
            window_size=args.window_size,
            stride=1,
            process_one_split=process_fn,
            splitting=cfg.splitting,
            verbose=0 if args.quiet else 1,
        )
        if not cfg.is_gpu:
            manager_kwargs["backend"] = cfg.backend
        manager = ResourceManager(**manager_kwargs)

        if cfg.is_gpu:
            torch.cuda.reset_peak_memory_stats()
        t0 = perf_counter()
        results = manager.process_all_data()
        elapsed = perf_counter() - t0

        if not args.quiet:
            print(f"Took {elapsed:.2f}s.")
        return results, elapsed

    if "gaussian" in args.detectors:
        results, elapsed = _run_detector("Gaussian GLRT", GaussianGLRT(cfg.backend).compute)
        if exporter.active or args.show_interactive:
            fig = plot_glrt_map(get_data_on_device(results, "numpy"), "Gaussian GLRT")
            exporter.save(fig, f"gaussian_{args.backend}", elapsed, close=not args.show_interactive)

    if "dcg" in args.detectors:
        dcg_detector = DeterministicCompoundGaussianGLRT(
            cfg.backend,
            verbosity=False,
            iteration_chunk_size=args.iteration_chunk,
            iter_max=args.iter_max,
        )
        try:
            results, elapsed = _run_detector("DCG GLRT", dcg_detector.compute)
        except torch.cuda.OutOfMemoryError:
            print(
                "ERROR: CUDA out of memory for DCG GLRT. "
                "Try increasing --splitting (e.g. --splitting '(8,8)') "
                "or decreasing --iteration-chunk."
            )
            sys.exit(1)

        if exporter.active or args.show_interactive:
            fig = plot_glrt_map(get_data_on_device(results, "numpy"), "DCG GLRT")
            exporter.save(fig, f"dcg_{args.backend}", elapsed, close=not args.show_interactive)

    if args.report_memory and cfg.is_gpu:
        print(f"PEAK_GPU_MEMORY_BYTES={torch.cuda.max_memory_allocated()}")

    if args.show_interactive:
        plt.show()

    if not args.quiet:
        print("\nDone.")

