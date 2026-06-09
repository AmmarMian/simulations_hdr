# Offline change detection on CPU or GPU.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from sar_experiments.detection import GaussianGLRT, DeterministicCompoundGaussianGLRT
from sar_experiments.utils import (
    add_common_args,
    setup_run,
    load_sits,
    DetectionMapExporter,
    plot_glrt_map,
)
from src.backend import get_data_on_device, permute, reset_peak_memory, peak_memory_bytes, oom_errors
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
    exporter = DetectionMapExporter(args, cfg)

    # Offline scripts use (n_times, n_features, n_rows, n_cols) for the batch managers.
    # We load as numpy and move to the target backend via permute + get_data_on_device.
    sits_np = load_sits(args)  # (n_times, n_rows, n_cols, n_features)
    sits_data = permute(cfg.backend, get_data_on_device(np.asarray(sits_np), cfg.backend), (0, 3, 1, 2))  # (T, p, H, W)
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
        if cfg.is_gpu:
            manager_kwargs["backend"] = cfg.backend
        else:
            manager_kwargs["backend"] = cfg.backend
        manager = ResourceManager(**manager_kwargs)

        reset_peak_memory(cfg.backend)
        t0 = perf_counter()
        results = manager.process_all_data()
        elapsed = perf_counter() - t0

        if not args.quiet:
            print(f"Took {elapsed:.2f}s.")
        return results, elapsed

    if "gaussian" in args.detectors:
        results, elapsed = _run_detector("Gaussian GLRT", GaussianGLRT(cfg.backend).compute)
        results_np = get_data_on_device(results, "numpy")
        exporter.save(results_np, f"gaussian_{args.backend}", elapsed, title="Gaussian GLRT")
        if args.show_interactive:
            plot_glrt_map(results_np, "Gaussian GLRT")

    if "dcg" in args.detectors:
        dcg_detector = DeterministicCompoundGaussianGLRT(
            cfg.backend,
            verbosity=False,
            iteration_chunk_size=args.iteration_chunk,
            iter_max=args.iter_max,
        )
        try:
            results, elapsed = _run_detector("DCG GLRT", dcg_detector.compute)
        except oom_errors(cfg.backend):
            print(
                "ERROR: Out of memory for DCG GLRT. "
                "Try increasing --splitting (e.g. --splitting '(8,8)') "
                "or decreasing --iteration-chunk."
            )
            sys.exit(1)

        results_np = get_data_on_device(results, "numpy")
        exporter.save(results_np, f"dcg_{args.backend}", elapsed, title="DCG GLRT")
        if args.show_interactive:
            plot_glrt_map(results_np, "DCG GLRT")

    if args.report_memory and cfg.is_gpu:
        mem = peak_memory_bytes(cfg.backend)
        if mem is not None:
            print(f"PEAK_GPU_MEMORY_BYTES={mem}")

    if args.show_interactive:
        plt.show()

    if not args.quiet:
        print("\nDone.")
