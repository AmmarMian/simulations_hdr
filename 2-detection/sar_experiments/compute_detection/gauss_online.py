# Online Gaussian GLRT change detection on CPU or GPU.
# Processes all dates sequentially while maintaining state per spatial split.
# Memory-efficient for large images (even when 2+ dates don't fit in RAM).

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
import matplotlib.pyplot as plt
from time import perf_counter

from sar_experiments.detection import OnlineGaussianGLRT
from sar_experiments.utils import (
    add_common_args,
    setup_run,
    load_sits,
    FigureExporter,
    plot_glrt_map,
)
from src.backend import get_data_on_device
from src.hardware_ressources import (
    OnlineImageResourceManager,
    OnlineImageGPURessourceManager,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Online Gaussian GLRT change detection.")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = setup_run(args)
    exporter = FigureExporter(args, cfg)

    sits_np = load_sits(args)  # (n_times, n_rows, n_cols, n_features)
    if not args.quiet:
        print(
            f"Image size: {sits_np.shape[1]}×{sits_np.shape[2]}, "
            f"time steps: {sits_np.shape[0]}, splitting: {cfg.splitting}"
        )

    if not args.quiet:
        print("\nComputing Online Gaussian GLRT...")

    detector = OnlineGaussianGLRT(cfg.backend)
    manager_kwargs = dict(
        image_data=sits_np,
        window_size=args.window_size,
        stride=1,
        detector=detector,
        splitting=cfg.splitting,
        verbose=0 if args.quiet else 1,
    )
    if cfg.is_gpu:
        torch.cuda.reset_peak_memory_stats()
        manager = OnlineImageGPURessourceManager(**manager_kwargs)
    else:
        manager = OnlineImageResourceManager(**manager_kwargs, backend=cfg.backend)

    t0 = perf_counter()
    results = manager.process_all_data()
    elapsed = perf_counter() - t0

    if not args.quiet:
        print(f"Took {elapsed:.2f}s.")

    if exporter.active or args.show_interactive:
        fig = plot_glrt_map(
            get_data_on_device(results, "numpy"),
            "Online Gaussian GLRT",
        )
        exporter.save(fig, f"gaussian_online_{args.backend}", elapsed, close=not args.show_interactive)

    if args.report_memory and cfg.is_gpu:
        print(f"PEAK_GPU_MEMORY_BYTES={torch.cuda.max_memory_allocated()}")

    if args.show_interactive:
        plt.show()

    if not args.quiet:
        print("\nDone.")

