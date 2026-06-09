# Online Gaussian GLRT change detection on CPU or GPU.
# Processes all dates sequentially while maintaining state per spatial split.
# Memory-efficient for large images (even when 2+ dates don't fit in RAM).

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import matplotlib.pyplot as plt
from time import perf_counter

from sar_experiments.detection import OnlineGaussianGLRT
from sar_experiments.utils import (
    add_common_args,
    setup_run,
    load_sits,
    DetectionMapExporter,
    plot_glrt_map,
)
from src.backend import get_data_on_device, reset_peak_memory, peak_memory_bytes
from src.logging_config import setup_logging, log_arguments
from src.hardware_ressources import (
    OnlineImageResourceManager,
    OnlineImageGPURessourceManager,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Online Gaussian GLRT change detection.")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(quiet=args.quiet, debug=args.log_debug)
    logger = logging.getLogger(__name__)
    log_arguments(args)

    cfg = setup_run(args)
    exporter = DetectionMapExporter(args, cfg)

    logger.info("Loading SITS data...")
    sits_np = load_sits(args)  # (n_times, n_rows, n_cols, n_features)
    logger.info(
        f"Data loaded: image size {sits_np.shape[1]}×{sits_np.shape[2]}, "
        f"time steps {sits_np.shape[0]}, splitting {cfg.splitting}"
    )

    logger.info("Starting online Gaussian GLRT detection...")

    detector = OnlineGaussianGLRT(cfg.backend)
    manager_kwargs = dict(
        image_data=sits_np,
        window_size=args.window_size,
        stride=1,
        detector=detector,
        splitting=cfg.splitting,
        verbose=0 if args.quiet else 1,
    )
    reset_peak_memory(cfg.backend)
    if cfg.is_gpu:
        manager = OnlineImageGPURessourceManager(**manager_kwargs, backend=cfg.backend)
    else:
        manager = OnlineImageResourceManager(**manager_kwargs, backend=cfg.backend)

    t0 = perf_counter()
    results = manager.process_all_data()
    elapsed = perf_counter() - t0

    logger.info(f"Detection completed in {elapsed:.2f}s.")

    results_np = get_data_on_device(results, "numpy")
    exporter.save(results_np, f"gaussian_online_{args.backend}", elapsed, title="Online Gaussian GLRT")
    if args.show_interactive:
        plot_glrt_map(results_np, "Online Gaussian GLRT")

    if args.report_memory and cfg.is_gpu:
        mem = peak_memory_bytes(cfg.backend)
        if mem is not None:
            logger.info(f"Peak GPU memory: {mem / 1e9:.2f} GB (PEAK_GPU_MEMORY_BYTES={mem})")

    if args.show_interactive:
        plt.show()

    logger.info("Done.")
