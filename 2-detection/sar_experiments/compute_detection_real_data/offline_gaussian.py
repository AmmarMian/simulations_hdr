# Offline Gaussian GLRT change detection on CPU or GPU.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from sar_experiments.detection import GaussianGLRT
from utils import (
    add_common_args,
    setup_run,
    load_sits,
    DetectionMapExporter,
    plot_glrt_map,
)
from hdrlib.core.backend import get_data_on_device, reset_peak_memory, peak_memory_bytes, oom_errors
from hdrlib.core.hardware_ressources import ImageCPURessourceManager, ImageGPURessourceManager
from hdrlib.core.logging_config import setup_logging, log_arguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Offline Gaussian GLRT change detection on CPU or GPU backend.")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(quiet=args.quiet, debug=args.log_debug)
    logger = logging.getLogger(__name__)
    log_arguments(args)

    cfg = setup_run(args)
    exporter = DetectionMapExporter(args, cfg)

    logger.info("Loading SITS data...")
    sits_np = load_sits(args)  # (n_times, n_rows, n_cols, n_features)
    sits_data = np.ascontiguousarray(sits_np.transpose(0, 3, 1, 2))  # (T, p, H, W) — CPU, splits moved to GPU on demand
    sits_np = None
    logger.info(
        f"Data loaded: image size {sits_data.shape[-2]}×{sits_data.shape[-1]}, "
        f"time steps {sits_data.shape[0]}, splitting {cfg.splitting}"
    )

    logger.info("Starting Gaussian GLRT detection...")

    manager = (ImageGPURessourceManager if cfg.is_gpu else ImageCPURessourceManager)(
        image_data=sits_data,
        window_size=args.window_size,
        stride=1,
        process_one_split=GaussianGLRT(cfg.backend).compute,
        splitting=cfg.splitting,
        backend=cfg.backend,
        verbose=0 if args.quiet else 1,
    )

    reset_peak_memory(cfg.backend)
    t0 = perf_counter()
    results = manager.process_all_data()
    elapsed = perf_counter() - t0

    logger.info(f"Detection completed in {elapsed:.2f}s.")

    results_np = get_data_on_device(results, "numpy")
    exporter.save(results_np, f"gaussian_{args.backend}", elapsed, title="Gaussian GLRT")
    if args.show_interactive:
        plot_glrt_map(results_np, "Gaussian GLRT")

    if args.report_memory and cfg.is_gpu:
        mem = peak_memory_bytes(cfg.backend)
        if mem is not None:
            logger.info(f"Peak GPU memory: {mem / 1e9:.2f} GB (PEAK_GPU_MEMORY_BYTES={mem})")

    if args.show_interactive:
        plt.show()

    logger.info("Done.")
