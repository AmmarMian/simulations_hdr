# Online DCG (Date-Class Gaussian) change detection on CPU or GPU.
# Processes all dates sequentially maintaining H0 (pooled) and H1 (per-date) estimates.
# Memory-efficient for large images.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
import matplotlib.pyplot as plt
from time import perf_counter

from hdrlib.sar.detection_online import OnlineDCGDetector
from hdrlib.core.backend import get_data_on_device, peak_memory_bytes, reset_peak_memory
from hdrlib.core.hardware_ressources import (
    OnlineImageResourceManager,
    OnlineImageGPURessourceManager,
)
from hdrlib.core.logging_config import setup_logging, log_arguments
from utils import (
    add_common_args,
    setup_run,
    load_sits,
    DetectionMapExporter,
    plot_glrt_map,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Online DCG change detection on CPU or GPU backend."
    )
    add_common_args(parser)
    parser.add_argument(
        "--iter-max",
        type=int,
        default=5,
        help="Maximum iterations for H0 and H1 natural gradient estimators (default 5).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Convergence tolerance for H1 estimator (default 1e-8).",
    )
    args = parser.parse_args()
    setup_logging(quiet=args.quiet, debug=args.log_debug)
    logger = logging.getLogger(__name__)
    log_arguments(args)

    cfg = setup_run(args)
    exporter = DetectionMapExporter(args, cfg)

    logger.info("Loading SITS data...")
    sits_np = load_sits(args)  # (n_times, n_rows, n_cols, n_features)
    n_times, n_rows, n_cols, n_features = sits_np.shape
    logger.info(
        f"Data loaded: shape {sits_np.shape}, "
        f"backend {args.backend}, window_size {args.window_size}, "
        f"splitting {cfg.splitting}"
    )

    detector = OnlineDCGDetector(
        backend_name=str(cfg.backend),
        iter_max=args.iter_max,
        tol=args.tol,
    )

    if cfg.is_gpu:
        resource_manager = OnlineImageGPURessourceManager(
            sits_np,
            window_size=args.window_size,
            stride=1,
            detector=detector,
            backend=cfg.backend,
            splitting=cfg.splitting,
            verbose=0 if args.quiet else 1,
        )
    else:
        resource_manager = OnlineImageResourceManager(
            sits_np,
            window_size=args.window_size,
            stride=1,
            detector=detector,
            backend=cfg.backend,
            splitting=cfg.splitting,
            verbose=0 if args.quiet else 1,
        )

    logger.info("Starting online DCG detection processing...")

    reset_peak_memory(cfg.backend)
    t0 = perf_counter()
    glrt_map = resource_manager.process_all_data()
    elapsed = perf_counter() - t0

    logger.info(f"Detection completed in {elapsed:.2f}s.")

    glrt_map_np = get_data_on_device(glrt_map, "numpy")

    exporter.save(
        glrt_map_np, "dcg_online", elapsed, title="Online DCG GLRT", cmap="jet"
    )
    if args.show_interactive:
        plot_glrt_map(glrt_map_np, "Online DCG GLRT", cmap="jet")

    if cfg.is_gpu and args.report_memory:
        mem = peak_memory_bytes(cfg.backend)
        if mem is not None:
            print(f"PEAK_GPU_MEMORY_BYTES={mem}")

    if args.show_interactive:
        plt.show()

    logger.info("Done.")
