# Online Kronecker structured scaled Gaussian change detection on CPU or GPU.
# Processes all dates sequentially maintaining H0 (pooled) and H1 (per-date) estimates.
# Memory-efficient for large images.
#
# Kronecker structure is derived from the data:
#   a = n_features  (number of SAR channels before wavelet)
#   b = R * L       (wavelet sub-bands; 1 if --wavelet not used)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from hdrlib.sar.detection_online import OnlineKroneckerDetector
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
    require_time_first,
    DetectionMapExporter,
    plot_glrt_map,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Online Kronecker structured scaled Gaussian change detection on CPU or GPU backend."
    )
    add_common_args(parser)
    parser.add_argument(
        "--iter-max",
        type=int,
        default=5,
        help="Maximum MM iterations for H0 warm-start and H1 per-date estimates (default 5).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for MM algorithms (default 1e-4).",
    )
    args = parser.parse_args()
    setup_logging(quiet=args.quiet, debug=args.log_debug)
    logger = logging.getLogger(__name__)
    log_arguments(args)

    if not args.wavelet:
        logger.warning(
            "Kronecker GLRT typically requires --wavelet to establish "
            "the Kronecker structure (a = n_features, b = R×L)."
        )

    cfg = setup_run(args)
    exporter = DetectionMapExporter(args, cfg)

    logger.info("Loading SITS data...")
    n_features = np.load(require_time_first(args.data_path), mmap_mode="r").shape[3]

    sits_np = load_sits(args)  # (n_times, n_rows, n_cols, p)
    n_times, n_rows, n_cols, p = sits_np.shape

    a = n_features
    b = args.wavelet_R * args.wavelet_L if args.wavelet else 1

    if a * b != p:
        raise ValueError(
            f"Kronecker structure a*b={a*b} does not match channel count p={p}. "
            "Make sure --wavelet-R and --wavelet-L match the data."
        )

    logger.info(
        f"Data loaded: image size {n_rows}×{n_cols}, time steps {n_times}, "
        f"channels {p}, Kronecker factors: a={a} (features), b={b} (R×L)"
    )

    detector = OnlineKroneckerDetector(
        a=a,
        b=b,
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

    logger.info(
        f"Starting online Kronecker detection "
        f"(a={a}, b={b}, backend={args.backend}, splitting={cfg.splitting})..."
    )

    reset_peak_memory(cfg.backend)
    t0 = perf_counter()
    glrt_map = resource_manager.process_all_data()
    elapsed = perf_counter() - t0

    logger.info(f"Detection completed in {elapsed:.2f}s.")

    glrt_map_np = get_data_on_device(glrt_map, "numpy")
    title = f"Online Kronecker GLRT (a={a}, b={b})"
    exporter.save(
        glrt_map_np, f"kronecker_online_a{a}b{b}", elapsed,
        title=title, cmap="jet"
    )
    if args.show_interactive:
        plot_glrt_map(glrt_map_np, title, cmap="jet")

    if cfg.is_gpu and args.report_memory:
        mem = peak_memory_bytes(cfg.backend)
        if mem is not None:
            print(f"PEAK_GPU_MEMORY_BYTES={mem}")

    if args.show_interactive:
        plt.show()

    logger.info("Done.")
