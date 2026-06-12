#!/usr/bin/env python
"""MC convergence test: OnlineGaussianGLRT telescopes to GaussianGLRT as T grows.

Backend selection:
  numpy     → multiprocessing.Pool, one trial per worker (avoids MKL contention)
  all other → all trials stacked in leading batch dim, single-pass on device
"""

from __future__ import annotations

import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

_HERE = Path(__file__).parent
_ROOT = str(_HERE.parent.parent)
for _p in (_ROOT, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.backend import get_data_on_device, to_numpy
from src.simulation import T_vec_logspace, generate_gaussian_data, make_sigma_true
from sar_experiments.detection_offline import GaussianGLRT
from sar_experiments.detection_online import OnlineGaussianGLRT
from utils import (
    MCResultExporter,
    add_mc_args,
    finish_h0,
    init_logging,
    make_mc_parser,
    online_single_pass,
    timed_run,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pool worker (numpy only — one trial per worker process)
# ---------------------------------------------------------------------------

def _worker(args):
    trial_data, T_vec = args  # trial_data: (T_max, n_samples, n_features) numpy
    offline_det = GaussianGLRT("numpy")
    online_det = OnlineGaussianGLRT("numpy")

    offline = {T: float(to_numpy(offline_det.compute(trial_data[:T]))) for T in T_vec}
    online_raw = online_single_pass(trial_data, T_vec, online_det)
    online = {T: float(to_numpy(v)) for T, v in online_raw.items()}
    return online, offline


def _run_pool(data_numpy: np.ndarray, T_vec: list[int], n_workers) -> tuple:
    n_trials = data_numpy.shape[0]
    worker_args = [(data_numpy[i], T_vec) for i in range(n_trials)]

    all_online: list[dict] = []
    all_offline: list[dict] = []

    logger.info(f"Starting H0 computation: {n_trials} trials via Pool, {len(T_vec)} T-checkpoints...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} trials"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]MC trials (Pool)...", total=n_trials)
        with Pool(processes=n_workers) as pool:
            for on, off in pool.imap_unordered(_worker, worker_args):
                all_online.append(on)
                all_offline.append(off)
                progress.advance(task)

    online_dict = {T: np.array([d[T] for d in all_online]) for T in T_vec}
    offline_dict = {T: np.array([d[T] for d in all_offline]) for T in T_vec}
    return online_dict, offline_dict


# ---------------------------------------------------------------------------
# Batched path (non-numpy backends)
# ---------------------------------------------------------------------------

def _run_batched(data_numpy: np.ndarray, T_vec: list[int], backend: str) -> tuple:
    """Stack all trials in leading batch dim and run on device."""
    data_device = get_data_on_device(data_numpy, backend)
    T_max = data_numpy.shape[1]
    T_set = set(T_vec)

    offline_det = GaussianGLRT(backend)
    online_det = OnlineGaussianGLRT(backend)

    online_raw: dict = {}
    offline_raw: dict = {}

    logger.info(f"Starting H0 batched computation on {backend} (T_max={T_max}, {len(T_vec)} T-pts)...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task_on = progress.add_task(f"[cyan]Online single-pass (T_max={T_max})...", total=T_max - 2)
        task_off = progress.add_task(f"[green]Offline ({len(T_vec)} points)...", total=len(T_vec))

        online_det.reset_state()
        stat = online_det.initialize(data_device[..., :2, :, :])
        if 2 in T_set:
            online_raw[2] = to_numpy(stat)

        for t in range(2, T_max):
            T_current = t + 1
            stat = online_det.update(stat, data_device[..., t, :, :])
            if T_current in T_set:
                online_raw[T_current] = to_numpy(stat)
            progress.advance(task_on)

        for T in T_vec:
            offline_raw[T] = to_numpy(offline_det.compute(data_device[..., :T, :, :]))
            progress.advance(task_off)

    return online_raw, offline_raw


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    add_mc_args(parser)
    args = parser.parse_args()

    init_logging(args.backend)

    p = args.n_features
    n_samples = 2 * p + 1
    T_vec = T_vec_logspace(args.T_min, args.T_max, args.n_T)
    T_max = max(T_vec)

    logger.info(f"Gaussian MC: p={p}, n_samples={n_samples}, n_trials={args.n_trials}, "
                f"T=[{T_vec[0]}..{T_vec[-1]}] ({len(T_vec)} pts), backend={args.backend}")

    Sigma_true = make_sigma_true(p, seed=args.sigma_seed, normalize="det")
    logger.info(f"Generating data ({args.n_trials}, {T_max}, {n_samples}, {p}) complex128...")
    data = generate_gaussian_data(args.n_trials, T_max, n_samples, p, Sigma_true, seed=args.seed)

    exporter = MCResultExporter(args, Path(args.export_path), f"p{p}_T{T_max}_n{args.n_trials}")

    (online_dict, offline_dict), elapsed = timed_run(
        args,
        lambda: _run_pool(data, T_vec, args.n_workers),
        lambda: _run_batched(data, T_vec, args.backend),
    )

    title = f"Gaussian GLRT convergence  (p={p}, n={n_samples})"
    finish_h0(args, exporter, online_dict, offline_dict, T_vec, "mc_gaussian", title, elapsed)


if __name__ == "__main__":
    main()
