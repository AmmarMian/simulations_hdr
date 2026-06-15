#!/usr/bin/env python
"""MC power simulation: detection performance of OnlineGaussianGLRT vs GaussianGLRT under H1.

For each T in T_vec, H1 data is generated fresh with the change at
  n_change_dates = max(2, int(T * change_fraction))   (default: T // 2)
so that pre- and post-change segments always have ~T/2 dates each.
This ensures the H1 statistic grows with T and power increases monotonically.

H0 statistics are collected in a single pass over T_max dates (no change).
Thresholds are set at the (1-PFA) percentile of H0 stats for each T.
"""

from __future__ import annotations

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn


from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.simulation import T_vec_logspace, generate_gaussian_data, make_sigma_true
from hdrlib.sar.simulation import generate_gaussian_data_h1
from hdrlib.sar.detectors import GaussianGLRT
from hdrlib.sar.detectors import OnlineGaussianGLRT
from hdrlib.core.mc import MCResultExporter, init_logging, make_mc_parser, timed_run
from hdrlib.sar.mc import (
    _MC_PLOT_TEMPLATE_H1,
    add_mc_args,
    add_mc_h1_args,
    finish_h1,
    online_run,
    online_single_pass,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pool workers
# ---------------------------------------------------------------------------

def _worker_h0(args):
    """H0 worker: one trial, all T checkpoints."""
    trial_data, T_vec = args
    offline = GaussianGLRT("numpy")
    online = OnlineGaussianGLRT("numpy")

    off = {T: float(to_numpy(offline.compute(trial_data[:T]))) for T in T_vec}
    on_raw = online_single_pass(trial_data, T_vec, online)
    on = {T: float(to_numpy(v)) for T, v in on_raw.items()}
    return on, off


def _worker_h1_at_T(trial_data):
    """H1 worker: one trial at a single T (trial_data has exactly T dates)."""
    offline = GaussianGLRT("numpy")
    online = OnlineGaussianGLRT("numpy")
    off = float(to_numpy(offline.compute(trial_data)))
    on = float(to_numpy(online_run(trial_data, online)))
    return on, off


# ---------------------------------------------------------------------------
# Pool runners
# ---------------------------------------------------------------------------

def _run_pool(data_h0, T_vec, n_workers, change_fraction, n_samples, n_features,
              Sigma_1, Sigma_2, seed):
    n_trials = data_h0.shape[0]

    # --- H0: single pass over all T checkpoints ---
    h0_on_all, h0_off_all = [], []
    logger.info(f"Starting H0 computation: {n_trials} trials via Pool, {len(T_vec)} T-checkpoints...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} trials"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]H0 trials...", total=n_trials)
        with Pool(processes=n_workers) as pool:
            for on, off in pool.imap_unordered(_worker_h0,
                                               [(data_h0[i], T_vec) for i in range(n_trials)]):
                h0_on_all.append(on)
                h0_off_all.append(off)
                progress.advance(task)

    h0_stats = {
        "online":  {T: np.array([d[T] for d in h0_on_all])  for T in T_vec},
        "offline": {T: np.array([d[T] for d in h0_off_all]) for T in T_vec},
    }

    # --- H1: fresh data per T, change at T * change_fraction ---
    h1_on_all = {T: [] for T in T_vec}
    h1_off_all = {T: [] for T in T_vec}

    logger.info(f"Starting H1 T-loop: {len(T_vec)} T-values, {n_trials} trials each...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} T-values"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[green]H1 T-loop...", total=len(T_vec))
        with Pool(processes=n_workers) as pool:
            for T in T_vec:
                n_change = max(2, int(T * change_fraction))
                data_h1_T = generate_gaussian_data_h1(
                    n_trials, T, n_samples, n_features, Sigma_1, Sigma_2,
                    seed=seed, n_change_dates=n_change,
                )
                for on, off in pool.imap_unordered(_worker_h1_at_T,
                                                   [data_h1_T[i] for i in range(n_trials)]):
                    h1_on_all[T].append(on)
                    h1_off_all[T].append(off)
                progress.advance(task)

    h1_stats = {
        "online":  {T: np.array(h1_on_all[T])  for T in T_vec},
        "offline": {T: np.array(h1_off_all[T]) for T in T_vec},
    }
    return h0_stats, h1_stats


# ---------------------------------------------------------------------------
# Batched path (non-numpy backends)
# ---------------------------------------------------------------------------

def _run_batched(data_h0, T_vec, backend, change_fraction, n_samples, n_features,
                 Sigma_1, Sigma_2, seed):
    dh0 = get_data_on_device(data_h0, backend)
    T_max = data_h0.shape[1]
    T_set = set(T_vec)
    n_trials = data_h0.shape[0]

    offline = GaussianGLRT(backend)
    online = OnlineGaussianGLRT(backend)

    # --- H0 ---
    h0_on_raw, h0_off_raw = {}, {}
    logger.info(f"Starting H0 batched computation on {backend} (T_max={T_max}, {len(T_vec)} T-pts)...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task_on = progress.add_task(f"[cyan]H0 online (T_max={T_max})...", total=T_max - 2)
        task_off = progress.add_task(f"[cyan]H0 offline ({len(T_vec)} pts)...", total=len(T_vec))

        online.reset_state()
        stat = online.initialize(dh0[..., :2, :, :])
        if 2 in T_set:
            h0_on_raw[2] = to_numpy(stat)
        for t in range(2, T_max):
            stat = online.update(stat, dh0[..., t, :, :])
            if t + 1 in T_set:
                h0_on_raw[t + 1] = to_numpy(stat)
            progress.advance(task_on)

        for T in T_vec:
            h0_off_raw[T] = to_numpy(offline.compute(dh0[..., :T, :, :]))
            progress.advance(task_off)

    h0_stats = {"online": h0_on_raw, "offline": h0_off_raw}

    # --- H1: per T ---
    h1_on_raw, h1_off_raw = {}, {}
    logger.info(f"Starting H1 batched T-loop on {backend} ({len(T_vec)} T-values)...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} T-values"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[green]H1 T-loop...", total=len(T_vec))
        for T in T_vec:
            n_change = max(2, int(T * change_fraction))
            data_h1_T = generate_gaussian_data_h1(
                n_trials, T, n_samples, n_features, Sigma_1, Sigma_2,
                seed=seed, n_change_dates=n_change,
            )
            dh1 = get_data_on_device(data_h1_T, backend)
            h1_off_raw[T] = to_numpy(offline.compute(dh1))
            h1_on_raw[T] = to_numpy(online_run(dh1, online))
            progress.advance(task)

    h1_stats = {"online": h1_on_raw, "offline": h1_off_raw}
    return h0_stats, h1_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    add_mc_args(parser)
    add_mc_h1_args(parser)
    args = parser.parse_args()

    init_logging(args.backend)

    min_trials = int(10 / args.pfa)
    if args.n_trials < min_trials:
        logger.warning(
            f"n_trials={args.n_trials} < 10/PFA={min_trials}. "
            f"Threshold estimate at PFA={args.pfa} will be unreliable. "
            f"Consider --n-trials {min_trials}."
        )

    p = args.n_features
    n_samples = 2 * p + 1
    T_vec = T_vec_logspace(args.T_min, args.T_max, args.n_T)
    T_max = max(T_vec)

    logger.info(f"Gaussian H1 MC: p={p}, n_samples={n_samples}, n_trials={args.n_trials}, "
                f"T=[{T_vec[0]}..{T_vec[-1]}] ({len(T_vec)} pts), backend={args.backend}")
    logger.info(f"  Change fraction={args.change_fraction} | PFA={args.pfa}")

    Sigma_1 = make_sigma_true(p, seed=args.sigma_seed,  normalize="det")
    Sigma_2 = make_sigma_true(p, seed=args.sigma2_seed, normalize="det")

    logger.info("Generating H0 data...")
    data_h0 = generate_gaussian_data(args.n_trials, T_max, n_samples, p, Sigma_1, seed=args.seed)

    exporter = MCResultExporter(
        args, Path(args.export_path), f"p{p}_T{T_max}_n{args.n_trials}",
        plot_template=_MC_PLOT_TEMPLATE_H1,
    )

    (h0_stats, h1_stats), elapsed = timed_run(
        args,
        lambda: _run_pool(data_h0, T_vec, args.n_workers, args.change_fraction, n_samples, p, Sigma_1, Sigma_2, args.seed),
        lambda: _run_batched(data_h0, T_vec, args.backend, args.change_fraction, n_samples, p, Sigma_1, Sigma_2, args.seed),
    )

    title = f"Gaussian GLRT power  (p={p}, n={n_samples}, PFA={args.pfa})"
    finish_h1(args, exporter, h0_stats, h1_stats, T_vec, "mc_gaussian_h1", title, elapsed)


if __name__ == "__main__":
    main()
