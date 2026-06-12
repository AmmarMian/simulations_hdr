#!/usr/bin/env python
"""MC power simulation: OnlineKroneckerDetector vs ScaleAndShapeKroneckerGLRT under H1.

For each T in T_vec, H1 data is generated fresh with the change at
  n_change_dates = max(2, int(T * change_fraction))   (default: T // 2)
so that pre- and post-change segments always have ~T/2 dates each.

H0 statistics are collected in a single pass over T_max dates (no change).
Thresholds are set at the (1-PFA) percentile of H0 stats for each T.

Backend selection:
  numpy     → multiprocessing.Pool, one trial per worker
  all other → trials in leading batch dim, single-pass on device
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

_HERE = Path(__file__).parent
_ROOT = str(_HERE.parent.parent)
for _p in (_ROOT, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.backend import get_data_on_device
from src.simulation import (
    T_vec_logspace,
    generate_kronecker_data,
    generate_kronecker_data_h1,
    make_ab_true,
)
from sar_experiments.detection_offline import ScaleAndShapeKroneckerGLRT
from src.detection_online import OnlineKroneckerDetector
from utils import (
    MCResultExporter,
    _MC_PLOT_TEMPLATE_H1,
    add_mc_args,
    add_mc_h1_args,
    aggregate_power,
    maybe_empty_cache,
    plot_mc_power,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)


def _online_run(data, detector):
    """Run online detector through all T dates, return final stat."""
    T = data.shape[-3]
    detector.reset_state()
    stat = detector.initialize(data[..., :2, :, :])
    for t in range(2, T):
        stat = detector.update(stat, data[..., t, :, :])
    return stat


def _online_single_pass(data, T_vec, detector):
    """Single-pass online detection, collecting stats at T checkpoints."""
    T_set = set(T_vec)
    T_max = data.shape[-3]
    results = {}
    detector.reset_state()
    stat = detector.initialize(data[..., :2, :, :])
    if 2 in T_set:
        results[2] = stat
    for t in range(2, T_max):
        T_current = t + 1
        stat = detector.update(stat, data[..., t, :, :])
        if T_current in T_set:
            results[T_current] = stat
    return results


# ---------------------------------------------------------------------------
# Pool workers
# ---------------------------------------------------------------------------

def _worker_h0(args):
    trial_data, T_vec, a, b, iter_max, tol_online, tol_offline = args
    offline = ScaleAndShapeKroneckerGLRT(a, b, "numpy", tol=tol_offline, iter_max=iter_max)
    online = OnlineKroneckerDetector(a, b, "numpy", iter_max=iter_max, tol=tol_online)

    off = {T: float(_to_numpy(offline.compute(trial_data[:T]))) for T in T_vec}
    on_raw = _online_single_pass(trial_data, T_vec, online)
    on = {T: float(_to_numpy(v)) for T, v in on_raw.items()}
    return on, off


def _worker_h1_at_T(args):
    trial_data, a, b, iter_max, tol_online, tol_offline = args
    offline = ScaleAndShapeKroneckerGLRT(a, b, "numpy", tol=tol_offline, iter_max=iter_max)
    online = OnlineKroneckerDetector(a, b, "numpy", iter_max=iter_max, tol=tol_online)
    off = float(_to_numpy(offline.compute(trial_data)))
    on = float(_to_numpy(_online_run(trial_data, online)))
    return on, off


# ---------------------------------------------------------------------------
# Pool runners
# ---------------------------------------------------------------------------

def _run_pool(data_h0, T_vec, n_workers, a, b, n_samples, change_fraction,
              A1, B1, A2, B2, seed, tau_shape, tau_scale,
              iter_max, tol_online, tol_offline):
    n_trials = data_h0.shape[0]

    # H0: single pass over all T checkpoints
    h0_on_all, h0_off_all = [], []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} trials"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]H0 trials...", total=n_trials)
        with Pool(processes=n_workers) as pool:
            worker_args = [
                (data_h0[i], T_vec, a, b, iter_max, tol_online, tol_offline)
                for i in range(n_trials)
            ]
            for on, off in pool.imap_unordered(_worker_h0, worker_args):
                h0_on_all.append(on)
                h0_off_all.append(off)
                progress.advance(task)

    h0_stats = {
        "online":  {T: np.array([d[T] for d in h0_on_all])  for T in T_vec},
        "offline": {T: np.array([d[T] for d in h0_off_all]) for T in T_vec},
    }

    # H1: fresh data per T, change at T * change_fraction
    h1_on_all = {T: [] for T in T_vec}
    h1_off_all = {T: [] for T in T_vec}

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
                data_h1_T = generate_kronecker_data_h1(
                    n_trials, T, n_samples, a, b, A1, B1, A2, B2,
                    seed=seed, n_change_dates=n_change,
                    tau_shape=tau_shape, tau_scale=tau_scale,
                )
                worker_args = [
                    (data_h1_T[i], a, b, iter_max, tol_online, tol_offline)
                    for i in range(n_trials)
                ]
                for on, off in pool.imap_unordered(_worker_h1_at_T, worker_args):
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

def _run_batched(data_h0, T_vec, backend, a, b, n_samples, change_fraction,
                 A1, B1, A2, B2, seed, tau_shape, tau_scale,
                 iter_max, tol_online, tol_offline):
    dh0 = get_data_on_device(data_h0, backend)
    T_max = data_h0.shape[1]
    T_set = set(T_vec)
    n_trials = data_h0.shape[0]

    offline = ScaleAndShapeKroneckerGLRT(a, b, backend, tol=tol_offline, iter_max=iter_max)
    online = OnlineKroneckerDetector(a, b, backend, iter_max=iter_max, tol=tol_online)

    # H0
    h0_on_raw, h0_off_raw = {}, {}
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
            h0_on_raw[2] = _to_numpy(stat)
        for t in range(2, T_max):
            stat = online.update(stat, dh0[..., t, :, :])
            if t + 1 in T_set:
                h0_on_raw[t + 1] = _to_numpy(stat)
            progress.advance(task_on)

        for T in T_vec:
            h0_off_raw[T] = _to_numpy(offline.compute(dh0[..., :T, :, :]))
            maybe_empty_cache(backend)
            progress.advance(task_off)

    h0_stats = {"online": h0_on_raw, "offline": h0_off_raw}

    # H1: per T
    h1_on_raw, h1_off_raw = {}, {}
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} T-values"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[green]H1 T-loop...", total=len(T_vec))
        for T in T_vec:
            n_change = max(2, int(T * change_fraction))
            data_h1_T = generate_kronecker_data_h1(
                n_trials, T, n_samples, a, b, A1, B1, A2, B2,
                seed=seed, n_change_dates=n_change,
                tau_shape=tau_shape, tau_scale=tau_scale,
            )
            dh1 = get_data_on_device(data_h1_T, backend)
            h1_off_raw[T] = _to_numpy(offline.compute(dh1))
            h1_on_raw[T] = _to_numpy(_online_run(dh1, online))
            maybe_empty_cache(backend)
            progress.advance(task)

    h1_stats = {"online": h1_on_raw, "offline": h1_off_raw}
    return h0_stats, h1_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_mc_args(parser)
    add_mc_h1_args(parser)
    parser.set_defaults(T_max=500, n_trials=50)

    parser.add_argument("--a", type=int, default=2,
        help="Size of first Kronecker factor (default 2).")
    parser.add_argument("--b", type=int, default=3,
        help="Size of second Kronecker factor (default 3).")
    parser.add_argument("--n-samples", type=int, default=None,
        help="Samples per date (default: p+1 = a*b+1).")
    parser.add_argument("--iter-max", type=int, default=20,
        help="Max MM iterations for H0 warm-start and H1 per-date estimates (default 20).")
    parser.add_argument("--tol-online", type=float, default=1e-4,
        help="Convergence tolerance for online estimator (default 1e-4).")
    parser.add_argument("--tol-offline", type=float, default=1e-4,
        help="Convergence tolerance for offline MM estimator (default 1e-4).")
    parser.add_argument("--tau-shape", type=float, default=1.0,
        help="Shape parameter of Gamma(shape, scale) texture (default 1.0).")
    parser.add_argument("--tau-scale", type=float, default=1.0,
        help="Scale parameter of Gamma(shape, scale) texture (default 1.0).")
    parser.add_argument("--seed-a", type=int, default=0,
        help="Seed for A1_true generation (default 0).")
    parser.add_argument("--seed-b", type=int, default=1,
        help="Seed for B1_true generation (default 1).")
    parser.add_argument("--seed-a2", type=int, default=8,
        help="Seed for A2_true (H1 distribution, default 8).")
    parser.add_argument("--seed-b2", type=int, default=3,
        help="Seed for B2_true (H1 distribution, default 3).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    min_trials = int(10 / args.pfa)
    if args.n_trials < min_trials:
        logger.warning(
            f"n_trials={args.n_trials} < 10/PFA={min_trials}. "
            f"Threshold estimate at PFA={args.pfa} will be unreliable. "
            f"Consider --n-trials {min_trials}."
        )

    a, b = args.a, args.b
    p = a * b
    n_samples = args.n_samples if args.n_samples is not None else p + 1
    T_vec = T_vec_logspace(args.T_min, args.T_max, args.n_T)
    T_max = max(T_vec)

    logger.info(f"Kronecker H1 MC: a={a}, b={b}, p={p}, n_samples={n_samples}, "
                f"n_trials={args.n_trials}, T=[{T_vec[0]}..{T_vec[-1]}] ({len(T_vec)} pts), "
                f"backend={args.backend}")
    logger.info(f"  tau~Gamma({args.tau_shape},{args.tau_scale}), iter_max={args.iter_max}")
    logger.info(f"  Change fraction={args.change_fraction} | PFA={args.pfa}")

    A1, B1 = make_ab_true(a, b, seed_a=args.seed_a,  seed_b=args.seed_b)
    A2, B2 = make_ab_true(a, b, seed_a=args.seed_a2, seed_b=args.seed_b2)

    logger.info("Generating H0 data...")
    data_h0 = generate_kronecker_data(
        args.n_trials, T_max, n_samples, a, b, A1, B1,
        seed=args.seed, tau_shape=args.tau_shape, tau_scale=args.tau_scale,
    )

    exporter = MCResultExporter(
        args, Path(args.export_path), f"a{a}_b{b}_T{T_max}_n{args.n_trials}",
        plot_template=_MC_PLOT_TEMPLATE_H1,
    )

    t0 = time.perf_counter()
    if args.backend == "numpy":
        h0_stats, h1_stats = _run_pool(
            data_h0, T_vec, args.n_workers, a, b, n_samples, args.change_fraction,
            A1, B1, A2, B2, args.seed,
            args.tau_shape, args.tau_scale,
            args.iter_max, args.tol_online, args.tol_offline,
        )
    else:
        h0_stats, h1_stats = _run_batched(
            data_h0, T_vec, args.backend, a, b, n_samples, args.change_fraction,
            A1, B1, A2, B2, args.seed,
            args.tau_shape, args.tau_scale,
            args.iter_max, args.tol_online, args.tol_offline,
        )
    elapsed = time.perf_counter() - t0

    stats = aggregate_power(h0_stats, h1_stats, T_vec, args.pfa)
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"  Power offline @ T_max: {stats['power_offline'][-1]:.3f}")
    logger.info(f"  Power online  @ T_max: {stats['power_online'][-1]:.3f}")

    title = (f"Kronecker GLRT power  (a={a}, b={b}, n={n_samples}, "
             f"tau~Gamma({args.tau_shape},{args.tau_scale}), PFA={args.pfa})")
    exporter.save(stats, "mc_kronecker_h1", elapsed, title=title)
    if args.show_interactive:
        plot_mc_power(stats, title=title)


if __name__ == "__main__":
    main()
