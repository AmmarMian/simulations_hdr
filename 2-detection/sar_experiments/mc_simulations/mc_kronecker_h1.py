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
    chunk_trial_ranges,
    finish_h1,
    init_logging,
    make_mc_parser,
    maybe_empty_cache,
    online_run,
    online_single_pass,
    timed_run,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pool workers
# ---------------------------------------------------------------------------

def _worker_h0(args):
    trial_data, T_vec, a, b, iter_max, tol_online, tol_offline = args
    offline = ScaleAndShapeKroneckerGLRT(a, b, "numpy", tol=tol_offline, iter_max=iter_max)
    online = OnlineKroneckerDetector(a, b, "numpy", iter_max=iter_max, tol=tol_online)

    off = {T: float(to_numpy(offline.compute(trial_data[:T]))) for T in T_vec}
    on_raw = online_single_pass(trial_data, T_vec, online)
    on = {T: float(to_numpy(v)) for T, v in on_raw.items()}
    return on, off


def _worker_h1_at_T(args):
    trial_data, a, b, iter_max, tol_online, tol_offline = args
    offline = ScaleAndShapeKroneckerGLRT(a, b, "numpy", tol=tol_offline, iter_max=iter_max)
    online = OnlineKroneckerDetector(a, b, "numpy", iter_max=iter_max, tol=tol_online)
    off = float(to_numpy(offline.compute(trial_data)))
    on = float(to_numpy(online_run(trial_data, online)))
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
    logger.info(f"Starting H0 computation: {n_trials} trials via Pool, {len(T_vec)} T-checkpoints...")
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
                 iter_max, tol_online, tol_offline, chunk_trials=None):
    n_trials = data_h0.shape[0]
    T_max = data_h0.shape[1]
    T_set = set(T_vec)

    # Chunk size bounds per-call GPU memory: each offline.compute() sees
    # (chunk, T, N, p) instead of (n_trials, T, N, p), preventing the
    # PyTorch caching allocator from accumulating GBs of MM intermediates.
    c_starts, chunk, n_chunks = chunk_trial_ranges(n_trials, chunk_trials)

    offline = ScaleAndShapeKroneckerGLRT(a, b, backend, tol=tol_offline, iter_max=iter_max)
    online = OnlineKroneckerDetector(a, b, backend, iter_max=iter_max, tol=tol_online)

    # H0 — process each chunk independently; results concatenated at the end
    h0_on_lists  = {T: [] for T in T_vec}
    h0_off_lists = {T: [] for T in T_vec}

    logger.info(f"Starting H0 batched computation on {backend} ({n_chunks} chunk(s), {len(T_vec)} T-pts)...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            f"[cyan]H0 ({n_chunks} chunk{'s' if n_chunks > 1 else ''}, "
            f"{len(T_vec)} T-pts)...",
            total=n_chunks * (T_max - 2 + len(T_vec)),
        )

        for c_start in c_starts:
            c_end = min(c_start + chunk, n_trials)
            dh0_c = get_data_on_device(data_h0[c_start:c_end], backend)

            # Online: single forward pass
            online.reset_state()
            stat = online.initialize(dh0_c[..., :2, :, :])
            if 2 in T_set:
                h0_on_lists[2].append(to_numpy(stat))
            for t in range(2, T_max):
                stat = online.update(stat, dh0_c[..., t, :, :])
                if t + 1 in T_set:
                    h0_on_lists[t + 1].append(to_numpy(stat))
                progress.advance(task)

            # Offline: one call per T checkpoint
            for T in T_vec:
                h0_off_lists[T].append(to_numpy(offline.compute(dh0_c[..., :T, :, :])))
                maybe_empty_cache(backend)
                progress.advance(task)

            del dh0_c
            maybe_empty_cache(backend)

    h0_stats = {
        "online":  {T: np.concatenate(h0_on_lists[T])  for T in T_vec},
        "offline": {T: np.concatenate(h0_off_lists[T]) for T in T_vec},
    }

    # H1 — generate data per T, then chunk over trials
    h1_on_lists  = {T: [] for T in T_vec}
    h1_off_lists = {T: [] for T in T_vec}

    logger.info(f"Starting H1 batched T-loop on {backend} ({len(T_vec)} T-values, {n_chunks} chunk(s))...")
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
            for c_start in c_starts:
                c_end = min(c_start + chunk, n_trials)
                dh1_c = get_data_on_device(data_h1_T[c_start:c_end], backend)
                h1_off_lists[T].append(to_numpy(offline.compute(dh1_c)))
                h1_on_lists[T].append(to_numpy(online_run(dh1_c, online)))
                del dh1_c
                maybe_empty_cache(backend)
            progress.advance(task)

    h1_stats = {
        "online":  {T: np.concatenate(h1_on_lists[T])  for T in T_vec},
        "offline": {T: np.concatenate(h1_off_lists[T]) for T in T_vec},
    }
    return h0_stats, h1_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
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
    parser.add_argument("--chunk-trials", type=int, default=None,
        help="Split n_trials into chunks of this size for GPU batched path. "
             "Bounds per-call GPU memory: peak ∝ chunk × T_max × N × p. "
             "Default: 32 for CUDA, all trials for other backends.")
    args = parser.parse_args()

    init_logging(args.backend)

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

    chunk_trials = args.chunk_trials
    if chunk_trials is None and "cuda" in args.backend:
        chunk_trials = 32

    logger.info(f"Kronecker H1 MC: a={a}, b={b}, p={p}, n_samples={n_samples}, "
                f"n_trials={args.n_trials}, T=[{T_vec[0]}..{T_vec[-1]}] ({len(T_vec)} pts), "
                f"backend={args.backend}"
                + (f", chunk_trials={chunk_trials}" if chunk_trials else ""))
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

    (h0_stats, h1_stats), elapsed = timed_run(
        args,
        lambda: _run_pool(data_h0, T_vec, args.n_workers, a, b, n_samples, args.change_fraction,
                          A1, B1, A2, B2, args.seed, args.tau_shape, args.tau_scale,
                          args.iter_max, args.tol_online, args.tol_offline),
        lambda: _run_batched(data_h0, T_vec, args.backend, a, b, n_samples, args.change_fraction,
                             A1, B1, A2, B2, args.seed, args.tau_shape, args.tau_scale,
                             args.iter_max, args.tol_online, args.tol_offline, chunk_trials=chunk_trials),
    )

    title = (f"Kronecker GLRT power  (a={a}, b={b}, n={n_samples}, "
             f"tau~Gamma({args.tau_shape},{args.tau_scale}), PFA={args.pfa})")
    finish_h1(args, exporter, h0_stats, h1_stats, T_vec, "mc_kronecker_h1", title, elapsed)


if __name__ == "__main__":
    main()
