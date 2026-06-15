#!/usr/bin/env python
"""MC detection-probability vs SNR for sonar two-array detectors.

Generates empirical PD curves at a fixed nominal PFA by sweeping SNR over a
logarithmic grid.  Two separate trial budgets are used: n_trials_h0 for
threshold estimation (PFA calibration) and n_trials_h1 for PD estimation.

Experiment covers both known-M detectors (M-NMF-G, M-NMF-R, M-NMF-I, SA-1,
SA-2, MIMO-MF) and their adaptive 2TYL variants.  Gaussian and K-distributed
clutter are both supported via --gaussian / --k-dist flags.

Backend selection:
  numpy     → multiprocessing.Pool for both H0 (trial chunks) and H1 (SNR points)
  all other → chunked batched path on device
"""

from __future__ import annotations

import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from hdrlib.core.backend import get_data_on_device
from hdrlib.core.estimation import SCMEstimator
from hdrlib.core.mc import (
    MCResultExporter,
    chunk_trial_ranges,
    make_mc_parser,
    maybe_empty_cache,
    timed_run,
)
from hdrlib.sonar import detectors as det
from hdrlib.sonar import estimation as est
from hdrlib.sonar import mc as smc
from hdrlib.sonar import simulation as sim

logger = logging.getLogger(__name__)

_CHUNK = 200


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

def _build_detectors(m, M, P, backend_name="numpy"):
    glrt  = det.MNMFGlrt(m, M, P, backend_name)
    rao   = det.MNMFRao(m, M, P, backend_name)
    indep = det.MNMFIndependent(m, M, P, backend_name)
    sa1   = det.NMFSingleArray(m, M, P, array_idx=0, backend_name=backend_name)
    sa2   = det.NMFSingleArray(m, M, P, array_idx=1, backend_name=backend_name)
    mmf   = det.MimoMatchedFilter(m, M, P, backend_name)

    tyl_est = est.TwoArrayTylerEstimator(m, backend_name=backend_name)
    scm_est = SCMEstimator(backend_name=backend_name)

    return {
        "M-NMF-G":       glrt,
        "M-NMF-R":       rao,
        "M-NMF-I":       indep,
        "SA-1":          sa1,
        "SA-2":          sa2,
        "MIMO-MF":       mmf,
        "M-ANMF-G-TYL":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P, backend_name), tyl_est),
        "M-ANMF-R-TYL":  det.AdaptiveSonarDetector(det.MNMFRao(m, M, P, backend_name),  tyl_est),
        "M-ANMF-G-SCM":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P, backend_name), scm_est),
        "M-ANMF-R-SCM":  det.AdaptiveSonarDetector(det.MNMFRao(m, M, P, backend_name),  scm_est),
    }


# ---------------------------------------------------------------------------
# Pool worker: H0 trial chunk
# ---------------------------------------------------------------------------

def _worker_h0(args):
    chunk_seed, n, m, K, M, P, tau_shape, tau_scale = args
    dets = _build_detectors(m, M, P, "numpy")
    x    = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale, seed=chunk_seed)
    xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale, seed=chunk_seed + 10000)
    return smc.run_detectors(x, dets, X_secondary=xsec)


def _run_pool_h0(n_trials, chunk_size, m, K, M, P, tau_shape, tau_scale, seed, n_workers):
    c_starts, chunk, n_chunks = chunk_trial_ranges(n_trials, chunk_size)
    worker_args = [
        (seed + c, min(chunk, n_trials - c), m, K, M, P, tau_shape, tau_scale)
        for c in c_starts
    ]
    all_stats: list[dict] = []

    logger.info(f"H0: {n_trials} trials via Pool ({n_chunks} chunks of ≤{chunk})...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]H0 trials (Pool)...", total=n_chunks)
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_worker_h0, worker_args):
                all_stats.append(result)
                progress.advance(task)

    return {name: np.concatenate([d[name] for d in all_stats]) for name in all_stats[0]}


# ---------------------------------------------------------------------------
# Pool worker: H1 — one SNR point per worker (includes inner trial loop)
# ---------------------------------------------------------------------------

def _worker_h1_snr(args):
    snr_idx, alpha, n_trials_h1, chunk_size, m, K, M, P, tau_shape, tau_scale, seed, thresholds = args
    dets = _build_detectors(m, M, P, "numpy")
    stats_h1: dict[str, list] = {name: [] for name in dets}
    c_starts, chunk, _ = chunk_trial_ranges(n_trials_h1, chunk_size)
    for c in c_starts:
        n = min(chunk, n_trials_h1 - c)
        x    = sim.generate_sonar_data_h1(n, m, M, P, alpha, tau_shape, tau_scale,
                                           seed=seed + c + snr_idx * 100000)
        xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale,
                                           seed=seed + 10000 + c + snr_idx * 100000)
        cs = smc.run_detectors(x, dets, X_secondary=xsec)
        for name, s in cs.items():
            stats_h1[name].append(s)
    stats_h1 = {name: np.concatenate(v) for name, v in stats_h1.items()}
    return {name: smc.empirical_pd(stats_h1[name], thresholds[name]) for name in dets}


def _run_pool_h1(snr_db, alphas, n_trials_h1, chunk_size, m, K, M, P,
                 tau_shape, tau_scale, seed, thresholds, n_workers):
    n_snr = len(snr_db)
    worker_args = [
        (i, float(alphas[i]), n_trials_h1, chunk_size, m, K, M, P,
         tau_shape, tau_scale, seed, thresholds)
        for i in range(n_snr)
    ]
    det_names = list(thresholds.keys())
    pd_by_snr: dict[str, np.ndarray] = {name: np.zeros(n_snr) for name in det_names}

    logger.info(f"H1: {n_snr} SNR points via Pool...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} SNR pts"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[green]H1 SNR sweep (Pool)...", total=n_snr)
        with Pool(processes=n_workers) as pool:
            for i, pd_dict in enumerate(pool.imap(_worker_h1_snr, worker_args)):
                for name, pd_val in pd_dict.items():
                    pd_by_snr[name][i] = pd_val
                progress.advance(task)

    return pd_by_snr


# ---------------------------------------------------------------------------
# Batched path (non-numpy backends)
# ---------------------------------------------------------------------------

def _run_batched_h0(n_trials, chunk_size, backend, m, K, M, P, tau_shape, tau_scale, seed):
    dets = _build_detectors(m, M, P, backend)
    c_starts, chunk, n_chunks = chunk_trial_ranges(n_trials, chunk_size)
    all_stats: dict[str, list] = {name: [] for name in dets}

    logger.info(f"H0: {n_trials} trials on {backend} ({n_chunks} chunks of ≤{chunk})...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]H0 batched ({backend})...", total=n_chunks)
        for c in c_starts:
            n = min(chunk, n_trials - c)
            x    = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale, seed=seed + c)
            xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale, seed=seed + c + 10000)
            cs = smc.run_detectors(get_data_on_device(x, backend), dets,
                                   X_secondary=get_data_on_device(xsec, backend))
            for name, s in cs.items():
                all_stats[name].append(s)
            maybe_empty_cache(backend)
            progress.advance(task)

    return {name: np.concatenate(v) for name, v in all_stats.items()}


def _run_batched_h1(snr_db, alphas, n_trials_h1, chunk_size, backend,
                    m, K, M, P, tau_shape, tau_scale, seed, thresholds):
    dets = _build_detectors(m, M, P, backend)
    n_snr = len(snr_db)
    det_names = list(thresholds.keys())
    pd_by_snr: dict[str, np.ndarray] = {name: np.zeros(n_snr) for name in det_names}
    c_starts, chunk, _ = chunk_trial_ranges(n_trials_h1, chunk_size)

    logger.info(f"H1: {n_snr} SNR points on {backend}...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} SNR pts"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[green]H1 SNR sweep ({backend})...", total=n_snr)
        for i, (snr, alpha) in enumerate(zip(snr_db, alphas)):
            stats_h1: dict[str, list] = {name: [] for name in dets}
            for c in c_starts:
                n = min(chunk, n_trials_h1 - c)
                x    = sim.generate_sonar_data_h1(n, m, M, P, float(alpha), tau_shape, tau_scale,
                                                   seed=seed + c + i * 100000)
                xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale,
                                                   seed=seed + 10000 + c + i * 100000)
                cs = smc.run_detectors(get_data_on_device(x, backend), dets,
                                       X_secondary=get_data_on_device(xsec, backend))
                for name, s in cs.items():
                    stats_h1[name].append(s)
                maybe_empty_cache(backend)
            for name in det_names:
                s_h1 = np.concatenate(stats_h1[name])
                pd_by_snr[name][i] = smc.empirical_pd(s_h1, thresholds[name])
            progress.advance(task)

    return pd_by_snr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    smc.add_mc_args(parser)
    parser.add_argument("--n-trials-h0", type=int, default=1000,
        help="Trials for H0 threshold calibration (default 1000).")
    parser.add_argument("--n-trials-h1", type=int, default=10000,
        help="Trials for H1 PD estimation (default 10000).")
    parser.add_argument("--chunk-size", type=int, default=_CHUNK,
        help=f"Trials per worker chunk (numpy/Pool) or per GPU memory batch (non-numpy). "
             f"Default {_CHUNK}.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = args.m
    K = smc.resolve_K(args)
    tau_shape, tau_scale = smc.clutter_params(args)

    logger.info(f"Sonar PD-vs-SNR: m={m}, K={K}, clutter={args.clutter} "
                f"(nu={args.nu}), n_h0={args.n_trials_h0}, n_h1={args.n_trials_h1}, "
                f"n_snr={args.n_snr}, backend={args.backend}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)
    P = sim.make_steering_matrix(m, args.theta1, args.theta2)

    snr_db, alphas = sim.snr_alpha_sweep(m, M, P, args.snr_min, args.snr_max, args.n_snr)

    # H0 phase
    logger.info("Running H0 trials...")
    stats_h0, t_h0 = timed_run(
        args,
        lambda: _run_pool_h0(args.n_trials_h0, args.chunk_size, m, K, M, P,
                             tau_shape, tau_scale, args.seed, args.n_workers),
        lambda: _run_batched_h0(args.n_trials_h0, args.chunk_size, args.backend, m, K, M, P,
                                tau_shape, tau_scale, args.seed),
    )
    thresholds = {name: smc.threshold_at_pfa(stats_h0[name], args.pfa) for name in stats_h0}

    # H1 phase
    logger.info("Running H1 trials across SNR grid...")
    pd_by_snr, t_h1 = timed_run(
        args,
        lambda: _run_pool_h1(snr_db, alphas, args.n_trials_h1, args.chunk_size, m, K, M, P,
                             tau_shape, tau_scale, args.seed + 1, thresholds, args.n_workers),
        lambda: _run_batched_h1(snr_db, alphas, args.n_trials_h1, args.chunk_size, args.backend,
                                m, K, M, P, tau_shape, tau_scale, args.seed + 1, thresholds),
    )

    elapsed = t_h0 + t_h1
    logger.info(f"Done in {elapsed:.1f}s (H0: {t_h0:.1f}s, H1: {t_h1:.1f}s)")

    detector_names = list(pd_by_snr.keys())
    export_stats = {"snr_db": snr_db, "detector_names": np.array(detector_names)}
    for name in detector_names:
        export_stats[f"pd_{name}"] = pd_by_snr[name]
        export_stats[f"h0_{name}"] = stats_h0[name]

    clutter_tag = args.clutter if args.clutter == "gaussian" else f"k_nu{args.nu}"
    stem_tag = f"m{m}_K{K}_{clutter_tag}_n{args.n_trials_h1}"

    exporter = MCResultExporter(args, Path(args.export_path), stem_tag,
                                plot_template=smc._MC_PD_SNR_TEMPLATE)
    title = (f"PD vs SNR — sonar (m={m}, K={K}, {args.clutter} "
             f"clutter, PFA={args.pfa})")
    exporter.save(export_stats, "sonar_pd_snr", elapsed, title)

    logger.info(f"Peak PD (M-NMF-G): {pd_by_snr['M-NMF-G'].max():.3f}")


if __name__ == "__main__":
    main()
