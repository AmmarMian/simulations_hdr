#!/usr/bin/env python
"""MC PD vs (theta1, theta2) angle map for sonar two-array detectors.

Sweeps over a 2-D grid of steering angles at a fixed SNR (--snr-db) and
fixed nominal PFA.  Threshold is calibrated from H0 trials with the
nominal steering direction.  Produces one PD heatmap per detector.

The angle grid is symmetric and covers both arrays simultaneously.

Backend selection:
  numpy     → multiprocessing.Pool for H0 (trial chunks) and H1 (angle grid cells)
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

def _build_detectors(m, M, P_nominal, backend_name="numpy"):
    glrt  = det.MNMFGlrt(m, M, P_nominal, backend_name)
    rao   = det.MNMFRao(m, M, P_nominal, backend_name)
    indep = det.MNMFIndependent(m, M, P_nominal, backend_name)

    tyl_est = est.TwoArrayTylerEstimator(m, backend_name=backend_name)
    scm_est = SCMEstimator(backend_name=backend_name)

    return {
        "M-NMF-G":       glrt,
        "M-NMF-R":       rao,
        "M-NMF-I":       indep,
        "M-ANMF-G-TYL":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P_nominal, backend_name), tyl_est),
        "M-ANMF-G-SCM":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P_nominal, backend_name), scm_est),
    }


# ---------------------------------------------------------------------------
# Pool worker: H0 trial chunk
# ---------------------------------------------------------------------------

def _worker_h0(args):
    chunk_seed, n, m, K, M, P_nominal, tau_shape, tau_scale = args
    dets = _build_detectors(m, M, P_nominal, "numpy")
    x    = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale, seed=chunk_seed)
    xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale, seed=chunk_seed + 10000)
    return smc.run_detectors(x, dets, X_secondary=xsec)


def _run_pool_h0(n_trials, chunk_size, m, K, M, P_nominal, tau_shape, tau_scale, seed, n_workers):
    c_starts, chunk, n_chunks = chunk_trial_ranges(n_trials, chunk_size)
    worker_args = [
        (seed + c, min(chunk, n_trials - c), m, K, M, P_nominal, tau_shape, tau_scale)
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
# Pool worker: H1 angle — one (theta1, theta2) cell per worker
# ---------------------------------------------------------------------------

def _worker_angle(args):
    i, j, th1, th2, alpha_fixed, n_trials, chunk_size, m, K, M, P_nominal, \
        tau_shape, tau_scale, seed, thresholds = args
    P_ij = sim.make_steering_matrix(m, float(th1), float(th2))
    dets = _build_detectors(m, M, P_nominal, "numpy")
    h1_stats: dict[str, list] = {name: [] for name in dets}
    c_starts, chunk, _ = chunk_trial_ranges(n_trials, chunk_size)
    for c in c_starts:
        n = min(chunk, n_trials - c)
        x    = sim.generate_sonar_data_h1(n, m, M, P_ij, alpha_fixed, tau_shape, tau_scale,
                                           seed=seed + c + i * 100000 + j * 1000)
        xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale,
                                           seed=seed + 10000 + c + i * 100000 + j * 1000)
        cs = smc.run_detectors(x, dets, X_secondary=xsec)
        for name, s in cs.items():
            h1_stats[name].append(s)
    h1_stats = {name: np.concatenate(v) for name, v in h1_stats.items()}
    return i, j, {name: smc.empirical_pd(h1_stats[name], thresholds[name]) for name in dets}


def _run_pool_angles(theta_grid, alpha_fixed, n_trials, chunk_size, m, K, M, P_nominal,
                     tau_shape, tau_scale, seed, thresholds, n_workers):
    n_ang = len(theta_grid)
    n_tasks = n_ang * n_ang
    worker_args = [
        (i, j, theta_grid[i], theta_grid[j], alpha_fixed, n_trials, chunk_size,
         m, K, M, P_nominal, tau_shape, tau_scale, seed, thresholds)
        for i in range(n_ang) for j in range(n_ang)
    ]
    det_names = list(thresholds.keys())
    pd_maps = {name: np.zeros((n_ang, n_ang)) for name in det_names}

    logger.info(f"H1 angle grid: {n_ang}×{n_ang}={n_tasks} cells via Pool...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} cells"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[green]Angle grid (Pool)...", total=n_tasks)
        with Pool(processes=n_workers) as pool:
            for i, j, pd_dict in pool.imap_unordered(_worker_angle, worker_args):
                for name, pd_val in pd_dict.items():
                    pd_maps[name][i, j] = pd_val
                progress.advance(task)

    return pd_maps


# ---------------------------------------------------------------------------
# Batched path (non-numpy backends)
# ---------------------------------------------------------------------------

def _run_batched_h0(n_trials, chunk_size, backend, m, K, M, P_nominal, tau_shape, tau_scale, seed):
    dets = _build_detectors(m, M, P_nominal, backend)
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


def _run_batched_angles(theta_grid, alpha_fixed, n_trials, chunk_size, backend,
                        m, K, M, P_nominal, tau_shape, tau_scale, seed, thresholds):
    n_ang = len(theta_grid)
    det_names = list(thresholds.keys())
    pd_maps = {name: np.zeros((n_ang, n_ang)) for name in det_names}
    c_starts, chunk, _ = chunk_trial_ranges(n_trials, chunk_size)

    logger.info(f"H1 angle grid: {n_ang}×{n_ang} cells on {backend}...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} cells"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[green]Angle grid ({backend})...", total=n_ang * n_ang)
        for i, th1 in enumerate(theta_grid):
            for j, th2 in enumerate(theta_grid):
                P_ij = sim.make_steering_matrix(m, float(th1), float(th2))
                dets = _build_detectors(m, M, P_ij, backend)
                h1_stats: dict[str, list] = {name: [] for name in dets}
                for c in c_starts:
                    n = min(chunk, n_trials - c)
                    x    = sim.generate_sonar_data_h1(n, m, M, P_ij, alpha_fixed, tau_shape, tau_scale,
                                                       seed=seed + c + i * 100000 + j * 1000)
                    xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale,
                                                       seed=seed + 10000 + c + i * 100000 + j * 1000)
                    cs = smc.run_detectors(get_data_on_device(x, backend), dets,
                                           X_secondary=get_data_on_device(xsec, backend))
                    for name, s in cs.items():
                        h1_stats[name].append(s)
                    maybe_empty_cache(backend)
                for name in det_names:
                    s_h1 = np.concatenate(h1_stats[name])
                    pd_maps[name][i, j] = smc.empirical_pd(s_h1, thresholds[name])
                progress.advance(task)

    return pd_maps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    smc.add_mc_args(parser)
    smc.add_angle_args(parser)
    parser.add_argument("--n-trials-h0", type=int, default=1000,
        help="Trials for H0 threshold calibration (default 1000).")
    parser.add_argument("--chunk-size", type=int, default=_CHUNK,
        help=f"Trials per worker chunk (numpy/Pool) or per GPU memory batch (non-numpy). "
             f"Default {_CHUNK}.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = args.m
    K = smc.resolve_K(args)
    tau_shape, tau_scale = smc.clutter_params(args)

    logger.info(f"Sonar PD-vs-angle: m={m}, K={K}, clutter={args.clutter}, "
                f"snr={args.snr_db} dB, n_angles={args.n_theta}×{args.n_theta}, "
                f"n_trials={args.n_trials}, backend={args.backend}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)
    P_nominal = sim.make_steering_matrix(m, args.theta1, args.theta2)

    # Calibrate thresholds from H0 (nominal steering, no signal)
    logger.info("Calibrating thresholds from H0...")
    stats_h0, t_h0 = timed_run(
        args,
        lambda: _run_pool_h0(args.n_trials_h0, args.chunk_size, m, K, M, P_nominal,
                             tau_shape, tau_scale, args.seed, args.n_workers),
        lambda: _run_batched_h0(args.n_trials_h0, args.chunk_size, args.backend, m, K, M, P_nominal,
                                tau_shape, tau_scale, args.seed),
    )
    thresholds = {name: smc.threshold_at_pfa(stats_h0[name], args.pfa) for name in stats_h0}
    logger.info(f"Thresholds: { {k: f'{v:.4f}' for k, v in thresholds.items()} }")

    # Signal amplitude for fixed SNR
    _, alphas = sim.snr_alpha_sweep(m, M, P_nominal,
                                    snr_min_db=args.snr_db, snr_max_db=args.snr_db, n_snr=1)
    alpha_fixed = float(alphas[0])
    logger.info(f"Alpha for SNR={args.snr_db} dB: {alpha_fixed:.6f}")

    theta_grid = np.linspace(args.theta_min, args.theta_max, args.n_theta)

    # H1 angle grid
    pd_maps, t_h1 = timed_run(
        args,
        lambda: _run_pool_angles(theta_grid, alpha_fixed, args.n_trials, args.chunk_size,
                                 m, K, M, P_nominal, tau_shape, tau_scale,
                                 args.seed + 1, thresholds, args.n_workers),
        lambda: _run_batched_angles(theta_grid, alpha_fixed, args.n_trials, args.chunk_size,
                                    args.backend, m, K, M, P_nominal, tau_shape, tau_scale,
                                    args.seed + 1, thresholds),
    )

    elapsed = t_h0 + t_h1
    logger.info(f"Done in {elapsed:.1f}s (H0: {t_h0:.1f}s, H1: {t_h1:.1f}s)")

    detector_names = list(pd_maps.keys())
    export_stats = {
        "theta_grid":     theta_grid,
        "detector_names": np.array(detector_names),
        "snr_db":         np.array([args.snr_db]),
    }
    for name in detector_names:
        export_stats[f"pd_map_{name}"] = pd_maps[name]

    clutter_tag = args.clutter if args.clutter == "gaussian" else f"k_nu{args.nu}"
    stem_tag = f"m{m}_K{K}_{clutter_tag}_snr{args.snr_db:.0f}dB_n{args.n_trials}"

    exporter = MCResultExporter(args, Path(args.export_path), stem_tag,
                                plot_template=smc._MC_PD_ANGLE_TEMPLATE)
    title = (f"PD angle map — sonar (m={m}, K={K}, {args.clutter}, SNR={args.snr_db} dB)")
    exporter.save(export_stats, "sonar_pd_angle", elapsed, title)


if __name__ == "__main__":
    main()
