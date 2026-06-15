#!/usr/bin/env python
"""MC PFA-vs-threshold curves for sonar two-array detectors.

Collects H0 test statistics from all detectors under Gaussian or K-distributed
clutter and plots empirical P(stat > eta) vs eta — verifying matrix-CFAR
behaviour.  Includes both known-M and adaptive (2TYL / SCM) variants.

Backend selection:
  numpy     → multiprocessing.Pool, one chunk of trials per worker
  all other → trials chunked on device (GPU memory budget via --chunk-size)
"""

from __future__ import annotations

import logging
import math
import os
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
    mmf   = det.MimoMatchedFilter(m, M, P, backend_name)

    tyl_est = est.TwoArrayTylerEstimator(m, backend_name=backend_name)
    scm_est = SCMEstimator(backend_name=backend_name)

    return {
        "M-NMF-G":       glrt,
        "M-NMF-R":       rao,
        "M-NMF-I":       indep,
        "MIMO-MF":       mmf,
        "M-ANMF-G-TYL":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P, backend_name), tyl_est),
        "M-ANMF-R-TYL":  det.AdaptiveSonarDetector(det.MNMFRao(m, M, P, backend_name),  tyl_est),
        "M-ANMF-G-SCM":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P, backend_name), scm_est),
    }


# ---------------------------------------------------------------------------
# Pool worker (numpy only — generates own data to avoid large IPC)
# ---------------------------------------------------------------------------

def _worker(args):
    chunk_seed, n, m, K, M, P, tau_shape, tau_scale = args
    dets = _build_detectors(m, M, P, "numpy")
    x    = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale, seed=chunk_seed)
    xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale, seed=chunk_seed + 10000)
    return smc.run_detectors(x, dets, X_secondary=xsec)  # {name: (n,) array}


def _run_pool(n_trials, chunk_size, m, K, M, P, tau_shape, tau_scale, seed, n_workers):
    # One task per worker — chunk_size is for GPU batching only.
    n_w = n_workers or os.cpu_count() or 1
    worker_chunk = math.ceil(n_trials / n_w)
    c_starts, chunk, n_chunks = chunk_trial_ranges(n_trials, worker_chunk)
    worker_args = [
        (seed + c, min(chunk, n_trials - c), m, K, M, P, tau_shape, tau_scale)
        for c in c_starts
    ]
    all_stats: list[dict] = []

    logger.info(f"H0: {n_trials} trials via Pool ({n_chunks} workers, ≤{chunk} trials each)...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} workers"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]MC trials (Pool)...", total=n_chunks)
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_worker, worker_args):
                all_stats.append(result)
                progress.advance(task)

    return {
        name: np.concatenate([d[name] for d in all_stats])
        for name in all_stats[0]
    }


# ---------------------------------------------------------------------------
# Batched path (non-numpy backends)
# ---------------------------------------------------------------------------

def _run_batched(n_trials, chunk_size, backend, m, K, M, P, tau_shape, tau_scale, seed):
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
        task = progress.add_task(f"[cyan]MC batched ({backend})...", total=n_chunks)
        for c in c_starts:
            n = min(chunk, n_trials - c)
            x    = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale, seed=seed + c)
            xsec = sim.generate_secondary_data(n, K, m, M, tau_shape, tau_scale, seed=seed + c + 10000)
            x_dev    = get_data_on_device(x,    backend)
            xsec_dev = get_data_on_device(xsec, backend)
            cs = smc.run_detectors(x_dev, dets, X_secondary=xsec_dev)
            for name, s in cs.items():
                all_stats[name].append(s)
            maybe_empty_cache(backend)
            progress.advance(task)

    return {name: np.concatenate(v) for name, v in all_stats.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    smc.add_mc_args(parser)
    parser.add_argument("--n-thresh", type=int, default=500,
        help="Number of threshold grid points for PFA curve (default 500).")
    parser.add_argument("--chunk-size", type=int, default=_CHUNK,
        help=f"Trials per worker chunk (numpy/Pool) or per GPU memory batch (non-numpy). "
             f"Default {_CHUNK}.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = args.m
    K = smc.resolve_K(args)
    tau_shape, tau_scale = smc.clutter_params(args)

    logger.info(f"Sonar PFA-vs-threshold: m={m}, K={K}, clutter={args.clutter}, "
                f"n_trials={args.n_trials}, backend={args.backend}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)
    P = sim.make_steering_matrix(m, args.theta1, args.theta2)

    all_stats, elapsed = timed_run(
        args,
        lambda: _run_pool(args.n_trials, args.chunk_size, m, K, M, P,
                          tau_shape, tau_scale, args.seed, args.n_workers),
        lambda: _run_batched(args.n_trials, args.chunk_size, args.backend, m, K, M, P,
                             tau_shape, tau_scale, args.seed),
    )

    logger.info(f"Done in {elapsed:.1f}s")

    detector_names = list(all_stats.keys())
    export_stats = {"detector_names": np.array(detector_names)}
    for name in detector_names:
        thresh, pfa = smc.aggregate_pfa_threshold(all_stats[name], args.n_thresh)
        export_stats[f"thresh_{name}"] = thresh
        export_stats[f"pfa_{name}"] = pfa

    clutter_tag = args.clutter if args.clutter == "gaussian" else f"k_nu{args.nu}"
    stem_tag = f"m{m}_K{K}_{clutter_tag}_n{args.n_trials}"

    exporter = MCResultExporter(args, Path(args.export_path), stem_tag,
                                plot_template=smc._MC_PFA_THRESHOLD_TEMPLATE)
    title = (f"PFA vs threshold — sonar (m={m}, K={K}, {args.clutter} clutter)")
    exporter.save(export_stats, "sonar_pfa_threshold", elapsed, title)


if __name__ == "__main__":
    main()
