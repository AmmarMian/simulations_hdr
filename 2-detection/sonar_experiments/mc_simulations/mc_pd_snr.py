#!/usr/bin/env python
"""MC detection-probability vs SNR for sonar two-array detectors.

Generates empirical PD curves at a fixed nominal PFA by sweeping SNR over a
logarithmic grid.  Two separate trial budgets are used: n_trials_h0 for
threshold estimation (PFA calibration) and n_trials_h1 for PD estimation.

Experiment covers both known-M detectors (M-NMF-G, M-NMF-R, M-NMF-I, SA-1,
SA-2, MIMO-MF) and their adaptive 2TYL variants.  Gaussian and K-distributed
clutter are both supported via --gaussian / --k-dist flags.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from hdrlib.core.mc import MCResultExporter, add_mc_base_args, init_logging, make_mc_parser
from hdrlib.sonar import detectors as det
from hdrlib.sonar import estimation as est
from hdrlib.sonar import mc as smc
from hdrlib.sonar import simulation as sim
from hdrlib.core.estimation import SCMEstimator

logger = logging.getLogger(__name__)

_CHUNK = 200  # max trials per chunk (limits memory of secondary-data buffer)


# ---------------------------------------------------------------------------
# Build detector registry
# ---------------------------------------------------------------------------

def _build_detectors(m, M, P):
    glrt  = det.MNMFGlrt(m, M, P)
    rao   = det.MNMFRao(m, M, P)
    indep = det.MNMFIndependent(m, M, P)
    sa1   = det.NMFSingleArray(m, M, P, array_idx=0)
    sa2   = det.NMFSingleArray(m, M, P, array_idx=1)
    mmf   = det.MimoMatchedFilter(m, M, P)

    tyl_est = est.TwoArrayTylerEstimator(m)
    scm_est = SCMEstimator()

    ada_glrt_tyl = det.AdaptiveSonarDetector(glrt, tyl_est)
    ada_rao_tyl  = det.AdaptiveSonarDetector(rao,  tyl_est)
    ada_glrt_scm = det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P), scm_est)
    ada_rao_scm  = det.AdaptiveSonarDetector(det.MNMFRao(m, M, P),  scm_est)

    return {
        "M-NMF-G":       glrt,
        "M-NMF-R":       rao,
        "M-NMF-I":       indep,
        "SA-1":          sa1,
        "SA-2":          sa2,
        "MIMO-MF":       mmf,
        "M-ANMF-G-TYL":  ada_glrt_tyl,
        "M-ANMF-R-TYL":  ada_rao_tyl,
        "M-ANMF-G-SCM":  ada_glrt_scm,
        "M-ANMF-R-SCM":  ada_rao_scm,
    }


# ---------------------------------------------------------------------------
# Chunked trial runner
# ---------------------------------------------------------------------------

def _run_h0(n_trials, m, K, M, detectors, tau_shape, tau_scale, seed):
    stats = {name: [] for name in detectors}
    chunk = min(_CHUNK, n_trials)
    for i in range(0, n_trials, chunk):
        n = min(chunk, n_trials - i)
        x   = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale, seed=seed + i)
        xsec = sim.generate_secondary_data(n, K, m, M, seed=seed + 10000 + i)
        chunk_s = smc.run_detectors(x, detectors, X_secondary=xsec)
        for name, s in chunk_s.items():
            stats[name].append(s)
    return {name: np.concatenate(v) for name, v in stats.items()}


def _run_h1(n_trials, m, K, M, P, alpha, detectors, tau_shape, tau_scale, seed):
    stats = {name: [] for name in detectors}
    chunk = min(_CHUNK, n_trials)
    for i in range(0, n_trials, chunk):
        n = min(chunk, n_trials - i)
        x   = sim.generate_sonar_data_h1(n, m, M, P, alpha, tau_shape, tau_scale,
                                          seed=seed + i)
        xsec = sim.generate_secondary_data(n, K, m, M, seed=seed + 10000 + i)
        chunk_s = smc.run_detectors(x, detectors, X_secondary=xsec)
        for name, s in chunk_s.items():
            stats[name].append(s)
    return {name: np.concatenate(v) for name, v in stats.items()}


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
        help=f"Trials per memory chunk (default {_CHUNK}).")
    args = parser.parse_args()

    init_logging(args.backend)
    m = args.m
    K = smc.resolve_K(args)
    tau_shape, tau_scale = smc.clutter_params(args)

    logger.info(f"Sonar PD-vs-SNR: m={m}, K={K}, clutter={args.clutter} "
                f"(nu={args.nu}), n_h0={args.n_trials_h0}, n_h1={args.n_trials_h1}, "
                f"n_snr={args.n_snr}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)
    P = sim.make_steering_matrix(m, args.theta1, args.theta2)
    detectors = _build_detectors(m, M, P)

    snr_db, alphas = sim.snr_alpha_sweep(m, M, P, args.snr_min, args.snr_max, args.n_snr)

    t0 = time.perf_counter()

    logger.info("Running H0 trials...")
    stats_h0 = _run_h0(args.n_trials_h0, m, K, M, detectors,
                        tau_shape, tau_scale, seed=args.seed)

    logger.info("Running H1 trials across SNR grid...")
    pd_by_snr = {name: np.zeros(args.n_snr) for name in detectors}
    for i, (snr, alpha) in enumerate(zip(snr_db, alphas)):
        stats_h1 = _run_h1(args.n_trials_h1, m, K, M, P, alpha, detectors,
                            tau_shape, tau_scale, seed=args.seed + 1 + i * 1000)
        for name in detectors:
            pd_by_snr[name][i] = smc.aggregate_pd_snr(
                stats_h0[name], [stats_h1[name]], args.pfa
            )[0]
        if (i + 1) % 10 == 0 or i == args.n_snr - 1:
            logger.info(f"  SNR {i+1}/{args.n_snr}  ({snr:.1f} dB)")

    elapsed = time.perf_counter() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    detector_names = list(detectors.keys())
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
