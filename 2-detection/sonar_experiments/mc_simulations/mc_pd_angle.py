#!/usr/bin/env python
"""MC PD vs (theta1, theta2) angle map for sonar two-array detectors.

Sweeps over a 2-D grid of steering angles at a fixed SNR (--snr-db) and
fixed nominal PFA.  Threshold is calibrated from H0 trials with the
nominal steering direction.  Produces one PD heatmap per detector.

The angle grid is symmetric and covers both arrays simultaneously.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from hdrlib.core.estimation import SCMEstimator
from hdrlib.core.mc import MCResultExporter, init_logging, make_mc_parser
from hdrlib.sonar import detectors as det
from hdrlib.sonar import estimation as est
from hdrlib.sonar import mc as smc
from hdrlib.sonar import simulation as sim

logger = logging.getLogger(__name__)

_CHUNK = 200


def _build_detectors(m, M, P_nominal):
    glrt  = det.MNMFGlrt(m, M, P_nominal)
    rao   = det.MNMFRao(m, M, P_nominal)
    indep = det.MNMFIndependent(m, M, P_nominal)

    tyl_est = est.TwoArrayTylerEstimator(m)
    scm_est = SCMEstimator()

    return {
        "M-NMF-G":       glrt,
        "M-NMF-R":       rao,
        "M-NMF-I":       indep,
        "M-ANMF-G-TYL":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P_nominal), tyl_est),
        "M-ANMF-G-SCM":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P_nominal), scm_est),
    }


def _h0_thresholds(n_trials, m, K, M, detectors, tau_shape, tau_scale, pfa, seed):
    all_stats = {name: [] for name in detectors}
    chunk = min(_CHUNK, n_trials)
    for i in range(0, n_trials, chunk):
        n = min(chunk, n_trials - i)
        x    = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale, seed=seed + i)
        xsec = sim.generate_secondary_data(n, K, m, M, seed=seed + 10000 + i)
        cs   = smc.run_detectors(x, detectors, X_secondary=xsec)
        for name, s in cs.items():
            all_stats[name].append(s)
    stats_h0 = {name: np.concatenate(v) for name, v in all_stats.items()}
    return {name: smc.threshold_at_pfa(s, pfa) for name, s in stats_h0.items()}


def main():
    parser = make_mc_parser(__doc__)
    smc.add_mc_args(parser)
    smc.add_angle_args(parser)
    parser.add_argument("--n-trials-h0", type=int, default=1000,
        help="Trials for H0 threshold calibration (default 1000).")
    parser.add_argument("--chunk-size", type=int, default=_CHUNK,
        help=f"Trials per memory chunk (default {_CHUNK}).")
    args = parser.parse_args()

    init_logging(args.backend)
    m = args.m
    K = smc.resolve_K(args)
    tau_shape, tau_scale = smc.clutter_params(args)

    logger.info(f"Sonar PD-vs-angle: m={m}, K={K}, clutter={args.clutter}, "
                f"snr={args.snr_db} dB, n_angles={args.n_theta}x{args.n_theta}, "
                f"n_trials={args.n_trials}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)
    # Nominal steering (used for detector construction and H0 threshold)
    P_nominal = sim.make_steering_matrix(m, args.theta1, args.theta2)
    detectors = _build_detectors(m, M, P_nominal)

    # Calibrate thresholds from H0
    logger.info("Calibrating thresholds from H0...")
    thresholds = _h0_thresholds(
        args.n_trials_h0, m, K, M, detectors, tau_shape, tau_scale,
        args.pfa, seed=args.seed,
    )
    logger.info(f"Thresholds: { {k: f'{v:.4f}' for k, v in thresholds.items()} }")

    # Build angle grid
    theta_grid = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    n_ang = len(theta_grid)

    # Determine signal amplitude for fixed SNR
    # Use nominal P for SNR normalisation (independent of the scan grid)
    snr_db_arr, alphas = sim.snr_alpha_sweep(
        m, M, P_nominal,
        snr_min_db=args.snr_db, snr_max_db=args.snr_db, n_snr=1,
    )
    alpha_fixed = float(alphas[0])
    logger.info(f"Alpha for SNR={args.snr_db} dB: {alpha_fixed:.6f}")

    t0 = time.perf_counter()

    pd_maps = {name: np.zeros((n_ang, n_ang)) for name in detectors}

    for i, th1 in enumerate(theta_grid):
        for j, th2 in enumerate(theta_grid):
            P_ij = sim.make_steering_matrix(m, float(th1), float(th2))
            # Generate H1 trials with true signal in direction (th1, th2)
            h1_stats = {name: [] for name in detectors}
            chunk = min(args.chunk_size, args.n_trials)
            for ci in range(0, args.n_trials, chunk):
                n = min(chunk, args.n_trials - ci)
                x    = sim.generate_sonar_data_h1(n, m, M, P_ij, alpha_fixed,
                                                   tau_shape, tau_scale,
                                                   seed=args.seed + ci + i * 100000 + j * 1000)
                xsec = sim.generate_secondary_data(n, K, m, M,
                                                   seed=args.seed + 10000 + ci + i * 100000 + j * 1000)
                cs = smc.run_detectors(x, detectors, X_secondary=xsec)
                for name, s in cs.items():
                    h1_stats[name].append(s)

            for name in detectors:
                s_h1 = np.concatenate(h1_stats[name])
                pd_maps[name][i, j] = smc.empirical_pd(s_h1, thresholds[name])

        if (i + 1) % 5 == 0 or i == n_ang - 1:
            logger.info(f"  Angle row {i+1}/{n_ang}  (theta1={th1:.1f} deg)")

    elapsed = time.perf_counter() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    detector_names = list(detectors.keys())
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
