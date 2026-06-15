#!/usr/bin/env python
"""MC PFA-vs-threshold curves for sonar two-array detectors.

Collects H0 test statistics from all detectors under Gaussian or K-distributed
clutter and plots empirical P(stat > eta) vs eta — verifying matrix-CFAR
behaviour.  Includes both known-M and adaptive (2TYL / SCM) variants.
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


def _build_detectors(m, M, P):
    glrt  = det.MNMFGlrt(m, M, P)
    rao   = det.MNMFRao(m, M, P)
    indep = det.MNMFIndependent(m, M, P)
    mmf   = det.MimoMatchedFilter(m, M, P)

    tyl_est = est.TwoArrayTylerEstimator(m)
    scm_est = SCMEstimator()

    return {
        "M-NMF-G":       glrt,
        "M-NMF-R":       rao,
        "M-NMF-I":       indep,
        "MIMO-MF":       mmf,
        "M-ANMF-G-TYL":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P), tyl_est),
        "M-ANMF-R-TYL":  det.AdaptiveSonarDetector(det.MNMFRao(m, M, P),  tyl_est),
        "M-ANMF-G-SCM":  det.AdaptiveSonarDetector(det.MNMFGlrt(m, M, P), scm_est),
    }


def main():
    parser = make_mc_parser(__doc__)
    smc.add_mc_args(parser)
    parser.add_argument("--n-thresh", type=int, default=500,
        help="Number of threshold grid points for PFA curve (default 500).")
    parser.add_argument("--chunk-size", type=int, default=_CHUNK,
        help=f"Trials per memory chunk (default {_CHUNK}).")
    args = parser.parse_args()

    init_logging(args.backend)
    m = args.m
    K = smc.resolve_K(args)
    tau_shape, tau_scale = smc.clutter_params(args)

    logger.info(f"Sonar PFA-vs-threshold: m={m}, K={K}, clutter={args.clutter}, "
                f"n_trials={args.n_trials}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)
    P = sim.make_steering_matrix(m, args.theta1, args.theta2)
    detectors = _build_detectors(m, M, P)

    t0 = time.perf_counter()

    all_stats = {name: [] for name in detectors}
    chunk = min(args.chunk_size, args.n_trials)
    for i in range(0, args.n_trials, chunk):
        n = min(chunk, args.n_trials - i)
        x    = sim.generate_sonar_data_h0(n, m, M, tau_shape, tau_scale,
                                           seed=args.seed + i)
        xsec = sim.generate_secondary_data(n, K, m, M, seed=args.seed + 10000 + i)
        cs = smc.run_detectors(x, detectors, X_secondary=xsec)
        for name, s in cs.items():
            all_stats[name].append(s)
        if (i // chunk + 1) % 5 == 0:
            logger.info(f"  {i + n}/{args.n_trials} trials done")

    elapsed = time.perf_counter() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    all_stats = {name: np.concatenate(v) for name, v in all_stats.items()}

    detector_names = list(detectors.keys())
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
