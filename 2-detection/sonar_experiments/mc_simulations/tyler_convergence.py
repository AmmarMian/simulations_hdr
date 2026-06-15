#!/usr/bin/env python
"""2TYL fixed-point convergence: relative Frobenius deviation vs iteration.

Tracks ||M̂^(k) - M̂^(k-1)||_F / ||M̂^(k-1)||_F for each iteration of the
two-array Tyler MLE, averaged over n_trials secondary datasets.  Compares
behaviour under Gaussian and K-distributed clutter and for different K values.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from hdrlib.core.mc import MCResultExporter, init_logging, make_mc_parser
from hdrlib.sonar import mc as smc
from hdrlib.sonar import simulation as sim

logger = logging.getLogger(__name__)


def main():
    parser = make_mc_parser(__doc__)
    smc.add_mc_args(parser)
    smc.add_tyler_conv_args(parser)
    args = parser.parse_args()

    init_logging(args.backend)
    m = args.m
    K = smc.resolve_K(args)
    K_conv = args.K_conv if args.K_conv is not None else K
    tau_shape, tau_scale = smc.clutter_params(args)
    iter_max = args.iter_max

    logger.info(f"2TYL convergence: m={m}, K_conv={K_conv}, clutter={args.clutter}, "
                f"n_trials={args.n_trials}, iter_max={iter_max}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)

    t0 = time.perf_counter()

    # Generate secondary data (use moderate n_trials to keep memory manageable)
    X_sec = sim.generate_secondary_data(
        args.n_trials, K_conv, m, M, tau_shape, tau_scale, seed=args.seed
    )
    logger.info(f"Secondary data generated: {X_sec.shape}")

    deviations = smc.tyler_relative_deviations(X_sec, m, iter_max=iter_max, tol=0.0)

    elapsed = time.perf_counter() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    iterations = np.arange(1, iter_max + 1)

    export_stats = {
        "iterations":       iterations,
        "deviations_mean":  deviations,
        "K":                np.array([K_conv]),
        "m":                np.array([m]),
    }

    clutter_tag = args.clutter if args.clutter == "gaussian" else f"k_nu{args.nu}"
    stem_tag = f"m{m}_K{K_conv}_{clutter_tag}_n{args.n_trials}"

    exporter = MCResultExporter(args, Path(args.export_path), stem_tag,
                                plot_template=smc._MC_TYLER_CONV_TEMPLATE)
    title = (f"2TYL convergence (m={m}, K={K_conv}, {args.clutter} clutter)")
    exporter.save(export_stats, "sonar_tyler_conv", elapsed, title)

    # Report iterations to convergence at standard tolerances
    for tol in [1e-3, 1e-6]:
        below = np.where(deviations < tol)[0]
        if below.size:
            logger.info(f"  tol={tol:.0e}: converged at iteration {below[0]+1}")
        else:
            logger.info(f"  tol={tol:.0e}: did not converge within {iter_max} iters")


if __name__ == "__main__":
    main()
