#!/usr/bin/env python
"""2TYL fixed-point convergence: relative Frobenius deviation vs iteration.

Tracks ||M̂^(k) - M̂^(k-1)||_F / ||M̂^(k-1)||_F for each iteration of the
two-array Tyler MLE, averaged over n_trials secondary datasets.  Compares
behaviour under Gaussian and K-distributed clutter and for different K values.

Backend selection:
  numpy     → single batched numpy pass (already parallelised across trials)
  all other → move secondary data to device, run Tyler iteration on device
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from hdrlib.core.backend import get_data_on_device
from hdrlib.core.mc import MCResultExporter, chunk_trial_ranges, make_mc_parser, timed_run
from hdrlib.sonar import mc as smc
from hdrlib.sonar import simulation as sim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numpy path (already batched across trials — no Pool needed)
# ---------------------------------------------------------------------------

def _run_numpy(n_trials, K_conv, m, M, tau_shape, tau_scale, iter_max, seed):
    X_sec = sim.generate_secondary_data(n_trials, K_conv, m, M, tau_shape, tau_scale, seed=seed)
    logger.info(f"Secondary data generated: {X_sec.shape}")
    return smc.tyler_relative_deviations(X_sec, m, iter_max=iter_max, backend_name="numpy")


# ---------------------------------------------------------------------------
# Batched path (non-numpy backends)
# ---------------------------------------------------------------------------

def _run_batched(n_trials, K_conv, m, M, tau_shape, tau_scale, iter_max, backend, seed):
    X_sec = sim.generate_secondary_data(n_trials, K_conv, m, M, tau_shape, tau_scale, seed=seed)
    logger.info(f"Secondary data generated: {X_sec.shape}; moving to {backend}...")
    X_dev = get_data_on_device(X_sec, backend)
    return np.asarray(
        smc.tyler_relative_deviations(X_dev, m, iter_max=iter_max, backend_name=backend)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    smc.add_mc_args(parser)
    smc.add_tyler_conv_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = args.m
    K = smc.resolve_K(args)
    K_conv = args.K_conv if args.K_conv is not None else K
    tau_shape, tau_scale = smc.clutter_params(args)
    iter_max = args.iter_max

    logger.info(f"2TYL convergence: m={m}, K_conv={K_conv}, clutter={args.clutter}, "
                f"n_trials={args.n_trials}, iter_max={iter_max}, backend={args.backend}")

    M = sim.make_sonar_covariance(m, args.beta, args.rho1, args.rho2)

    deviations, elapsed = timed_run(
        args,
        lambda: _run_numpy(args.n_trials, K_conv, m, M, tau_shape, tau_scale,
                           iter_max, args.seed),
        lambda: _run_batched(args.n_trials, K_conv, m, M, tau_shape, tau_scale,
                             iter_max, args.backend, args.seed),
    )

    logger.info(f"Done in {elapsed:.1f}s")

    iterations = np.arange(1, iter_max + 1)

    export_stats = {
        "iterations":      iterations,
        "deviations_mean": deviations,
        "K":               np.array([K_conv]),
        "m":               np.array([m]),
    }

    clutter_tag = args.clutter if args.clutter == "gaussian" else f"k_nu{args.nu}"
    stem_tag = f"m{m}_K{K_conv}_{clutter_tag}_n{args.n_trials}"

    exporter = MCResultExporter(args, Path(args.export_path), stem_tag,
                                plot_template=smc._MC_TYLER_CONV_TEMPLATE)
    title = (f"2TYL convergence (m={m}, K={K_conv}, {args.clutter} clutter)")
    exporter.save(export_stats, "sonar_tyler_conv", elapsed, title)

    for tol in [1e-3, 1e-6]:
        below = np.where(deviations < tol)[0]
        if below.size:
            logger.info(f"  tol={tol:.0e}: converged at iteration {below[0]+1}")
        else:
            logger.info(f"  tol={tol:.0e}: did not converge within {iter_max} iters")


if __name__ == "__main__":
    main()
