#!/usr/bin/env python
"""ROC curves for the four change detectors, offline and recursive.

Companion of mc_power_detectors.py, and its exact complement: the power figure
reads one point of this plane -- Pd at a fixed Pfa -- against T, this one reads
the whole plane at a few fixed T. Both rest on the same H0 and H1 statistics,
so the data generation, the detectors and the Monte-Carlo loop are imported
from mc_power_detectors rather than duplicated. The only thing that differs is
the reduction: aggregate_roc_multi instead of aggregate_power_multi.

Replaces the six per-detector experiments sar_mc_{dcg,gauss,kron}_{h0,h1},
which covered one detector pair at a time.

Four detectors, in the notation of the paper:
  SG     -- offline scale-and-shape GLRT, unstructured  (Lambda_SG)
  K-SG   -- offline GLRT with Kronecker structure       (Lambda_K-SG)
  SG-O   -- recursive counterpart of SG                 (Lambda_SG-O)
  K-SG-O -- recursive counterpart of K-SG               (Lambda_K-SG-O)

The Pfa axis stops at 10 / n_trials: below that a threshold rests on fewer than
ten H0 exceedances and the curve draws sampling noise. Reaching Pfa = 1e-3
therefore asks for --n-trials 10000.

Backend selection:
  numpy     -> multiprocessing.Pool, one trial per worker
  all other -> trials in leading batch dim, single-pass on device
"""

from __future__ import annotations

import logging
from pathlib import Path

from hdrlib.core.mc import MCResultExporter, init_logging, make_mc_parser, timed_run
from hdrlib.sar.simulation import (
    make_ab_toeplitz,
    generate_kronecker_data,
    generate_kronecker_data_h1,
)
from hdrlib.sar.mc import (
    _MC_PLOT_TEMPLATE_ROC_MULTI,
    add_mc_args,
    add_mc_h1_args,
    finish_roc_multi,
)

# Same machinery as the power experiment. Imported, not copied: the two figures
# must not be able to drift apart on the data or the detectors.
from mc_power_detectors import _run_pool, _run_batched

logger = logging.getLogger(__name__)


def main():
    parser = make_mc_parser(__doc__)
    add_mc_args(parser)
    add_mc_h1_args(parser)
    parser.set_defaults(n_trials=10000)

    parser.add_argument("--a", type=int, default=3, help="Size of the first Kronecker factor.")
    parser.add_argument("--b", type=int, default=4, help="Size of the second Kronecker factor.")
    parser.add_argument("--T-list", type=str, default="5,12,49",
        help="Comma-separated T values at which the ROC is read (default '5,12,49').")
    parser.add_argument("--n-samples", type=int, default=None,
        help="Samples per date (default: p+1 = a*b+1 = 13).")
    parser.add_argument("--texture", type=str, default="k", choices=["k", "gaussian"],
        help="'k' for K-distributed data with shape --nu (default), 'gaussian' for tau = 1.")
    parser.add_argument("--nu", type=float, default=1.0,
        help="Shape of the K-distribution texture when --texture k (default 1.0).")
    parser.add_argument("--rho-a0", type=str, default="0.3+0.7j", help="Toeplitz coefficient of A under H0.")
    parser.add_argument("--rho-b0", type=str, default="0.3+0.6j", help="Toeplitz coefficient of B under H0.")
    parser.add_argument("--rho-a1", type=str, default="0.3+0.5j", help="Toeplitz coefficient of A after the change.")
    parser.add_argument("--rho-b1", type=str, default="0.4+0.5j", help="Toeplitz coefficient of B after the change.")
    parser.add_argument("--step-rule", type=str, default="fixed", choices=["fixed", "armijo"],
        help="Step of the recursive estimators (default 'fixed', the schedule of eq. 19).")
    parser.add_argument("--init-mode", type=str, default="mm", choices=["mm", "identity"],
        help="Initialisation of the recursive Kronecker estimator (default 'mm').")
    parser.add_argument("--alpha-0", type=float, default=1.0, help="Initial step (default 1.0).")
    parser.add_argument("--iter-max", type=int, default=30, help="Max fixed-point / MM iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance.")
    parser.add_argument("--with-gaussian", action="store_true",
        help="Add the Gaussian covariance equality GLRT as a fifth baseline.")
    parser.add_argument("--detectors", type=str, default=None,
        help="Comma-separated subset of detectors to run (e.g. 'SG-O'). Default: all four.")
    parser.add_argument("--debug", action="store_true",
        help="Tiny configuration (60 trials, T up to 10) to validate the pipeline "
             "in seconds. Results are NOT publication grade.")
    args = parser.parse_args()

    init_logging(args)
    if args.debug:
        args.n_trials, args.T_list = 60, "3,10"
        logger.warning("--debug: 60 trials, T in {3, 10}. Pipeline check only.")

    a, b = args.a, args.b
    p = a * b
    n_samples = args.n_samples if args.n_samples is not None else p + 1
    T_vec = [int(t) for t in args.T_list.split(",")]
    T_max = max(T_vec)
    tau_shape = None if args.texture == "gaussian" else args.nu
    tau_scale = 1.0 if tau_shape is None else 1.0 / args.nu

    cfg = {
        "detectors": ([d.strip() for d in args.detectors.split(",")]
                      if args.detectors else None),
        "step_rule": args.step_rule,
        "init_mode": args.init_mode,
        "alpha_0": args.alpha_0,
        "iter_max": args.iter_max,
        "tol": args.tol,
        "with_gaussian": args.with_gaussian,
    }

    logger.info(f"ROC: a={a}, b={b}, p={p}, n_samples={n_samples}, texture={args.texture}, "
                f"n_trials={args.n_trials}, T={T_vec}, backend={args.backend}")
    logger.info(f"Smallest resolvable Pfa: {10 / args.n_trials:.1e}")

    A0, B0 = make_ab_toeplitz(a, b, complex(args.rho_a0), complex(args.rho_b0))
    A1, B1 = make_ab_toeplitz(a, b, complex(args.rho_a1), complex(args.rho_b1))

    logger.info("Generating H0 data...")
    data_h0 = generate_kronecker_data(
        args.n_trials, T_max, n_samples, a, b, A0, B0,
        seed=args.seed, tau_shape=tau_shape, tau_scale=tau_scale)

    logger.info("Generating H1 data (one series per T, change at T/2)...")
    h1_data = {}
    for T in T_vec:
        n_change = max(1, int(T * args.change_fraction))
        h1_data[T] = generate_kronecker_data_h1(
            args.n_trials, T, n_samples, a, b, A0, B0, A1, B1,
            seed=args.seed + 1000 + T, n_change_dates=n_change,
            tau_shape=tau_shape, tau_scale=tau_scale)

    exporter = MCResultExporter(
        args, Path(args.export_path), f"{args.texture}_a{a}_b{b}_T{T_max}_n{args.n_trials}",
        plot_template=_MC_PLOT_TEMPLATE_ROC_MULTI,
    )

    (h0, h1), elapsed = timed_run(
        args,
        lambda: _run_pool(data_h0, h1_data, T_vec, args.n_workers, a, b, cfg),
        lambda: _run_batched(data_h0, h1_data, T_vec, args.backend, a, b, cfg),
    )

    regime = "gaussien" if args.texture == "gaussian" else f"K, nu={args.nu}"
    title = f"Courbes ROC ({regime}, a={a}, b={b}, n={n_samples}, {args.n_trials} tirages)"
    finish_roc_multi(args, exporter, h0, h1, T_vec, "mc_roc", title, elapsed)


if __name__ == "__main__":
    main()
