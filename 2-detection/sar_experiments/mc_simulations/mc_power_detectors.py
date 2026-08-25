#!/usr/bin/env python
"""Power against T for the four change detectors, offline and online.

Replaces the three ROC planes of Mian et al., Signal Processing 224 (2024)
(Figures 4 to 6) by the single quantity the chapter argues about: how fast the
online detectors catch up with their offline counterparts as the time series
grows, at a fixed false alarm probability.

Four detectors, in the notation of the paper:
  SG     -- offline scale-and-shape GLRT, unstructured  (Lambda_SG)
  K-SG   -- offline GLRT with Kronecker structure       (Lambda_K-SG)
  SG-O   -- recursive counterpart of SG                 (Lambda_SG-O)
  K-SG-O -- recursive counterpart of K-SG               (Lambda_K-SG-O)
and optionally G, the Gaussian covariance equality GLRT, as a baseline.

Data follow the same setup as the ROC configuration of the released code:
Toeplitz factors of unit determinant, rho changing at T/2 under H1, patches of
n = a*b+1 samples. Run once with --texture k (K-distributed, nu=1) and once
with --texture gaussian to expose the result the paper found surprising: the
online detectors converge more slowly in the Gaussian case than in the
heterogeneous one.

Thresholds are set per detector and per T at the (1 - PFA) quantile of the H0
statistics, so every curve is read at the same false alarm rate.

Backend selection:
  numpy     → multiprocessing.Pool, one trial per worker
  all other → trials in leading batch dim, single-pass on device
"""

from __future__ import annotations

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.simulation import T_vec_logspace
from hdrlib.core.mc import (
    MCResultExporter,
    init_logging,
    make_mc_parser,
    maybe_empty_cache,
    timed_run,
)
from hdrlib.sar.simulation import (
    make_ab_toeplitz,
    generate_kronecker_data,
    generate_kronecker_data_h1,
)
from hdrlib.sar.detectors import (
    DeterministicCompoundGaussianGLRT,
    GaussianGLRT,
    ScaleAndShapeKroneckerGLRT,
)
from hdrlib.sar.detection_online import OnlineDCGDetector, OnlineKroneckerDetector
from hdrlib.sar.mc import (
    _MC_PLOT_TEMPLATE_POWER_MULTI,
    add_mc_args,
    add_mc_h1_args,
    finish_power_multi,
    online_single_pass,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------

def _build_detectors(a, b, cfg, backend):
    """Return {name: (kind, detector)} with kind in {"offline", "online"}."""
    dets = {
        "SG": ("offline", DeterministicCompoundGaussianGLRT(
            backend_name=backend, tol=cfg["tol"], iter_max=cfg["iter_max"])),
        "K-SG": ("offline", ScaleAndShapeKroneckerGLRT(
            a, b, backend, tol=cfg["tol"], iter_max=cfg["iter_max"])),
        "SG-O": ("online", OnlineDCGDetector(
            backend_name=backend, h0_step_rule=cfg["step_rule"],
            h0_alpha_0=cfg["alpha_0"], iter_max=cfg["iter_max"], tol=cfg["tol"])),
        "K-SG-O": ("online", OnlineKroneckerDetector(
            a, b, backend, h0_step_rule=cfg["step_rule"], h0_init_mode=cfg["init_mode"],
            h0_alpha_0=cfg["alpha_0"], iter_max=cfg["iter_max"], tol=cfg["tol"])),
    }
    if cfg["with_gaussian"]:
        dets["G"] = ("offline", GaussianGLRT(backend))
    keep = cfg.get("detectors")
    if keep:
        missing = set(keep) - set(dets)
        if missing:
            raise ValueError(f"unknown detector(s) {sorted(missing)}; have {sorted(dets)}")
        dets = {k: v for k, v in dets.items() if k in keep}
    return dets


def _statistics(data, T_vec, a, b, cfg, backend):
    """Statistics of every detector at every T checkpoint, for one data set.

    data : (..., T_max, N, p). Offline detectors are evaluated on the first T
    dates; online ones are streamed once and read at each checkpoint.
    """
    dets = _build_detectors(a, b, cfg, backend)
    out = {}
    for name, (kind, det) in dets.items():
        if kind == "online":
            raw = online_single_pass(data, T_vec, det)
            out[name] = {T: to_numpy(v) for T, v in raw.items()}
        else:
            out[name] = {T: to_numpy(det.compute(data[..., :T, :, :])) for T in T_vec}
        maybe_empty_cache(backend)
    return out


# ---------------------------------------------------------------------------
# Pool worker (numpy)
# ---------------------------------------------------------------------------

def _worker(worker_args):
    data_h0, h1_per_T, T_vec, a, b, cfg = worker_args
    h0 = _statistics(data_h0, T_vec, a, b, cfg, "numpy")
    # Under H1 the change sits at T/2, so a fresh series is needed for each T.
    h1 = {}
    for T in T_vec:
        stats_T = _statistics(h1_per_T[T], [T], a, b, cfg, "numpy")
        for name, d in stats_T.items():
            h1.setdefault(name, {})[T] = d[T]
    return h0, h1


def _run_pool(data_h0, h1_data, T_vec, n_workers, a, b, cfg):
    n_trials = data_h0.shape[0]
    worker_args = [
        (data_h0[i], {T: h1_data[T][i] for T in T_vec}, T_vec, a, b, cfg)
        for i in range(n_trials)
    ]
    all_h0, all_h1 = [], []
    logger.info(f"Starting power computation: {n_trials} trials via Pool, {len(T_vec)} T-points...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} trials"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]MC trials (Pool)...", total=n_trials)
        with Pool(processes=n_workers) as pool:
            for h0, h1 in pool.imap_unordered(_worker, worker_args):
                all_h0.append(h0)
                all_h1.append(h1)
                progress.advance(task)

    def _stack(dicts):
        names = dicts[0].keys()
        return {name: {T: np.array([np.squeeze(d[name][T]) for d in dicts]) for T in T_vec}
                for name in names}

    return _stack(all_h0), _stack(all_h1)


# ---------------------------------------------------------------------------
# Batched path
# ---------------------------------------------------------------------------

def _run_batched(data_h0, h1_data, T_vec, backend, a, b, cfg):
    logger.info(f"Starting batched power computation on {backend}...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]H0 + H1 per T...", total=1 + len(T_vec))
        h0 = _statistics(get_data_on_device(data_h0, backend), T_vec, a, b, cfg, backend)
        progress.advance(task)
        h1 = {}
        for T in T_vec:
            stats_T = _statistics(get_data_on_device(h1_data[T], backend), [T], a, b, cfg, backend)
            for name, d in stats_T.items():
                h1.setdefault(name, {})[T] = d[T]
            progress.advance(task)
    return h0, h1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    add_mc_args(parser)
    add_mc_h1_args(parser)
    parser.set_defaults(T_max=50, T_min=2, n_T=8, n_trials=5000, pfa=1e-2)

    parser.add_argument("--a", type=int, default=3, help="Size of the first Kronecker factor.")
    parser.add_argument("--b", type=int, default=4, help="Size of the second Kronecker factor.")
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
        help="Comma-separated subset of detectors to run (e.g. 'SG-O'). Default: all "
             "four. Use it to iterate on one curve without paying for the others; the "
             "thresholds are per detector anyway, so a subset gives the same numbers.")
    parser.add_argument("--debug", action="store_true",
        help="Tiny configuration (60 trials, T up to 10, PFA 0.1) to validate the "
             "pipeline in seconds. Results are NOT publication grade.")
    args = parser.parse_args()

    if args.debug:
        args.n_trials, args.T_max, args.n_T, args.T_min, args.pfa = 60, 10, 3, 2, 0.1

    init_logging(args.backend)
    if args.debug:
        logger.warning("--debug: 60 trials, T_max=10, PFA=0.1. Pipeline check only.")
    else:
        min_trials = int(10 / args.pfa)
        if args.n_trials < min_trials:
            logger.warning(
                f"n_trials={args.n_trials} < 10/PFA={min_trials}: the threshold at "
                f"PFA={args.pfa} will be poorly estimated. Consider --n-trials {min_trials}.")

    a, b = args.a, args.b
    p = a * b
    n_samples = args.n_samples if args.n_samples is not None else p + 1
    T_vec = T_vec_logspace(args.T_min, args.T_max, args.n_T)
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

    logger.info(f"Power vs T: a={a}, b={b}, p={p}, n_samples={n_samples}, texture={args.texture}, "
                f"n_trials={args.n_trials}, T=[{T_vec[0]}..{T_vec[-1]}] ({len(T_vec)} pts), "
                f"PFA={args.pfa}, backend={args.backend}")

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
        plot_template=_MC_PLOT_TEMPLATE_POWER_MULTI,
    )

    (h0, h1), elapsed = timed_run(
        args,
        lambda: _run_pool(data_h0, h1_data, T_vec, args.n_workers, a, b, cfg),
        lambda: _run_batched(data_h0, h1_data, T_vec, args.backend, a, b, cfg),
    )

    regime = "gaussien" if args.texture == "gaussian" else f"K, nu={args.nu}"
    title = f"Puissance à $P_{{fa}}$={args.pfa:g}  ({regime}, a={a}, b={b}, n={n_samples})"
    finish_power_multi(args, exporter, h0, h1, T_vec, "mc_power", title, elapsed)


if __name__ == "__main__":
    main()
