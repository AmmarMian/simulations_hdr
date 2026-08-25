#!/usr/bin/env python
"""MSE vs T of the Kronecker scaled-Gaussian estimators, against the ICRB.

Reproduces Figures 2 and 3 of Mian et al., Signal Processing 224 (2024), with
the setup of the released configuration rather than of the body text: Toeplitz
factors of unit determinant, a=3, b=4, n=a*b+1=13, K-distributed data (texture
Gamma(nu, 1/nu), nu=1). The body text of Section 5.1 announces a=4, b=3, n=8
and random factors of condition number 10; the published figure captions and
the released code both use the setup implemented here.

Measured quantities, per component of theta = (A, B, tau): the squared
geodesic distances of equation (18), averaged over trials, for

  * the offline MLE (MM by default, Riemannian gradient descent with --offline gd),
  * the recursive estimator of equation (19), one gradient step per new date,

together with the intrinsic Cramer-Rao bounds of equation (25).

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
from hdrlib.sar.simulation import make_ab_toeplitz, generate_kronecker_data
from hdrlib.sar.estimation_kronecker import kronecker_mm_h0, kronecker_riemannian_gd_h0
from hdrlib.sar.estimation_online import OnlineKroneckerEstimator
from hdrlib.sar.icrb import icrb_kronecker_scaled_gaussian, kronecker_component_errors
from hdrlib.sar.mc import _MC_PLOT_TEMPLATE_MSE, add_mc_args, finish_mse

logger = logging.getLogger(__name__)

_COMPONENTS = ("A", "B", "tau", "total")


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def _offline_estimate(X, a, b, cfg, backend):
    """Offline MLE on X of shape (..., T, N, p)."""
    if cfg["offline"] == "gd":
        return kronecker_riemannian_gd_h0(
            X, a, b, iter_max=cfg["gd_iter_max"], tol=cfg["gd_tol"], backend_name=backend)
    A, B, tau = kronecker_mm_h0(
        X, a, b, tol=cfg["mm_tol"], iter_max=cfg["mm_iter_max"], backend_name=backend)
    return A, B, tau[..., None] if tau.ndim == X.ndim - 2 else tau


def _errors(A, B, tau, truth, a, b, n_samples, backend):
    A_true, B_true, tau_true = truth
    return kronecker_component_errors(
        A, B, tau, A_true, B_true, tau_true, a, b, n_samples, backend_name=backend)


def _online_checkpoints(X, T_vec, truth, a, b, n_samples, cfg, backend):
    """Stream X (..., T_max, N, p) through the recursive estimator.

    Returns {component: {T: array}} evaluated at each T in T_vec.
    """
    T_set = set(T_vec)
    T_max = X.shape[-3]
    est = OnlineKroneckerEstimator(
        a, b, n_samples,
        step_rule=cfg["step_rule"], init_mode=cfg["init_mode"], alpha_0=cfg["alpha_0"],
        iter_max=cfg["mm_iter_max"], tol=cfg["mm_tol"], backend_name=backend,
    )
    est.reset()
    out = {c: {} for c in _COMPONENTS}
    for t in range(T_max):
        A, B, tau = est.update(X[..., t, :, :])
        T_current = t + 1
        if T_current in T_set:
            err = _errors(A, B, tau, truth, a, b, n_samples, backend)
            for c in _COMPONENTS:
                out[c][T_current] = to_numpy(err[c])
    return out


# ---------------------------------------------------------------------------
# Pool worker (numpy only — one trial per worker process)
# ---------------------------------------------------------------------------

def _worker(worker_args):
    X, tau_true, T_vec, a, b, n_samples, A_true, B_true, cfg = worker_args
    truth = (A_true, B_true, tau_true)
    online = _online_checkpoints(X, T_vec, truth, a, b, n_samples, cfg, "numpy")

    offline = {c: {} for c in _COMPONENTS}
    for T in T_vec:
        A, B, tau = _offline_estimate(X[None, :T], a, b, cfg, "numpy")
        err = _errors(A[0], B[0], tau[0], truth, a, b, n_samples, "numpy")
        for c in _COMPONENTS:
            offline[c][T] = float(np.squeeze(err[c]))
    return online, offline


def _run_pool(data, tau_true, T_vec, n_workers, a, b, n_samples, A_true, B_true, cfg):
    n_trials = data.shape[0]
    worker_args = [
        (data[i], tau_true[i], T_vec, a, b, n_samples, A_true, B_true, cfg)
        for i in range(n_trials)
    ]
    all_online, all_offline = [], []
    logger.info(f"Starting MSE computation: {n_trials} trials via Pool, {len(T_vec)} T-checkpoints...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} trials"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]MC trials (Pool)...", total=n_trials)
        with Pool(processes=n_workers) as pool:
            for on, off in pool.imap_unordered(_worker, worker_args):
                all_online.append(on)
                all_offline.append(off)
                progress.advance(task)

    def _stack(dicts):
        return {c: {T: np.array([np.squeeze(d[c][T]) for d in dicts]) for T in T_vec}
                for c in _COMPONENTS}

    return _stack(all_online), _stack(all_offline)


# ---------------------------------------------------------------------------
# Batched path (non-numpy backends)
# ---------------------------------------------------------------------------

def _run_batched(data, tau_true, T_vec, backend, a, b, n_samples, A_true, B_true, cfg):
    X = get_data_on_device(data, backend)
    A_t = get_data_on_device(A_true, backend)
    B_t = get_data_on_device(B_true, backend)
    tau_t = get_data_on_device(tau_true, backend)
    truth = (A_t, B_t, tau_t)

    logger.info(f"Starting batched MSE computation on {backend} ({len(T_vec)} T-points)...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task_on = progress.add_task("[cyan]Online single-pass...", total=1)
        online = _online_checkpoints(X, T_vec, truth, a, b, n_samples, cfg, backend)
        progress.advance(task_on)

        task_off = progress.add_task(f"[green]Offline ({len(T_vec)} points)...", total=len(T_vec))
        offline = {c: {} for c in _COMPONENTS}
        for T in T_vec:
            A, B, tau = _offline_estimate(X[..., :T, :, :], a, b, cfg, backend)
            err = _errors(A, B, tau, truth, a, b, n_samples, backend)
            for c in _COMPONENTS:
                offline[c][T] = to_numpy(err[c])
            maybe_empty_cache(backend)
            progress.advance(task_off)

    return online, offline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    add_mc_args(parser)
    parser.set_defaults(T_max=1000, T_min=2, n_T=12, n_trials=1000)

    parser.add_argument("--a", type=int, default=3,
        help="Size of first Kronecker factor (default 3, as in the released config).")
    parser.add_argument("--b", type=int, default=4,
        help="Size of second Kronecker factor (default 4).")
    parser.add_argument("--n-samples", type=int, default=None,
        help="Samples per date (default: p+1 = a*b+1 = 13).")
    parser.add_argument("--nu", type=float, default=1.0,
        help="Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0). "
             "Large nu approaches the Gaussian case.")
    parser.add_argument("--rho-a", type=str, default="0.3+0.7j",
        help="Toeplitz coefficient of A, as a complex literal (default 0.3+0.7j).")
    parser.add_argument("--rho-b", type=str, default="0.3+0.6j",
        help="Toeplitz coefficient of B, as a complex literal (default 0.3+0.6j).")
    parser.add_argument("--offline", type=str, default="mm", choices=["mm", "gd"],
        help="Offline reference: 'mm' majorisation-minimisation (fast, default) or "
             "'gd' Riemannian gradient descent (the reference of the paper). "
             "Both target the same MLE and agree numerically.")
    parser.add_argument("--step-rule", type=str, default="fixed", choices=["fixed", "armijo"],
        help="Step of the recursive estimator: 'fixed' is alpha_0/t, the schedule of "
             "equation (19) (default); 'armijo' is a line search at each update.")
    parser.add_argument("--init-mode", type=str, default="mm", choices=["mm", "identity"],
        help="Initialisation of the recursive estimator: 'mm' warm-starts on the first "
             "date (default), 'identity' starts from (I, I, 1) as the released code does.")
    parser.add_argument("--alpha-0", type=float, default=None,
        help="Initial step (default: 1.0 for 'fixed', 0.1 for 'armijo').")
    parser.add_argument("--mm-iter-max", type=int, default=50,
        help="Max MM iterations (default 50).")
    parser.add_argument("--mm-tol", type=float, default=1e-8,
        help="MM convergence tolerance (default 1e-8).")
    parser.add_argument("--gd-iter-max", type=int, default=200,
        help="Max iterations of the offline Riemannian gradient descent (default 200).")
    parser.add_argument("--gd-tol", type=float, default=1e-8,
        help="Tolerance of the offline Riemannian gradient descent (default 1e-8).")
    parser.add_argument("--debug", action="store_true",
        help="Tiny configuration (8 trials, T up to 50) to validate the pipeline in "
             "seconds. Results are NOT publication grade.")
    args = parser.parse_args()

    if args.debug:
        args.n_trials, args.T_max, args.n_T, args.T_min = 8, 50, 5, 2
        args.mm_iter_max, args.gd_iter_max = 20, 50

    init_logging(args.backend)
    if args.debug:
        logger.warning("--debug: 8 trials, T_max=50. Pipeline check only, not a result.")

    a, b = args.a, args.b
    p = a * b
    n_samples = args.n_samples if args.n_samples is not None else p + 1
    T_vec = T_vec_logspace(args.T_min, args.T_max, args.n_T)
    T_max = max(T_vec)

    cfg = {
        "offline": args.offline,
        "step_rule": args.step_rule,
        "init_mode": args.init_mode,
        "alpha_0": args.alpha_0,
        "mm_iter_max": args.mm_iter_max,
        "mm_tol": args.mm_tol,
        "gd_iter_max": args.gd_iter_max,
        "gd_tol": args.gd_tol,
    }

    logger.info(f"Kronecker MSE/ICRB: a={a}, b={b}, p={p}, n_samples={n_samples}, "
                f"n_trials={args.n_trials}, T=[{T_vec[0]}..{T_vec[-1]}] ({len(T_vec)} pts), "
                f"backend={args.backend}")
    logger.info(f"  texture ~ Gamma({args.nu}, {1 / args.nu:.3g}) | offline={args.offline} | "
                f"online step={args.step_rule}, init={args.init_mode}")

    A_true, B_true = make_ab_toeplitz(a, b, complex(args.rho_a), complex(args.rho_b))
    logger.info(f"Generating Kronecker data ({args.n_trials}, {T_max}, {n_samples}, {p}) complex128...")
    data, tau_true = generate_kronecker_data(
        args.n_trials, T_max, n_samples, a, b, A_true, B_true,
        seed=args.seed, tau_shape=args.nu, tau_scale=1.0 / args.nu, return_tau=True,
    )

    exporter = MCResultExporter(
        args, Path(args.export_path), f"a{a}_b{b}_T{T_max}_n{args.n_trials}",
        plot_template=_MC_PLOT_TEMPLATE_MSE,
    )

    (online_err, offline_err), elapsed = timed_run(
        args,
        lambda: _run_pool(data, tau_true, T_vec, args.n_workers, a, b, n_samples, A_true, B_true, cfg),
        lambda: _run_batched(data, tau_true, T_vec, args.backend, a, b, n_samples, A_true, B_true, cfg),
    )

    icrb = icrb_kronecker_scaled_gaussian(a, b, n_samples, np.asarray(T_vec))
    title = (f"Estimation Kronecker gaussienne à échelle  (a={a}, b={b}, n={n_samples}, "
             f"nu={args.nu})")
    finish_mse(args, exporter, online_err, offline_err, icrb, T_vec, "mc_kron_mse", title, elapsed)


if __name__ == "__main__":
    main()
