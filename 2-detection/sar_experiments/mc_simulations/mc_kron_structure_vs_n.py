#!/usr/bin/env python
"""What the Kronecker structure buys: estimation error against the patch size N.

At a fixed number of dates T, the same H0 data is fitted twice with the same
geometry, the same gradient and the same line search, once with the shape
matrix constrained to A (x) B and once with it free in sH++(p).  The error
reported is the total squared Riemannian distance d^2_M of equation (18),
averaged over trials, together with the intrinsic Cramer-Rao bound of each
parametrisation.

The gap between the two bounds is a ratio of dimensions,
((a^2-1) + (b^2-1) + N) / ((p^2-1) + N): wide when the patch is small, closing
as N grows.  This is the same trade -- constrain the model to lower the sample
support needed -- that the multi-ping sonar section makes with Kronecker plus
Toeplitz, measured here on the change detection model.

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
from hdrlib.core.estimation import scaled_gaussian_riemannian_gd_h0
from hdrlib.core.mc import (
    MCResultExporter,
    init_logging,
    make_mc_parser,
    maybe_empty_cache,
    timed_run,
)
from hdrlib.sar.simulation import make_ab_toeplitz, generate_kronecker_data
from hdrlib.sar.estimation_kronecker import kronecker_mm_h0, kronecker_riemannian_gd_h0
from hdrlib.sar.icrb import (
    icrb_kronecker_scaled_gaussian,
    icrb_scaled_gaussian,
    kronecker_component_errors,
    scaled_gaussian_component_errors,
)
from hdrlib.sar.mc import _MC_PLOT_TEMPLATE_STRUCT, add_mc_base_args, finish_struct

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One (N, trial batch) evaluation
# ---------------------------------------------------------------------------

def _fit_both(X, a, b, N, truth, cfg, backend):
    """Fit structured and unstructured models on X of shape (..., T, N, p)."""
    A_true, B_true, tau_true = truth
    p = a * b

    if cfg["offline"] == "gd":
        A, B, tau_k = kronecker_riemannian_gd_h0(
            X, a, b, iter_max=cfg["gd_iter_max"], tol=cfg["gd_tol"], backend_name=backend)
    else:
        A, B, tau_flat = kronecker_mm_h0(
            X, a, b, tol=cfg["mm_tol"], iter_max=cfg["mm_iter_max"], backend_name=backend)
        tau_k = tau_flat[..., None] if tau_flat.ndim == X.ndim - 2 else tau_flat
    err_k = kronecker_component_errors(
        A, B, tau_k, A_true, B_true, tau_true, a, b, N, backend_name=backend)

    Sigma, tau_f = scaled_gaussian_riemannian_gd_h0(
        X, iter_max=cfg["gd_iter_max"], tol=cfg["gd_tol"], backend_name=backend)
    Sigma_true = np.kron(A_true, B_true)
    err_f = scaled_gaussian_component_errors(
        Sigma, tau_f, get_data_on_device(Sigma_true, backend), tau_true, p, N,
        backend_name=backend)

    return err_k["total"], err_f["total"]


def _worker(worker_args):
    X, tau_true, a, b, N, A_true, B_true, cfg = worker_args
    e_k, e_f = _fit_both(X[None], a, b, N, (A_true, B_true, tau_true), cfg, "numpy")
    return float(np.squeeze(to_numpy(e_k))), float(np.squeeze(to_numpy(e_f)))


def _run_pool(datasets, N_vec, n_workers, a, b, A_true, B_true, cfg):
    kron_err, full_err = {}, {}
    logger.info(f"Starting structured/unstructured comparison over {len(N_vec)} values of N (Pool)...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]N values...", total=len(N_vec))
        for N in N_vec:
            data, tau_true = datasets[N]
            worker_args = [
                (data[i], tau_true[i], a, b, N, A_true, B_true, cfg)
                for i in range(data.shape[0])
            ]
            with Pool(processes=n_workers) as pool:
                results = list(pool.imap_unordered(_worker, worker_args))
            kron_err[N] = np.array([r[0] for r in results])
            full_err[N] = np.array([r[1] for r in results])
            progress.advance(task)
    return kron_err, full_err


def _run_batched(datasets, N_vec, backend, a, b, A_true, B_true, cfg):
    kron_err, full_err = {}, {}
    A_t = get_data_on_device(A_true, backend)
    B_t = get_data_on_device(B_true, backend)
    logger.info(f"Starting batched structured/unstructured comparison on {backend}...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]N values...", total=len(N_vec))
        for N in N_vec:
            data, tau_true = datasets[N]
            X = get_data_on_device(data, backend)
            truth = (A_t, B_t, get_data_on_device(tau_true, backend))
            e_k, e_f = _fit_both(X, a, b, N, truth, cfg, backend)
            kron_err[N] = to_numpy(e_k)
            full_err[N] = to_numpy(e_f)
            maybe_empty_cache(backend)
            progress.advance(task)
    return kron_err, full_err


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = make_mc_parser(__doc__)
    add_mc_base_args(parser)
    parser.set_defaults(n_trials=200)

    parser.add_argument("--a", type=int, default=3, help="Size of the first Kronecker factor.")
    parser.add_argument("--b", type=int, default=4, help="Size of the second Kronecker factor.")
    parser.add_argument("--T", type=int, default=25,
        help="Number of dates, held fixed while N varies (default 25).")
    parser.add_argument("--N-list", type=int, nargs="+", default=[4, 6, 9, 13, 20, 30, 45, 70],
        help="Patch sizes to sweep (default 4 6 9 13 20 30 45 70; p = a*b = 12).")
    parser.add_argument("--nu", type=float, default=1.0,
        help="Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0).")
    parser.add_argument("--rho-a", type=str, default="0.3+0.7j", help="Toeplitz coefficient of A.")
    parser.add_argument("--rho-b", type=str, default="0.3+0.6j", help="Toeplitz coefficient of B.")
    parser.add_argument("--offline", type=str, default="gd", choices=["mm", "gd"],
        help="Structured estimator: 'gd' Riemannian gradient descent, term for term "
             "comparable with the unstructured one (default), or 'mm'.")
    parser.add_argument("--mm-iter-max", type=int, default=50, help="Max MM iterations.")
    parser.add_argument("--mm-tol", type=float, default=1e-8, help="MM tolerance.")
    parser.add_argument("--gd-iter-max", type=int, default=200, help="Max GD iterations.")
    parser.add_argument("--gd-tol", type=float, default=1e-8, help="GD tolerance.")
    parser.add_argument("--debug", action="store_true",
        help="Tiny configuration (6 trials, 3 values of N, T=8) to validate the "
             "pipeline in seconds. Results are NOT publication grade.")
    args = parser.parse_args()

    if args.debug:
        args.n_trials, args.T, args.N_list = 6, 8, [6, 13, 30]
        args.gd_iter_max, args.mm_iter_max = 50, 20

    init_logging(args.backend)
    if args.debug:
        logger.warning("--debug: 6 trials, T=8, N in {6, 13, 30}. Pipeline check only.")

    a, b, p, T = args.a, args.b, args.a * args.b, args.T
    N_vec = sorted(set(args.N_list))

    cfg = {
        "offline": args.offline,
        "mm_iter_max": args.mm_iter_max,
        "mm_tol": args.mm_tol,
        "gd_iter_max": args.gd_iter_max,
        "gd_tol": args.gd_tol,
    }

    logger.info(f"Structure vs N: a={a}, b={b}, p={p}, T={T}, N={N_vec}, "
                f"n_trials={args.n_trials}, backend={args.backend}")
    logger.info(f"  texture ~ Gamma({args.nu}, {1 / args.nu:.3g}) | structured estimator={args.offline}")

    A_true, B_true = make_ab_toeplitz(a, b, complex(args.rho_a), complex(args.rho_b))

    datasets = {}
    for N in N_vec:
        datasets[N] = generate_kronecker_data(
            args.n_trials, T, N, a, b, A_true, B_true,
            seed=args.seed + N, tau_shape=args.nu, tau_scale=1.0 / args.nu, return_tau=True,
        )

    exporter = MCResultExporter(
        args, Path(args.export_path), f"a{a}_b{b}_T{T}_n{args.n_trials}",
        plot_template=_MC_PLOT_TEMPLATE_STRUCT,
    )

    (kron_err, full_err), elapsed = timed_run(
        args,
        lambda: _run_pool(datasets, N_vec, args.n_workers, a, b, A_true, B_true, cfg),
        lambda: _run_batched(datasets, N_vec, args.backend, a, b, A_true, B_true, cfg),
    )

    icrb_kron = np.array([icrb_kronecker_scaled_gaussian(a, b, N, T)["total"] for N in N_vec])
    icrb_full = np.array([icrb_scaled_gaussian(p, N, T)["total"] for N in N_vec])

    title = f"Ce que la structure achète  (a={a}, b={b}, T={T}, nu={args.nu})"
    finish_struct(args, exporter, kron_err, full_err, icrb_kron, icrb_full, N_vec,
                  "mc_kron_struct", title, elapsed)


if __name__ == "__main__":
    main()
