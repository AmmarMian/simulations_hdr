# Showing behavior of LWF  on simple data with case dim  > N_obs

import argparse

import numpy as np
import matplotlib.pyplot as plt
from rich.progress import Progress

import os
from multiprocessing import Pool

from itertools import product

from matplot2tikz import save

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def compute_estimation(params):
    d, N, trial_no, alpha = params
    rng = np.random.default_rng(
        d + N + trial_no
    )  # To be sure the generation of data is different
    mean = np.ones((d, N))
    data = mean + rng.standard_normal((d, N))
    estimate_cov_scm = np.cov(data)
    estimate_cov_lwf = (1 - alpha) * estimate_cov_scm + alpha * np.eye(d)
    return [
        d,
        np.linalg.cond(estimate_cov_scm),
        np.linalg.cond(estimate_cov_lwf),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "MC simulation of estimating covariance with increasing number of variables (lwf)"
    )
    parser.add_argument("--N", type=int, default=30, help="Number of observations.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of MC-trials.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/error_estimation_lwf_underregime",
        help="Directory for LaTeX exports.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Coefficient of regularization."
    )
    args = parser.parse_args()

    # Constants
    N = args.N
    d_vec = np.unique(np.logspace(0.5, 3, args.N, dtype=int))
    n_trials = args.n_trials

    # Montecarlo with progressbar.
    # Inspired from: https://github.com/Textualize/rich/discussions/884#discussioncomment-269200
    print("Launching simulation")
    param_grid = product(d_vec, [N], list(range(n_trials)), [args.alpha])
    results = []
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Working...", total=len(d_vec) * n_trials)
        with Pool(os.cpu_count()) as pool:
            for result in pool.imap(compute_estimation, param_grid):
                results.append(result)
                progress.advance(task_id)

        results = np.stack(results)

    # Averaging monte-carlo
    cond_scm_mean = np.stack([results[results[:, 0] == d, 1].mean() for d in d_vec])
    cond_scm_std = np.stack([results[results[:, 0] == d, 1].std() for d in d_vec])

    cond_lwf_mean = np.stack([results[results[:, 0] == d, 2].mean() for d in d_vec])
    cond_lwf_std = np.stack([results[results[:, 0] == d, 2].std() for d in d_vec])
    print("Done.")

    # Plotting
    fig = plt.figure()
    plt.scatter(
        d_vec, cond_scm_mean, marker="o", facecolors="none", edgecolors="k", label="scm"
    )
    plt.scatter(
        d_vec,
        cond_lwf_mean,
        marker="s",
        facecolors="none",
        edgecolors="k",
        label="lwf",
    )
    # plt.errorbar(d_vec, cond_cov_mean, yerr=cond_cov_std, linestyle="", capsize=5)
    plt.axvline(x=N, color="k", label=r"$d=N$", linestyle="--")
    plt.xlabel("$d$")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(
        r"$\operatorname{Cond}\left(\hat{\boldsymbol{\Sigma}}_\mathcal{X}\right)$"
    )
    plt.title(
        f"Condition number of estimated covariance with {n_trials} Monte-carlo trials"
    )
    plt.legend()
    save_path = os.path.join(args.output_dir, "cond.tex")
    save(save_path)
    print(f"Saved cov error in {save_path}")

    plt.show()
