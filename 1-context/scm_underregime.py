# Showing behavior of SCM on simple data with case dim  > N_obs

import argparse

import numpy as np
import matplotlib.pyplot as plt
from rich.progress import Progress

import os
from multiprocessing import Pool

from itertools import product

from matplot2tikz import clean_figure, save

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def compute_estimation(params):
    d, N, trial_no = params
    rng = np.random.default_rng(
        d + N + trial_no
    )  # To be sure the generation of data is different
    mean = np.ones((d, N))
    data = mean + rng.standard_normal((d, N))

    estimate_mean = np.mean(data, axis=1)
    estimate_cov = np.cov(data)
    return [
        d,
        np.linalg.norm(estimate_mean - mean[:, 0]),
        np.linalg.norm(estimate_cov - np.eye(d), ord="fro"),
        np.linalg.cond(estimate_cov),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "MC simulation of estimating mean and covariance with increasing samples."
    )
    parser.add_argument("--N", type=int, default=30, help="Number of observations.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of MC-trials.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/error_estimation_scm_underregime",
        help="Directory for LaTeX exports.",
    )
    args = parser.parse_args()

    # Constants
    N = args.N
    d_vec = np.unique(np.logspace(0.5, 3, args.N, dtype=int))
    n_trials = args.n_trials

    # Montecarlo with progressbar.
    # Inspired from: https://github.com/Textualize/rich/discussions/884#discussioncomment-269200
    print("Launching simulation")
    param_grid = product(d_vec, [N], list(range(n_trials)))
    results = []
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Working...", total=len(d_vec) * n_trials)
        with Pool(os.cpu_count()) as pool:
            for result in pool.imap(compute_estimation, param_grid):
                results.append(result)
                progress.advance(task_id)

        results = np.stack(results)

    # Averaging monte-carlo
    error_mean_mean = np.stack([results[results[:, 0] == d, 1].mean() for d in d_vec])
    error_mean_std = np.stack([results[results[:, 0] == d, 1].std() for d in d_vec])

    error_cov_mean = np.stack([results[results[:, 0] == d, 2].mean() for d in d_vec])
    error_cov_std = np.stack([results[results[:, 0] == d, 2].std() for d in d_vec])

    cond_cov_mean = np.stack([results[results[:, 0] == d, 3].mean() for d in d_vec])
    cond_cov_std = np.stack([results[results[:, 0] == d, 3].std() for d in d_vec])
    print("Done.")

    # Plotting
    fig = plt.figure()
    plt.scatter(d_vec, error_mean_mean, marker="o", facecolors="none", edgecolors="k")
    plt.errorbar(d_vec, error_mean_mean, yerr=error_mean_std, linestyle="", capsize=5)
    plt.xlabel(r"$d$")
    plt.ylabel(
        r"$\|\hat{\boldsymbol{\mu}}_\mathcal{X} - \boldsymbol{\mu}_\mathcal{X}\|_2$"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Error of mean estimation with {n_trials} Monte-carlo trials")
    save_path = os.path.join(args.output_dir, "mean.tex")
    # save(save_path)
    print(f"Saved mean error in {save_path}")

    fig = plt.figure()
    plt.scatter(d_vec, error_cov_mean, marker="o", facecolors="none", edgecolors="k")
    plt.errorbar(d_vec, error_cov_mean, yerr=error_cov_std, linestyle="", capsize=5)
    plt.xlabel("$d$")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(
        r"$\|\hat{\boldsymbol{\Sigma}}_\mathcal{X} - \boldsymbol{\Sigma}_\mathcal{X}\|_2$"
    )
    plt.title(f"Error of mean estimation with {n_trials} Monte-carlo trials")
    save_path = os.path.join(args.output_dir, "cov.tex")
    # save(save_path)
    print(f"Saved cov error in {save_path}")

    fig = plt.figure()
    plt.scatter(d_vec, cond_cov_mean, marker="o", facecolors="none", edgecolors="k")
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
    # save(save_path)
    print(f"Saved cov error in {save_path}")

    plt.show()
