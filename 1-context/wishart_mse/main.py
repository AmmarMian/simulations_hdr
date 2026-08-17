# Monte-Carlo check of the closed-form MSE of the SCM under a Gaussian model.
#
# Under x_k ~ N(mu, Sigma) i.i.d., the unbiased SCM S/(N-1) is Wishart
# distributed, which gives the exact mean squared error
#
#     E ||Sigma_hat - Sigma||_F^2 = (||Sigma||_F^2 + tr(Sigma)^2) / (N - 1).
#
# Two sweeps confirm both regimes of that expression: the 1/N decay at fixed
# dimension, and the d^2 growth at fixed sample support.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from rich.progress import Progress

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.estimation import SCMEstimator
from hdrlib.core.simulation import T_vec_logspace


def make_covariance(d, kind, rho):
    """Real SPD covariance matrix, matching the regimes of the context chapter."""
    if kind == "identity":
        return np.eye(d)
    if kind == "toeplitz":
        return toeplitz(np.power(rho, np.arange(d)))
    raise ValueError(f"Unknown covariance kind: {kind}")


def theoretical_mse(cov, n_samples):
    """Exact MSE of the unbiased SCM, from the Wishart moments."""
    return (
        np.linalg.norm(cov, ord="fro") ** 2 + np.trace(cov) ** 2
    ) / (n_samples - 1)


def empirical_mse(cov, n_samples, n_trials, rng, max_elements=4_000_000):
    """Monte-Carlo mean and standard error of the SCM squared error.

    Trials are processed in chunks so memory stays bounded when both
    n_samples and n_trials are large.
    """
    d = cov.shape[0]
    cholesky = np.linalg.cholesky(cov)
    # Unbiased SCM with estimated mean: SCMEstimator divides by n_samples, so
    # the n_samples / (n_samples - 1) factor restores the (N-1) normalisation.
    estimator = SCMEstimator(assume_centered=False)
    correction = n_samples / (n_samples - 1)

    chunk = max(1, min(n_trials, int(max_elements // (n_samples * d))))
    errors = []
    for start in range(0, n_trials, chunk):
        size = min(chunk, n_trials - start)
        data = rng.standard_normal((size, n_samples, d)) @ cholesky.T
        estimates = correction * estimator.compute(data)
        errors.append(
            np.linalg.norm(estimates - cov, ord="fro", axis=(-2, -1)) ** 2
        )
    errors = np.concatenate(errors)
    # Standard error of the mean: we are checking the expectation itself, not
    # the per-trial spread, which for a chi-squared-like quantity is of the
    # same order as the mean and would go negative on a log axis.
    return errors.mean(), errors.std() / np.sqrt(errors.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Monte-Carlo verification of the closed-form MSE of the SCM."
    )
    parser.add_argument(
        "--n_trials", type=int, default=10000, help="Number of MC-trials per point."
    )
    parser.add_argument(
        "--d_sweep_values", type=int, nargs="+", default=[7, 20],
        help="Dimensions shown in the N-sweep panel.",
    )
    parser.add_argument(
        "--n_sweep_values", type=int, nargs="+", default=[100, 500],
        help="Sample supports shown in the d-sweep panel.",
    )
    parser.add_argument(
        "--n_min", type=int, default=30, help="Smallest N of the N-sweep."
    )
    parser.add_argument(
        "--n_max", type=int, default=10000, help="Largest N of the N-sweep."
    )
    parser.add_argument(
        "--n_points", type=int, default=12, help="Number of points per sweep."
    )
    parser.add_argument(
        "--d_max", type=int, default=40, help="Largest dimension of the d-sweep."
    )
    parser.add_argument(
        "--covariance", type=str, default="toeplitz", choices=["toeplitz", "identity"],
        help="True covariance regime. Toeplitz exercises the full formula, "
             "identity reduces it to d(d+1)/(N-1).",
    )
    parser.add_argument(
        "--rho", type=float, default=0.8, help="Correlation of the Toeplitz regime."
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="outputs/wishart_mse",
        help="Output directory for LaTeX exports (injected by qanat, or set manually).",
    )
    parser.add_argument(
        "--show-interactive",
        action="store_true",
        help="Show plots interactively with matplotlib.",
    )
    parser.add_argument(
        "--export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save TikZ/PGFPlots figure (.tex) (default: True).",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed generation base seed")
    args = parser.parse_args()

    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    N_vec = np.array(T_vec_logspace(args.n_min, args.n_max, args.n_points))
    d_vec = np.array(T_vec_logspace(2, args.d_max, args.n_points))

    n_points_total = len(args.d_sweep_values) * len(N_vec) + len(args.n_sweep_values) * len(d_vec)
    print("Launching simulation")

    # Sweep 1: error against N, at fixed dimension
    mse_vs_N, std_vs_N, theory_vs_N = {}, {}, {}
    # Sweep 2: error against d, at fixed sample support
    mse_vs_d, std_vs_d, theory_vs_d = {}, {}, {}

    with Progress() as progress:
        task_id = progress.add_task("[cyan]Working...", total=n_points_total)

        for d in args.d_sweep_values:
            cov = make_covariance(d, args.covariance, args.rho)
            means, stds, theory = [], [], []
            for n_samples in N_vec:
                mean, std = empirical_mse(cov, int(n_samples), args.n_trials, rng)
                means.append(mean)
                stds.append(std)
                theory.append(theoretical_mse(cov, int(n_samples)))
                progress.advance(task_id)
            mse_vs_N[d], std_vs_N[d], theory_vs_N[d] = (
                np.array(means), np.array(stds), np.array(theory)
            )

        for n_samples in args.n_sweep_values:
            means, stds, theory = [], [], []
            for d in d_vec:
                cov = make_covariance(int(d), args.covariance, args.rho)
                mean, std = empirical_mse(cov, n_samples, args.n_trials, rng)
                means.append(mean)
                stds.append(std)
                theory.append(theoretical_mse(cov, n_samples))
                progress.advance(task_id)
            mse_vs_d[n_samples], std_vs_d[n_samples], theory_vs_d[n_samples] = (
                np.array(means), np.array(stds), np.array(theory)
            )

    print("Done.")

    # Largest relative deviation, as a scalar sanity check
    deviations = [
        np.abs(mse_vs_N[d] / theory_vs_N[d] - 1).max() for d in args.d_sweep_values
    ] + [
        np.abs(mse_vs_d[n] / theory_vs_d[n] - 1).max() for n in args.n_sweep_values
    ]
    print(f"Largest relative deviation from theory: {max(deviations):.2%}")

    # Save results
    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed,
        n_trials=args.n_trials,
        covariance=args.covariance,
        rho=args.rho,
        N_vec=N_vec,
        d_vec=d_vec,
        d_sweep_values=np.array(args.d_sweep_values),
        n_sweep_values=np.array(args.n_sweep_values),
        **{f"mse_vs_N_d{d}": mse_vs_N[d] for d in args.d_sweep_values},
        **{f"std_vs_N_d{d}": std_vs_N[d] for d in args.d_sweep_values},
        **{f"theory_vs_N_d{d}": theory_vs_N[d] for d in args.d_sweep_values},
        **{f"mse_vs_d_N{n}": mse_vs_d[n] for n in args.n_sweep_values},
        **{f"std_vs_d_N{n}": std_vs_d[n] for n in args.n_sweep_values},
        **{f"theory_vs_d_N{n}": theory_vs_d[n] for n in args.n_sweep_values},
    )

    # Plotting: ratio of the Monte-Carlo estimate to the closed-form value.
    # Plotting both in absolute value on a log axis spanning several decades
    # would hide any disagreement — the ratio is what actually tests the
    # formula, with the error bars giving the scale of what counts as a
    # deviation.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    def plot_ratio(ax, x, mse, standard_error, theory, color, label):
        ratio = mse / theory
        ax.scatter(
            x, ratio, marker="o", s=22,
            facecolors="none", edgecolors=color, linewidths=0.9,
            label=label, zorder=3,
        )
        errline, _, _ = ax.errorbar(
            x, ratio, yerr=2 * standard_error / theory,
            linestyle="", capsize=3, ecolor=color, zorder=1,
        )
        # matplot2tikz exports an empty linestyle as a solid connecting line
        errline.set_visible(False)

    for i, d in enumerate(args.d_sweep_values):
        plot_ratio(
            axes[0], N_vec, mse_vs_N[d], std_vs_N[d], theory_vs_N[d],
            f"C{i}", rf"$d = {d}$",
        )
    axes[0].axhline(1.0, color="C7", linestyle="--", linewidth=1.0, zorder=0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$N$")
    axes[0].set_ylabel(r"$\mathrm{EQM}_{\mathrm{MC}} / \mathrm{EQM}_{\mathrm{th}}$")
    axes[0].legend()

    for i, n_samples in enumerate(args.n_sweep_values):
        plot_ratio(
            axes[1], d_vec, mse_vs_d[n_samples], std_vs_d[n_samples],
            theory_vs_d[n_samples], f"C{i}", rf"$N = {n_samples}$",
        )
    axes[1].axhline(1.0, color="C7", linestyle="--", linewidth=1.0, zorder=0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$d$")
    axes[1].legend()

    fig.tight_layout()

    if args.export:
        save_path = os.path.join(args.storage_path, "wishart_mse.tex")
        save(save_path)
        write_prov_sidecar(save_path, args)
        print(f"Saved MSE verification in {save_path}")

    if args.show_interactive:
        plt.show()
