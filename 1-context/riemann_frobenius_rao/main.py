# The same two estimators, measured with two different rulers
#
# The estimation error of the SCM and of Tyler's estimator is followed as a
# function of the number of observations, on Student data, and measured twice:
# once with the Frobenius norm of the difference, once with the Rao distance,
# which for the centred Gaussian model is the affine-invariant distance of the
# cone.
#
# Both estimators and the truth are normalised to unit determinant before the
# comparison: Tyler's estimator only identifies a shape, and the two rulers
# would otherwise be comparing scales rather than models.
#
# What the figure shows is that the two rulers do not cross at the same place.
# In the very short-sample regime — a handful of observations more than the
# dimension — the Frobenius norm already prefers Tyler while the Rao distance
# still prefers the SCM: the ordering of two estimators is a property of the
# metric, and not of the estimators alone.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from rich.progress import Progress

from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.estimation import SCMEstimator, minimize_tyler_fixed_point
from hdrlib.core.elliptical import StudentTDistribution, sample_elliptical
from hdrlib.core.manifolds import HermitianPositiveDefinite


def normalize_determinant(matrix):
    """Rescale to unit determinant, the only scale Tyler's estimator fixes."""
    n_features = matrix.shape[-1]
    return matrix / np.linalg.det(matrix) ** (1.0 / n_features)


def scatter_matrix(n_features, condition, rng):
    """Ill-conditioned scatter matrix of unit determinant."""
    eigenvalues = np.logspace(
        -0.5 * np.log10(condition), 0.5 * np.log10(condition), n_features
    )
    rotation = np.linalg.qr(rng.standard_normal((n_features, n_features)))[0]
    return normalize_determinant(rotation @ np.diag(eigenvalues) @ rotation.T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Estimation error of the SCM and of Tyler's estimator, in Frobenius "
        "norm and in Rao distance."
    )
    parser.add_argument(
        "--n_features", type=int, default=7, help="Dimension of the observations."
    )
    parser.add_argument(
        "--n_samples", type=int, nargs="+",
        default=[9, 10, 12, 14, 18, 25, 40, 70, 120, 200],
        help="Sample sizes at which the error is evaluated. The first ones are "
             "just above the dimension, which is where the two metrics "
             "disagree.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=500, help="Number of MC-trials per sample size."
    )
    parser.add_argument(
        "--dof", type=float, default=3.0,
        help="Degrees of freedom of the Student data. Heavy enough for the SCM "
             "to suffer, light enough for its covariance to exist.",
    )
    parser.add_argument(
        "--condition", type=float, default=50.0,
        help="Condition number of the true scatter matrix.",
    )
    parser.add_argument(
        "--iter_max", type=int, default=300, help="Fixed-point iterations for Tyler."
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/riemann_frobenius_rao",
        help="Output directory for LaTeX exports (injected by qanat, or set manually).",
    )
    parser.add_argument(
        "--show-interactive", action="store_true",
        help="Show plots interactively with matplotlib.",
    )
    parser.add_argument(
        "--export", action=argparse.BooleanOptionalAction, default=True,
        help="Save TikZ/PGFPlots figure (.tex) (default: True).",
    )
    parser.add_argument(
        "--axis_width", type=str, default="0.45\\textwidth",
        help="Width of a single panel in the exported PGFPlots figure. Set "
             "here rather than patched into the .tex afterwards, so that a "
             "re-sync into the dissertation does not undo it.",
    )
    parser.add_argument(
        "--axis_height", type=str, default="4.6cm",
        help="Height of a single panel in the exported PGFPlots figure.",
    )
    parser.add_argument(
        "--backend", type=str, default="numpy",
        help="Compute backend (numpy, torch-cpu, torch-mps, ...).",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed generation base seed")
    args = parser.parse_args()

    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)

    d = args.n_features
    manifold = HermitianPositiveDefinite(d, backend_name=args.backend)

    rng = np.random.default_rng(args.seed)
    scatter = scatter_matrix(d, args.condition, rng)
    scatter_device = get_data_on_device(scatter, args.backend)
    mean_device = get_data_on_device(np.zeros(d), args.backend)
    distribution = StudentTDistribution(d, dof=args.dof, backend_name=args.backend)

    estimators = ("scm", "Tyler")
    errors = {
        (name, metric): np.zeros((len(args.n_samples), args.n_trials))
        for name in estimators
        for metric in ("frobenius", "rao")
    }

    with Progress() as progress:
        task = progress.add_task(
            "Monte-Carlo", total=len(args.n_samples) * args.n_trials
        )
        for index, n_samples in enumerate(args.n_samples):
            for trial in range(args.n_trials):
                data = sample_elliptical(
                    n_samples, mean_device, scatter_device, distribution,
                    seed=args.seed + 1000 * index + trial,
                )
                tyler, _ = minimize_tyler_fixed_point(
                    data, iter_max=args.iter_max, backend_name=args.backend
                )
                estimates = {
                    "scm": normalize_determinant(
                        to_numpy(SCMEstimator(backend_name=args.backend).compute(data))
                    ),
                    "Tyler": normalize_determinant(to_numpy(tyler)),
                }
                for name, estimate in estimates.items():
                    errors[(name, "frobenius")][index, trial] = np.linalg.norm(
                        estimate - scatter, ord="fro"
                    )
                    errors[(name, "rao")][index, trial] = float(
                        manifold.dist(
                            get_data_on_device(estimate, args.backend), scatter_device
                        )
                    )
                progress.advance(task)

    means = {key: value.mean(axis=1) for key, value in errors.items()}

    colors = {"scm": "C1", "Tyler": "C2"}
    markers = {"scm": "o", "Tyler": "s"}
    titles = {
        "frobenius": r"norme de Frobenius",
        "rao": r"distance de Rao",
    }

    n_samples = np.array(args.n_samples)
    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4))
    for column, (ax, metric) in enumerate(zip(axes, ("frobenius", "rao"))):
        for name in estimators:
            # Labelled on the left panel only: matplot2tikz gathers the
            # labelled curves of every axis into the one exported legend.
            ax.plot(
                n_samples, means[(name, metric)],
                color=colors[name], marker=markers[name], markersize=4,
                linewidth=1.4, label=name if column == 0 else None,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N$")
        ax.set_title(titles[metric])
    axes[0].set_ylabel("erreur moyenne")
    # Legend below the panels rather than inside one of them, see the other
    # Riemannian figures: at this width an inner legend covers the curves.
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.45), ncol=2,
        frameon=False, fontsize=8,
    )

    fig.tight_layout()

    print(f"d = {d}, Student nu = {args.dof:g}, condition {args.condition:g}, "
          f"{args.n_trials} trials")
    for index, n in enumerate(args.n_samples):
        frobenius = {name: means[(name, "frobenius")][index] for name in estimators}
        rao = {name: means[(name, "rao")][index] for name in estimators}
        print(
            f"  N = {n:4d}   Frobenius: scm {frobenius['scm']:8.3f} "
            f"Tyler {frobenius['Tyler']:8.3f} -> "
            f"{min(frobenius, key=frobenius.get):5}   |   "
            f"Rao: scm {rao['scm']:6.3f} Tyler {rao['Tyler']:6.3f} -> "
            f"{min(rao, key=rao.get):5}"
        )

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed, n_features=d, n_trials=args.n_trials, dof=args.dof,
        condition=args.condition, n_samples=n_samples, scatter=scatter,
        **{f"{name}_{metric}": errors[(name, metric)]
           for name in estimators for metric in ("frobenius", "rao")},
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "erreurrao.tex")
        save_tikz(
            save_path, axis_width=args.axis_width, axis_height=args.axis_height
        )
        write_prov_sidecar(save_path, args)
        print(f"Saved error curves in {save_path}")

    if args.show_interactive:
        plt.show()
