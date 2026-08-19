# Three ways of minimising the same criterion
#
# Tyler's cost is minimised from the same starting point by three algorithms:
# the fixed-point iteration, the Riemannian gradient descent with a
# backtracking line search, and a Euclidean gradient descent that has to
# project its iterates back onto the cone. The left panel counts iterations,
# the right one seconds.
#
# The first two curves are nearly on top of each other, which is the point of
# the chapter: the fixed point *is* a Riemannian gradient step of unit length,
# so it cannot do better than the descent it turns out to be an instance of.
# The Euclidean method minimises the same function and reaches the same
# minimiser, only much more slowly: its gradient ignores the geometry that
# the congruence Sigma . Sigma restores.
#
# The three minimisers come from hdrlib.core.estimation and are written out
# there — gradient, step rule and retraction — rather than delegated to a
# manifold optimisation library.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.estimation import (
    minimize_tyler_fixed_point,
    minimize_tyler_riemannian,
    minimize_tyler_euclidean,
)
from hdrlib.core.elliptical import StudentTDistribution, sample_elliptical
from hdrlib.core.manifolds import HermitianPositiveDefinite


def scatter_matrix(n_features, condition, rng):
    """Ill-conditioned scatter matrix of unit determinant."""
    eigenvalues = np.logspace(
        -0.5 * np.log10(condition), 0.5 * np.log10(condition), n_features
    )
    rotation = np.linalg.qr(rng.standard_normal((n_features, n_features)))[0]
    scatter = rotation @ np.diag(eigenvalues) @ rotation.T
    return scatter / np.linalg.det(scatter) ** (1.0 / n_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fixed point, Riemannian descent and projected Euclidean descent on "
        "Tyler's cost."
    )
    parser.add_argument(
        "--n_features", type=int, default=10, help="Dimension of the observations."
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of observations used by the three algorithms.",
    )
    parser.add_argument(
        "--dof", type=float, default=3.0,
        help="Degrees of freedom of the Student data.",
    )
    parser.add_argument(
        "--condition", type=float, default=100.0,
        help="Condition number of the true scatter matrix. The larger it is, "
             "the further the identity — the common starting point — sits "
             "from the solution.",
    )
    parser.add_argument(
        "--iter_max", type=int, default=150,
        help="Maximum number of iterations granted to each algorithm.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-12,
        help="Stopping tolerance on the Riemannian gradient norm. Deliberately "
             "unreachable, so that every algorithm spends its whole budget and "
             "the curves can be compared over their full length.",
    )
    parser.add_argument(
        "--floor", type=float, default=1e-14,
        help="Smallest optimality gap shown; below it the cost is dominated by "
             "rounding rather than by the algorithm.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/riemann_convergence",
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
    distribution = StudentTDistribution(d, dof=args.dof, backend_name=args.backend)
    data = sample_elliptical(
        args.n_samples,
        get_data_on_device(np.zeros(d), args.backend),
        get_data_on_device(scatter, args.backend),
        distribution,
        seed=args.seed,
    )

    # Same budget, same starting point — the identity — for the three.
    solutions, histories = {}, {}
    runs = {
        "point fixe": minimize_tyler_fixed_point,
        "gradient riemannien": minimize_tyler_riemannian,
        "gradient euclidien projeté": minimize_tyler_euclidean,
    }
    for name, minimize in runs.items():
        solutions[name], histories[name] = minimize(
            data, iter_max=args.iter_max, tol=args.tol, backend_name=args.backend
        )

    # The reference value is the smallest cost any of the three reached: the
    # criterion has a single minimum, so this is the common target and what
    # makes the three gaps comparable.
    optimum = min(min(history["cost"]) for history in histories.values())
    # Anything below this is the noise of double precision on the cost, not
    # convergence, so the curves are clipped there rather than plunging to
    # whatever rounding happened to produce.
    floor = args.floor

    colors = {
        "point fixe": "C1",
        "gradient riemannien": "C2",
        "gradient euclidien projeté": "C3",
    }
    markers = {
        "point fixe": "o",
        "gradient riemannien": "s",
        "gradient euclidien projeté": "^",
    }

    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4), sharey=True)
    for name, history in histories.items():
        gap = np.maximum(np.array(history["cost"]) - optimum, floor)
        # The label is attached to the left panel only: matplot2tikz collects
        # the labelled curves of every axis into the single exported legend,
        # so labelling both panels lists each algorithm twice.
        for column, abscissa in enumerate(
            [np.arange(len(gap)), np.array(history["time"])]
        ):
            axes[column].plot(
                abscissa, gap,
                color=colors[name], linewidth=1.4,
                marker=markers[name], markersize=3, markevery=10,
                label=name if column == 0 else None,
            )
    axes[0].set_xlabel("itération")
    axes[0].set_ylabel(r"$L - L^{\star}$")
    axes[1].set_xlabel("temps (s)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_yscale("log")
    axes[0].set_ylim(bottom=0.5 * floor)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)

    fig.tight_layout()

    print(f"d = {d}, N = {args.n_samples}, Student nu = {args.dof:g}, "
          f"condition {args.condition:g}")
    for name, history in histories.items():
        distance = float(
            manifold.dist(
                get_data_on_device(solutions[name], args.backend),
                get_data_on_device(solutions["point fixe"], args.backend),
            )
        )
        print(
            f"  {name:27} {len(history['cost']) - 1:4d} iterations   "
            f"final gap {history['cost'][-1] - optimum:.2e}   "
            f"gradient {history['gradient_norm'][-1]:.2e}   "
            f"{history['time'][-1]:.3f} s   "
            f"distance to the fixed point {distance:.2e}"
        )

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed, n_features=d, n_samples=args.n_samples, dof=args.dof,
        condition=args.condition, iter_max=args.iter_max, tol=args.tol,
        scatter=scatter, data=to_numpy(data), optimum=optimum,
        **{
            f"{key}_{name.replace(' ', '_').replace('é', 'e')}": np.array(values)
            for name, history in histories.items()
            for key, values in history.items()
            if key in ("cost", "gradient_norm", "time")
        },
        **{
            f"solution_{name.replace(' ', '_').replace('é', 'e')}":
            to_numpy(solution)
            for name, solution in solutions.items()
        },
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "convergence.tex")
        save(save_path, axis_width=args.axis_width, axis_height=args.axis_height)
        write_prov_sidecar(save_path, args)
        print(f"Saved convergence curves in {save_path}")

    if args.show_interactive:
        plt.show()
