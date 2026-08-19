# Tyler's cost along a segment and along a geodesic
#
# The same cost is read along the two paths joining the same two points of the
# cone: the Euclidean segment (1-t) A + t B, which does stay inside the cone
# since it is convex, and the affine-invariant geodesic A #_t B.
#
# The two endpoints are a strongly ill-conditioned matrix and its inverse,
# which are at equal distance from the identity — the true scatter matrix of
# the data — so the middle of both paths is where the minimiser should be.
# It is where the geodesic finds it, and it is a local *maximum* of the
# Euclidean reading, which therefore shows two spurious minima at its ends.
# The criterion is the same in both panels: what changes is the notion of
# straight line along which it is read, and with it the convexity.
#
# The cost is hdrlib.core.estimation.tyler_cost, the very function the
# convergence experiment minimises.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.estimation import tyler_cost
from hdrlib.core.elliptical import StudentTDistribution, sample_elliptical
from hdrlib.core.manifolds import HermitianPositiveDefinite


def endpoints(condition, rotation):
    """A matrix of unit determinant and its inverse, in a common basis.

    Their eigenvalues span the given condition number symmetrically in
    logarithm, so the two are exchanged by inversion and their geometric mean
    — the middle of the geodesic — is the identity.
    """
    n_features = rotation.shape[0]
    eigenvalues = np.logspace(
        -0.5 * np.log10(condition), 0.5 * np.log10(condition), n_features
    )
    start = rotation @ np.diag(eigenvalues) @ rotation.T
    end = rotation @ np.diag(1.0 / eigenvalues) @ rotation.T
    return start, end


def local_minima(times, values):
    """Indices of the strict local minima of a sampled curve, ends included."""
    interior = [
        index
        for index in range(1, len(values) - 1)
        if values[index] < values[index - 1] and values[index] < values[index + 1]
    ]
    if values[0] < values[1]:
        interior.insert(0, 0)
    if values[-1] < values[-2]:
        interior.append(len(values) - 1)
    return interior


def second_difference(times, values):
    """Discrete second derivative, negative wherever the curve is concave."""
    step = times[1] - times[0]
    return (values[2:] - 2 * values[1:-1] + values[:-2]) / step**2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Tyler's cost read along a Euclidean segment and along a geodesic."
    )
    parser.add_argument(
        "--n_features", type=int, default=3, help="Dimension of the observations."
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of observations. A short sample makes the cost surface "
             "sharper, hence the effect easier to see; the phenomenon itself "
             "does not depend on it.",
    )
    parser.add_argument(
        "--dof", type=float, default=3.0,
        help="Degrees of freedom of the Student data.",
    )
    parser.add_argument(
        "--condition", type=float, default=1e4,
        help="Condition number of the two endpoints. The larger it is, the "
             "more pronounced the interior maximum of the Euclidean reading.",
    )
    parser.add_argument(
        "--n_points", type=int, default=201,
        help="Number of points at which the cost is evaluated on each path.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/riemann_gconvexite",
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
    rotation = np.linalg.qr(rng.standard_normal((d, d)))[0]
    start, end = endpoints(args.condition, rotation)

    # Spherical data: the true scatter matrix is the identity, which is both
    # the middle of the geodesic and the point the two endpoints surround.
    distribution = StudentTDistribution(d, dof=args.dof, backend_name=args.backend)
    data = to_numpy(
        sample_elliptical(
            args.n_samples,
            get_data_on_device(np.zeros(d), args.backend),
            get_data_on_device(np.eye(d), args.backend),
            distribution,
            seed=args.seed,
        )
    )
    data_device = get_data_on_device(data, args.backend)
    start_device = get_data_on_device(start, args.backend)
    end_device = get_data_on_device(end, args.backend)
    direction = manifold.log(start_device, end_device)

    times = np.linspace(0.0, 1.0, args.n_points)
    costs = {
        "segment euclidien": np.array([
            tyler_cost(
                data_device,
                get_data_on_device((1.0 - t) * start + t * end, args.backend),
                args.backend,
            )
            for t in times
        ]),
        "géodésique": np.array([
            tyler_cost(
                data_device, manifold.exp(start_device, t * direction), args.backend
            )
            for t in times
        ]),
    }

    colors = {"segment euclidien": "C1", "géodésique": "C2"}

    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4), sharey=True)
    for ax, (name, values) in zip(axes, costs.items()):
        ax.plot(times, values, color=colors[name], linewidth=1.5)
        minima = local_minima(times, values)
        ax.plot(
            times[minima], values[minima],
            marker="o", markersize=4, linestyle="none", color="C7", zorder=3,
        )
        ax.set_xlabel(r"$t$")
        ax.set_title(name)
    axes[0].set_ylabel(r"$L$")

    fig.tight_layout()

    print(f"d = {d}, N = {args.n_samples}, Student nu = {args.dof:g}, "
          f"condition {args.condition:g}, geodesic distance between the "
          f"endpoints {float(manifold.dist(start_device, end_device)):.3f}")
    for name, values in costs.items():
        curvature = second_difference(times, values)
        minima = local_minima(times, values)
        print(
            f"  {name:18} min curvature {curvature.min():+9.2f}   "
            f"local minima at t = "
            + ", ".join(f"{times[index]:.2f}" for index in minima)
        )

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed, n_features=d, n_samples=args.n_samples, dof=args.dof,
        condition=args.condition, n_points=args.n_points,
        start=start, end=end, data=data, times=times,
        cost_euclidean=costs["segment euclidien"],
        cost_geodesic=costs["géodésique"],
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "gconvexite.tex")
        save(save_path, axis_width=args.axis_width, axis_height=args.axis_height)
        write_prov_sidecar(save_path, args)
        print(f"Saved cost profiles in {save_path}")

    if args.show_interactive:
        plt.show()
