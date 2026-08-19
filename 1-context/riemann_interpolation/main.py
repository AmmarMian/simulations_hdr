# Three ways of going from one covariance matrix to another
#
# The same two 2x2 matrices are joined by the Euclidean segment, by the
# affine-invariant geodesic and by the log-Euclidean geodesic. Each path is
# drawn as a family of concentration ellipses, and the last panel follows the
# determinant along the three of them.
#
# The two endpoints are deliberately taken with the same determinant and
# orthogonal principal directions, which is the configuration where the
# Euclidean average of two ellipses is visibly larger than either of them:
# the determinant bulges in the middle. The two Riemannian paths interpolate
# the determinant geometrically instead, so nothing is created along the way.
#
# The geodesic and the exponential come from hdrlib.core.manifolds unchanged;
# the log-Euclidean path is the one closed form the chapter gives explicitly,
# so it is written out here rather than hidden behind a manifold object.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.manifolds import HermitianPositiveDefinite, logm_psd


def rotation(angle):
    """Plane rotation of the given angle, in radians."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def endpoints(condition, angle):
    """Two matrices of unit determinant, of the same shape but rotated.

    Both have eigenvalues sqrt(condition) and 1/sqrt(condition), so the
    Euclidean path between them cannot change the eigenvalues without
    changing the determinant — which is exactly what it does.
    """
    spread = np.sqrt(condition)
    diagonal = np.diag([spread, 1.0 / spread])
    rotated = rotation(angle) @ diagonal @ rotation(angle).T
    return diagonal, rotated


def euclidean_path(start, end, times):
    """The straight segment (1-t) A + t B, which stays in the cone."""
    return np.stack([(1.0 - t) * start + t * end for t in times])


def affine_invariant_path(start, end, times, manifold, backend):
    """The geodesic A #_t B, obtained as exp_A(t log_A(B))."""
    start_device = get_data_on_device(start, backend)
    end_device = get_data_on_device(end, backend)
    direction = manifold.log(start_device, end_device)
    return np.stack(
        [to_numpy(manifold.exp(start_device, t * direction)) for t in times]
    )


def log_euclidean_path(start, end, times, backend):
    """expm((1-t) logm A + t logm B): the segment read in logarithm coordinates."""
    log_start = to_numpy(logm_psd(get_data_on_device(start, backend), backend))
    log_end = to_numpy(logm_psd(get_data_on_device(end, backend), backend))
    path = []
    for t in times:
        values, vectors = np.linalg.eigh((1.0 - t) * log_start + t * log_end)
        path.append(vectors @ np.diag(np.exp(values)) @ vectors.T)
    return np.stack(path)


def concentration_ellipse(shape, radius, n_points=200):
    """Curve {x : x^T shape^{-1} x = radius^2}, as a (2, n_points) array."""
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    values, vectors = np.linalg.eigh(shape)
    return vectors @ np.diag(np.sqrt(values)) @ circle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Euclidean, affine-invariant and log-Euclidean paths between two covariances."
    )
    parser.add_argument(
        "--condition", type=float, default=16.0,
        help="Ratio of the two eigenvalues of each endpoint. The larger it is, "
             "the more elongated the ellipses and the more visible the "
             "swelling of the Euclidean path.",
    )
    parser.add_argument(
        "--angle", type=float, default=0.35,
        help="Angle between the principal directions of the two endpoints, in "
             "units of pi. A quarter turn, 0.5, is the worst case for the "
             "Euclidean path, but it makes the two endpoints commute, and the "
             "two Riemannian paths then coincide exactly; a value away from it "
             "keeps the swelling and separates them.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=7,
        help="Number of ellipses drawn along each path, endpoints included.",
    )
    parser.add_argument(
        "--radius", type=float, default=1.0,
        help="Radius of the drawn ellipses, in units of the Mahalanobis distance.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/riemann_interpolation",
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

    # Constant(s)
    d = 2

    manifold = HermitianPositiveDefinite(d, backend_name=args.backend)
    start, end = endpoints(args.condition, args.angle * np.pi)
    times = np.linspace(0.0, 1.0, args.n_steps)

    paths = {
        "euclidienne": euclidean_path(start, end, times),
        "affine invariante": affine_invariant_path(
            start, end, times, manifold, args.backend
        ),
        "log-euclidienne": log_euclidean_path(start, end, times, args.backend),
    }
    determinants = {
        name: np.array([np.linalg.det(matrix) for matrix in path])
        for name, path in paths.items()
    }

    colors = {
        "euclidienne": "C1",
        "affine invariante": "C2",
        "log-euclidienne": "C3",
    }

    fig, axes = plt.subplots(2, 2, figsize=(3.4 * 2, 3.4 * 2))
    axes = axes.ravel()

    limit = 1.15 * max(
        np.abs(concentration_ellipse(matrix, args.radius)).max()
        for path in paths.values() for matrix in path
    )

    for index, (ax, (name, path)) in enumerate(zip(axes, paths.items())):
        for step, matrix in enumerate(path):
            curve = concentration_ellipse(matrix, args.radius)
            # The endpoints are shared by the three panels and drawn alike, so
            # that only what happens between them distinguishes the metrics.
            endpoint = step in (0, len(path) - 1)
            ax.plot(
                curve[0], curve[1],
                color="C7" if endpoint else colors[name],
                linestyle="--" if endpoint else "-",
                linewidth=1.3 if endpoint else 1.0,
                zorder=3 if endpoint else 2,
            )
        ax.set_aspect("equal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(name)
        # Outer labels only: with four panels stacked in a text block this
        # narrow, an inner label runs into the title of the panel below it.
        if index >= 2:
            ax.set_xlabel(r"$x_1$")
        if index % 2 == 0:
            ax.set_ylabel(r"$x_2$")

    ax = axes[3]
    fine_times = np.linspace(0.0, 1.0, 101)
    fine_paths = {
        "euclidienne": euclidean_path(start, end, fine_times),
        "affine invariante": affine_invariant_path(
            start, end, fine_times, manifold, args.backend
        ),
        "log-euclidienne": log_euclidean_path(start, end, fine_times, args.backend),
    }
    for name, path in fine_paths.items():
        ax.plot(
            fine_times,
            [np.linalg.det(matrix) for matrix in path],
            color=colors[name], linewidth=1.4, label=name,
        )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\det$")
    ax.set_title("déterminant le long du chemin")

    # Legend below the grid rather than inside a panel: the panels are barely
    # five centimetres wide once exported, and an inner legend either covers
    # the curves or spills over the frame. It is attached to the panel whose
    # curves it names, and pushed left of it so that it spans the whole width:
    # matplot2tikz only exports entries for the labelled curves of the very
    # axis the legend belongs to.
    ax.legend(
        loc="upper left", bbox_to_anchor=(-1.3, -0.45), ncol=3,
        frameon=False, fontsize=8,
    )

    fig.tight_layout()

    print(f"Endpoints of determinant {np.linalg.det(start):.3f} and "
          f"{np.linalg.det(end):.3f}, condition number {args.condition:g}")
    for name, values in determinants.items():
        print(f"  {name:18} det at t=1/2: {values[len(values) // 2]:.3f}"
              f"   max: {values.max():.3f}")

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed, condition=args.condition, angle=args.angle,
        n_steps=args.n_steps, radius=args.radius,
        times=times, start=start, end=end,
        fine_times=fine_times,
        **{f"path_{name.replace(' ', '_').replace('-', '_')}": path
           for name, path in paths.items()},
        **{f"det_{name.replace(' ', '_').replace('-', '_')}":
           np.array([np.linalg.det(matrix) for matrix in path])
           for name, path in fine_paths.items()},
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "interpolation.tex")
        save_tikz(
            save_path, axis_width=args.axis_width, axis_height=args.axis_height
        )
        write_prov_sidecar(save_path, args)
        print(f"Saved interpolation paths in {save_path}")

    if args.show_interactive:
        plt.show()
