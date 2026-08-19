# Three means of the same set of covariance matrices
#
# A cloud of 2x2 matrices is drawn around a common centre, then averaged in
# three ways: arithmetically, log-Euclidean-wise, and by the Fréchet mean of
# the affine-invariant metric. The left panel shows the ellipses, the right
# one the determinants, which is where the three differ most clearly.
#
# The cloud is generated *on the manifold* rather than by perturbing the
# entries: each matrix is the Riemannian exponential of a random tangent
# vector at the centre, so the set is symmetric around it in the geometry the
# Fréchet mean uses, and the centre is by construction what that mean should
# recover.
#
# The Fréchet mean has no closed form for more than two matrices and is
# computed by the hand-written Riemannian descent of
# hdrlib.core.estimation.frechet_mean_affine_invariant.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.estimation import frechet_mean_affine_invariant
from hdrlib.core.manifolds import HermitianPositiveDefinite, logm_psd, multiherm


def sample_cloud(center, dispersion, n_matrices, manifold, backend, seed):
    """Matrices spread around a centre along random geodesics.

    Draws a symmetric tangent vector of unit Riemannian norm at the centre,
    scales it by a random length of standard deviation ``dispersion``, and
    follows the geodesic. The result is a cloud whose spread is measured in
    the affine-invariant metric and not in the entries of the matrices.
    """
    rng = np.random.default_rng(seed)
    center_device = get_data_on_device(center, backend)
    n_features = center.shape[-1]

    cloud = []
    for _ in range(n_matrices):
        noise = rng.standard_normal((n_features, n_features))
        tangent = to_numpy(
            multiherm(get_data_on_device(noise, backend), backend)
        )
        tangent_device = get_data_on_device(tangent, backend)
        norm = float(manifold.norm(center_device, tangent_device))
        length = dispersion * abs(rng.standard_normal())
        tangent_device = get_data_on_device(
            length * tangent / norm, backend
        )
        cloud.append(to_numpy(manifold.exp(center_device, tangent_device)))
    return np.stack(cloud)


def log_euclidean_mean(covariances, backend):
    """expm of the arithmetic mean of the logarithms — the closed form."""
    logarithms = to_numpy(
        logm_psd(get_data_on_device(covariances, backend), backend)
    )
    values, vectors = np.linalg.eigh(logarithms.mean(axis=0))
    return vectors @ np.diag(np.exp(values)) @ vectors.T


def concentration_ellipse(shape, radius, n_points=200):
    """Curve {x : x^T shape^{-1} x = radius^2}, as a (2, n_points) array."""
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    values, vectors = np.linalg.eigh(shape)
    return vectors @ np.diag(np.sqrt(values)) @ circle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Arithmetic, log-Euclidean and Fréchet means of a cloud of covariances."
    )
    parser.add_argument(
        "--n_matrices", type=int, default=15,
        help="Number of matrices averaged. Kept small enough for the cloud to "
             "remain readable as a set of ellipses.",
    )
    parser.add_argument(
        "--dispersion", type=float, default=0.9,
        help="Standard deviation of the geodesic distance between a matrix of "
             "the cloud and its centre, in the affine-invariant metric.",
    )
    parser.add_argument(
        "--condition", type=float, default=4.0,
        help="Ratio of the eigenvalues of the centre of the cloud.",
    )
    parser.add_argument(
        "--radius", type=float, default=1.0,
        help="Radius of the drawn ellipses, in units of the Mahalanobis distance.",
    )
    parser.add_argument(
        "--iter_max", type=int, default=100,
        help="Maximum number of Riemannian gradient steps for the Fréchet mean.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-10,
        help="Stopping tolerance on the gradient norm of the Fréchet mean.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/riemann_moyennes",
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
    spread = np.sqrt(args.condition)
    center = np.diag([spread, 1.0 / spread])
    cloud = sample_cloud(
        center, args.dispersion, args.n_matrices, manifold, args.backend, args.seed
    )

    frechet, history = frechet_mean_affine_invariant(
        get_data_on_device(cloud, args.backend),
        iter_max=args.iter_max, tol=args.tol, backend_name=args.backend,
    )
    # Insertion order is drawing order, and the dashed log-Euclidean mean is
    # kept last so that it stays visible where it lands on the Fréchet one.
    means = {
        "arithmétique": cloud.mean(axis=0),
        "de Fréchet": to_numpy(frechet),
        "log-euclidienne": log_euclidean_mean(cloud, args.backend),
    }

    colors = {
        "arithmétique": "C1",
        "log-euclidienne": "C3",
        "de Fréchet": "C2",
    }
    # The log-Euclidean and Fréchet means are close enough to overlap on both
    # panels, which is itself worth seeing: one of the two is dashed so that
    # the superposition reads as a superposition and not as a missing curve.
    linestyles = {
        "arithmétique": "-",
        "log-euclidienne": "--",
        "de Fréchet": "-",
    }

    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4))

    ax = axes[0]
    for index, matrix in enumerate(cloud):
        curve = concentration_ellipse(matrix, args.radius)
        ax.plot(
            curve[0], curve[1], color="C0", linewidth=0.9, alpha=0.7, zorder=1,
            label="échantillon" if index == 0 else None,
        )
    for name, matrix in means.items():
        curve = concentration_ellipse(matrix, args.radius)
        ax.plot(
            curve[0], curve[1], color=colors[name], linewidth=1.6, zorder=3,
            linestyle=linestyles[name],
        )
    ax.set_aspect("equal")
    # Framed on a high quantile rather than the maximum: one very elongated
    # draw would otherwise shrink everything else to a dot. It is still drawn,
    # simply not entirely inside the frame.
    limit = 1.1 * np.quantile(
        np.abs(
            np.stack(
                [concentration_ellipse(matrix, args.radius) for matrix in cloud]
            )
        ),
        0.99,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("ellipses de concentration")

    # Determinants, sorted, with the three means as horizontal lines: the
    # arithmetic one sits above the cloud, the two others inside it.
    ax = axes[1]
    determinants = np.array([np.linalg.det(matrix) for matrix in cloud])
    ax.plot(
        np.arange(1, args.n_matrices + 1), np.sort(determinants),
        marker="o", markersize=3, linestyle="none", color="C0",
        label="échantillon",
    )
    for name, matrix in means.items():
        ax.axhline(
            np.linalg.det(matrix), color=colors[name], linewidth=1.4,
            linestyle=linestyles[name], label=name,
        )
    ax.set_yscale("log")
    ax.set_xlabel("matrices, triées par déterminant")
    ax.set_ylabel(r"$\det$")
    ax.set_title("déterminants")
    # Legend on this panel rather than above the grid: the ellipses fill their
    # own frame, and matplot2tikz exports axis legends but drops figure ones.
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()

    print(f"N = {args.n_matrices} matrices, dispersion {args.dispersion:g}, "
          f"Fréchet mean in {len(history['variance']) - 1} iterations "
          f"(gradient norm {history['gradient_norm'][-1]:.2e})")
    print(f"  geometric mean of the determinants: "
          f"{np.exp(np.log(determinants).mean()):.3f}")
    for name, matrix in means.items():
        distance = float(
            manifold.dist(
                get_data_on_device(matrix, args.backend),
                get_data_on_device(center, args.backend),
            )
        )
        print(f"  {name:16} det = {np.linalg.det(matrix):7.3f}   "
              f"distance to the centre = {distance:.3f}")

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed, n_matrices=args.n_matrices, dispersion=args.dispersion,
        condition=args.condition, radius=args.radius,
        center=center, cloud=cloud,
        variance=np.array(history["variance"]),
        gradient_norm=np.array(history["gradient_norm"]),
        **{f"mean_{name.replace(' ', '_').replace('-', '_').replace('é', 'e')}": matrix
           for name, matrix in means.items()},
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "moyennes.tex")
        save(save_path, axis_width=args.axis_width, axis_height=args.axis_height)
        write_prov_sidecar(save_path, args)
        print(f"Saved means in {save_path}")

    if args.show_interactive:
        plt.show()
