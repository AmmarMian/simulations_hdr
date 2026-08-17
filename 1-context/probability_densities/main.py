# Isodensity contours of the bivariate Gaussian for three covariance regimes

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar


def gaussian_pdf(grid_x, grid_y, mean, cov):
    """Evaluate the bivariate Gaussian d.d.p on a meshgrid."""
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=0) - mean[:, None]
    quad = np.einsum("ik,ij,jk->k", points, np.linalg.inv(cov), points)
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return (norm * np.exp(-0.5 * quad)).reshape(grid_x.shape)


def mahalanobis_levels(cov, probabilities):
    """d.d.p values whose contours enclose the given probability masses.

    For a bivariate Gaussian the squared Mahalanobis distance follows a
    chi-squared law with 2 degrees of freedom, which gives the radii
    directly.
    """
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    radii_squared = chi2.ppf(probabilities, df=2)
    return np.sort(norm * np.exp(-0.5 * radii_squared))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Isodensity contours of the bivariate Gaussian for three covariance regimes."
    )
    parser.add_argument(
        "--n_samples", type=int, default=300, help="Number of samples drawn per regime."
    )
    parser.add_argument(
        "--rho", type=float, default=0.8, help="Correlation coefficient of the correlated regime."
    )
    parser.add_argument(
        "--condition", type=float, default=50.0,
        help="Condition number of the ill-conditioned regime.",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="outputs/gaussian_isocontours",
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

    # Constant(s)
    d = 2
    mean = np.zeros(d)
    probabilities = np.array([0.5, 0.9, 0.99])

    # Covariance matrices: identity, correlated, ill-conditioned
    cov_id = np.eye(d)

    cov_corr = np.array([[1.0, args.rho], [args.rho, 1.0]])

    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    eigenvalues = np.array([1.0, 1.0 / args.condition])
    cov_ill = rotation @ np.diag(eigenvalues) @ rotation.T

    covariances = [cov_id, cov_corr, cov_ill]
    titles = [
        r"$\mathbf{\Sigma} = \mathbf{I}_2$",
        rf"$\rho = {args.rho}$",
        rf"$k(\mathbf{{\Sigma}}) = {args.condition:.0f}$",
    ]

    # Sampling
    samples = [
        rng.multivariate_normal(mean, cov, size=args.n_samples) for cov in covariances
    ]

    # Common extent so the three panels are visually comparable
    limit = 1.15 * max(np.abs(np.concatenate(samples)).max(), 3.0)
    axis_grid = np.linspace(-limit, limit, 400)
    grid_x, grid_y = np.meshgrid(axis_grid, axis_grid)

    # Single figure with all three regimes
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for i, (ax, cov, data, title) in enumerate(zip(axes, covariances, samples, titles)):
        density = gaussian_pdf(grid_x, grid_y, mean, cov)
        ax.scatter(data[:, 0], data[:, 1], s=6, alpha=0.35, color="C0", linewidths=0)
        ax.contour(
            grid_x,
            grid_y,
            density,
            levels=mahalanobis_levels(cov, probabilities),
            colors="C1",
            linewidths=1.2,
        )

        # Principal axes, scaled by the square root of the eigenvalues
        eigvals, eigvecs = np.linalg.eigh(cov)
        for eigval, eigvec in zip(eigvals, eigvecs.T):
            axis_end = np.sqrt(eigval) * eigvec
            ax.plot(
                [-axis_end[0], axis_end[0]],
                [-axis_end[1], axis_end[1]],
                color="C3",
                linewidth=1.0,
                linestyle="--",
            )

        ax.set_aspect("equal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel(r"$x_1$")
        if i == 0:
            ax.set_ylabel(r"$x_2$")
        ax.set_title(title)

    fig.tight_layout()

    # Save results
    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed,
        n_samples=args.n_samples,
        rho=args.rho,
        condition=args.condition,
        probabilities=probabilities,
        covariances=np.stack(covariances),
        samples=np.stack(samples),
        titles=np.array(titles),
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "gaussian_isocontours.tex")
        save(save_path)
        write_prov_sidecar(save_path, args)
        print(f"Saved isocontours in {save_path}")

    if args.show_interactive:
        plt.show()
