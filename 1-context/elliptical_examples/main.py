# Isodensity contours and draws for several elliptical distributions
#
# All panels share the same scatter matrix, so the elliptical symmetry is
# common to all of them: only the law of the modular variate — that is, the
# density generator — changes, which is what the tails show.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.elliptical import (
    GaussianDistribution,
    StudentTDistribution,
    KDistribution,
    GeneralizedGaussianDistribution,
    sample_elliptical,
    isodensity_ellipse,
)


def build_distributions(names, n_features, dof_student, dof_k, shape_gengauss, backend):
    """Instantiate the requested distributions, all normalised to Cov = Xi."""
    factories = {
        "gaussian": lambda: GaussianDistribution(n_features, backend_name=backend),
        "student": lambda: StudentTDistribution(
            n_features, dof=dof_student, backend_name=backend
        ),
        "k": lambda: KDistribution(n_features, dof=dof_k, backend_name=backend),
        "gengauss": lambda: GeneralizedGaussianDistribution(
            n_features, shape=shape_gengauss, backend_name=backend
        ),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(f"Unknown distribution(s): {sorted(unknown)}")
    return [factories[name]() for name in names]


def panel_title(name, distribution):
    """Math-only title: matplotlib's mathtext does not take LaTeX accents."""
    if name == "gaussian":
        return r"$\mathcal{N}$"
    if name == "student":
        return rf"$t,\ \nu = {distribution.dof:.0f}$"
    if name == "k":
        return rf"$K,\ \nu = {distribution.dof:.0f}$"
    return rf"$\mathcal{{GG}},\ s = {distribution.shape}$"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Isodensity contours of elliptical distributions sharing a scatter matrix."
    )
    parser.add_argument(
        "--distributions", type=str, nargs="+",
        default=["gaussian", "student", "k", "gengauss"],
        help="Distributions to show, one panel each. The Gaussian acts as the "
             "reference against which the tails are read.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of samples drawn per panel. Drawn as hollow markers so the "
             "isodensity contours stay readable underneath.",
    )
    parser.add_argument(
        "--rho", type=float, default=0.8,
        help="Correlation coefficient of the shared scatter matrix.",
    )
    parser.add_argument(
        "--dof_student", type=float, default=3.0,
        help="Degrees of freedom of the t distribution (>2 for a finite covariance).",
    )
    parser.add_argument(
        "--dof_k", type=float, default=2.0, help="Texture shape of the K distribution."
    )
    parser.add_argument(
        "--shape_gengauss", type=float, default=0.5,
        help="Exponent s of the generalized Gaussian (s<1 gives heavier tails).",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="outputs/elliptical_examples",
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
    parser.add_argument(
        "--backend", type=str, default="numpy",
        help="Compute backend for the draws (numpy, torch-cpu, torch-mps, ...).",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed generation base seed")
    args = parser.parse_args()

    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)

    # Constant(s)
    d = 2
    mean = np.zeros(d)
    probabilities = np.array([0.5, 0.9, 0.99])

    # A single scatter matrix shared by every panel
    scatter = np.array([[1.0, args.rho], [args.rho, 1.0]])

    distributions = build_distributions(
        args.distributions, d, args.dof_student, args.dof_k, args.shape_gengauss,
        args.backend,
    )

    scatter_device = get_data_on_device(scatter, args.backend)
    mean_device = get_data_on_device(mean, args.backend)
    samples = [
        to_numpy(
            sample_elliptical(
                args.n_samples, mean_device, scatter_device, distribution,
                seed=args.seed + offset,
            )
        )
        for offset, distribution in enumerate(distributions)
    ]
    contours = [
        [isodensity_ellipse(scatter, distribution, p) for p in probabilities]
        for distribution in distributions
    ]

    # Common extent, driven by the widest outer contour so that the panels stay
    # comparable and the framing does not move with --seed.
    limit = 1.1 * max(
        np.abs(curves[-1]).max() for curves in contours
    )

    # A grid rather than a single row: squarer figures sit better both in the
    # dissertation's text width and on the docs page.
    n_panels = len(distributions)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, name, distribution, data, curves) in enumerate(
        zip(axes, args.distributions, distributions, samples, contours)
    ):
        ax.scatter(
            data[:, 0], data[:, 1],
            marker="o", s=14, facecolors="none", edgecolors="C0", linewidths=0.7,
            zorder=3,
        )
        for curve in curves:
            ax.plot(curve[0], curve[1], color="C1", linewidth=1.2, zorder=2)

        ax.set_aspect("equal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel(r"$x_1$")
        if i % n_cols == 0:
            ax.set_ylabel(r"$x_2$")
        ax.set_title(panel_title(name, distribution))

    # Hide any leftover cell when the panel count does not fill the grid
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()

    # Save results
    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed,
        n_samples=args.n_samples,
        rho=args.rho,
        dof_student=args.dof_student,
        dof_k=args.dof_k,
        shape_gengauss=args.shape_gengauss,
        probabilities=probabilities,
        scatter=scatter,
        names=np.array(args.distributions),
        samples=np.stack(samples),
        contours=np.stack([np.stack(curves) for curves in contours]),
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "elliptical_tails.tex")
        save(save_path)
        write_prov_sidecar(save_path, args)
        print(f"Saved elliptical examples in {save_path}")

    if args.show_interactive:
        plt.show()
