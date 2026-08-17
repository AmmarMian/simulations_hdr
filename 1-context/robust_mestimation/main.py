# A single estimation, seen in the plane
#
# One panel per elliptical model, all sharing the same true shape matrix. Each
# shows one draw of the data together with the concentration ellipse of three
# estimators: the SCM, the maximum-likelihood M-estimator of that model, and
# Tyler's distribution-free estimator.
#
# The fixed-point engine and Tyler's estimator come from hdrlib.core.estimation
# unchanged; each model supplies its own weight function through the public
# m_estimator_function hook.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.estimation import (
    SCMEstimator,
    TylerEstimator,
    fixed_point_m_estimation_centered,
)
from hdrlib.core.elliptical import (
    GaussianDistribution,
    StudentTDistribution,
    KDistribution,
    GeneralizedGaussianDistribution,
    sample_elliptical,
)


def build_distributions(names, n_features, dof_student, dof_k, shape_gengauss, backend):
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
        return rf"$t,\ \nu = {distribution.dof:g}$"
    if name == "k":
        return rf"$K,\ \nu = {distribution.dof:g}$"
    return rf"$\mathcal{{GG}},\ s = {distribution.shape}$"


def normalize_shape(matrix, n_features):
    """Normalise by the trace so only the shape is compared."""
    return n_features * matrix / np.trace(matrix)


def concentration_ellipse(shape, radius, n_points=300):
    """Ellipse {x : x^T shape^{-1} x = radius^2}, as a (2, n_points) array."""
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    return np.linalg.cholesky(shape) @ circle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Concentration ellipses of the SCM, the model MLE and Tyler's estimator."
    )
    parser.add_argument(
        "--distributions", type=str, nargs="+",
        default=["gaussian", "student", "k", "gengauss"],
        help="Models to show, one panel each.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of observations of the single estimation shown. Kept small "
             "on purpose: with a large support every estimator is accurate and "
             "the ellipses become indistinguishable.",
    )
    parser.add_argument(
        "--rho", type=float, default=0.8,
        help="Correlation of the shared true shape matrix.",
    )
    parser.add_argument(
        "--dof_student", type=float, default=2.1,
        help="Degrees of freedom of the t model. Just above 2, where the "
             "covariance still exists but the tails are very heavy.",
    )
    parser.add_argument(
        "--dof_k", type=float, default=0.1,
        help="Texture shape of the K model; the smaller, the heavier.",
    )
    parser.add_argument(
        "--shape_gengauss", type=float, default=0.15,
        help="Exponent s of the generalized Gaussian; s<1 gives heavier tails.",
    )
    parser.add_argument(
        "--iter_max", type=int, default=100, help="Fixed-point iterations."
    )
    parser.add_argument("--tol", type=float, default=1e-8, help="Fixed-point tolerance.")
    parser.add_argument(
        "--storage_path", type=str, default="outputs/robust_mestimation",
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
    mean = np.zeros(d)
    # Radius of the drawn ellipses, in units of the Mahalanobis distance
    radius = 2.0

    shape_true = np.array([[1.0, args.rho], [args.rho, 1.0]])
    shape_true = normalize_shape(shape_true, d)

    distributions = build_distributions(
        args.distributions, d, args.dof_student, args.dof_k, args.shape_gengauss,
        args.backend,
    )
    mean_device = get_data_on_device(mean, args.backend)
    shape_device = get_data_on_device(shape_true, args.backend)

    samples, estimates = [], []
    for offset, distribution in enumerate(distributions):
        data = to_numpy(
            sample_elliptical(
                args.n_samples, mean_device, shape_device, distribution,
                seed=args.seed + offset,
            )
        )
        samples.append(data)

        scm = np.asarray(SCMEstimator().compute(data))
        tyler = np.asarray(
            TylerEstimator(
                normalization="trace", iter_max=args.iter_max, tol=args.tol
            ).compute(data)
        )
        # The engine is reused as-is; only the weight changes, through the
        # public m_estimator_function hook.
        #
        # No normalisation during the iterations, unlike Tyler: the scale of a
        # genuine MLE is identifiable, so renormalising at each step would move
        # the fixed point and bias the estimate. The result is trace-normalised
        # afterwards, only to compare shapes.
        mle = np.asarray(
            fixed_point_m_estimation_centered(
                data,
                m_estimator_function=distribution.weight_function,
                iter_max=args.iter_max,
                tol=args.tol,
                normalization=None,
            )
        )
        estimates.append({
            "scm": normalize_shape(scm, d),
            "mle": normalize_shape(mle, d),
            "tyler": normalize_shape(tyler, d),
        })

    styles = {
        "true": dict(color="C7", linestyle="--", label=r"vraie $\xi$"),
        "scm": dict(color="C1", linestyle="-", label="scm"),
        "mle": dict(color="C2", linestyle="-", label="mle"),
        "tyler": dict(color="C3", linestyle="-", label="Tyler"),
    }

    # Frame on a high quantile rather than the maximum: a single extreme draw
    # would otherwise shrink the cloud to a dot. Outliers beyond the frame are
    # still what drags the SCM ellipse, they are simply not all drawn.
    limit = 1.1 * max(np.quantile(np.abs(data), 0.995) for data in samples)

    n_panels = len(distributions)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (ax, name, distribution, data, estimate) in enumerate(
        zip(axes, args.distributions, distributions, samples, estimates)
    ):
        ax.scatter(
            data[:, 0], data[:, 1],
            marker="o", s=10, facecolors="none", edgecolors="C0", linewidths=0.5,
            zorder=2,
        )
        for key in ("true", "scm", "mle", "tyler"):
            shape = shape_true if key == "true" else estimate[key]
            curve = concentration_ellipse(shape, radius)
            style = dict(styles[key])
            label = style.pop("label")
            label = label if i == 0 else None
            ax.plot(curve[0], curve[1], linewidth=1.3, zorder=3, label=label, **style)

        ax.set_aspect("equal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel(r"$x_1$")
        if i % n_cols == 0:
            ax.set_ylabel(r"$x_2$")
        ax.set_title(panel_title(name, distribution))

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Legend outside the grid, below it. It is attached to a single axis
    # rather than to the figure because matplot2tikz exports axis legends but
    # silently drops figure-level ones.
    axes[0].legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.14),
        ncol=len(styles), frameon=False, fontsize=9,
    )
    fig.tight_layout()

    # Errors, printed and stored, so the visual reading can be checked
    print(f"Shape estimation error (Frobenius), N = {args.n_samples}:")
    for name, estimate in zip(args.distributions, estimates):
        errors = {
            key: np.linalg.norm(value - shape_true, ord="fro")
            for key, value in estimate.items()
        }
        print(
            f"  {name:9} "
            + "  ".join(f"{key}={value:.3f}" for key, value in errors.items())
        )

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed, n_samples=args.n_samples, rho=args.rho, radius=radius,
        dof_student=args.dof_student, dof_k=args.dof_k,
        shape_gengauss=args.shape_gengauss,
        names=np.array(args.distributions),
        shape_true=shape_true,
        samples=np.stack(samples),
        **{
            f"{key}_{name}": estimate[key]
            for name, estimate in zip(args.distributions, estimates)
            for key in estimate
        },
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "scmvstyler.tex")
        save(save_path)
        write_prov_sidecar(save_path, args)
        print(f"Saved ellipses in {save_path}")

    if args.show_interactive:
        plt.show()
