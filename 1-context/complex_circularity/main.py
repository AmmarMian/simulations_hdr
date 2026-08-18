# What the covariance of a complex vector does not say
#
# One panel per value of the pseudo-covariance, all sharing the *same*
# covariance. The dashed circle is what the covariance alone predicts; the
# solid ellipse is the actual concentration curve. They coincide only in the
# first panel, where the pseudo-covariance vanishes — that is, only under
# circularity.
#
# Everything is done in the scalar case d = 1, where the whole second-order
# structure is two numbers: the real variance Gamma and the complex
# pseudo-variance C. Picinbono's admissibility condition then reads
# |C| <= Gamma, and the panels sweep that segment.

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from matplot2tikz import save
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import write_prov_sidecar


def real_covariance(gamma, pseudo):
    """Real 2x2 covariance of (Re z, Im z) from the pair (Gamma, C).

    Inverts the block relations of the dissertation: with
    Gamma = E{z z*} and C = E{z z}, one has
    Gamma_x = Re(Gamma + C)/2, Gamma_y = Re(Gamma - C)/2 and
    Gamma_xy = Im(C - Gamma)/2, which is Im(C)/2 for a real Gamma.
    """
    var_x = 0.5 * (gamma + pseudo.real)
    var_y = 0.5 * (gamma - pseudo.real)
    cov_xy = 0.5 * pseudo.imag
    return np.array([[var_x, cov_xy], [cov_xy, var_y]])


def is_admissible(gamma, pseudo):
    """Picinbono's condition, which reads |C| <= Gamma in the scalar case.

    The error variance of the widely linear predictor of z* from z is
    P = Gamma - |C|^2 / Gamma, and it must be non-negative.
    """
    return gamma - abs(pseudo) ** 2 / gamma >= -1e-12


def concentration_ellipse(covariance, probability, n_points=400):
    """Curve {v : v^T Sigma^-1 v = q} enclosing a given probability mass.

    For a bivariate Gaussian the quadratic form is chi-squared with 2 degrees
    of freedom, whose quantile is available in closed form — which avoids a
    scipy dependency for a single number.
    """
    radius = np.sqrt(-2.0 * np.log(1.0 - probability))
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    # A degenerate covariance (|C| = Gamma) is only positive *semi*-definite,
    # so the Cholesky factor is taken through the eigendecomposition.
    values, vectors = np.linalg.eigh(covariance)
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))) @ circle


def panel_title(rho, phase):
    if rho == 0:
        return r"$C = 0$"
    return rf"$|C|/\Gamma = {rho:g},\ \arg C = {phase / np.pi:g}\pi$"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Same covariance, four pseudo-covariances: circularity seen in the plane."
    )
    parser.add_argument(
        "--rho", type=float, nargs="+", default=[0.0, 0.5, 0.5, 0.9],
        help="Moduli |C|/Gamma of the pseudo-covariance, one panel each. Must "
             "lie in [0,1]: 0 is circular, 1 is a degenerate (real) variable.",
    )
    parser.add_argument(
        "--phase", type=float, nargs="+", default=[0.0, 0.0, 0.3333, 0.3333],
        help="Arguments of the pseudo-covariance, in units of pi, one per rho.",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="The covariance, shared by every panel. It is what the panels hold "
             "fixed, so that only the pseudo-covariance distinguishes them.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=600,
        help="Observations drawn per panel.",
    )
    parser.add_argument(
        "--probability", type=float, default=0.9,
        help="Probability mass enclosed by the drawn concentration curves.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/complex_circularity",
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
    parser.add_argument("--seed", type=int, default=42, help="random seed generation base seed")
    args = parser.parse_args()

    if len(args.rho) != len(args.phase):
        raise ValueError("--rho and --phase must have the same length")

    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)

    gamma = args.gamma
    phases = [p * np.pi for p in args.phase]
    pseudos = [rho * gamma * np.exp(1j * phase) for rho, phase in zip(args.rho, phases)]

    for rho, pseudo in zip(args.rho, pseudos):
        if not is_admissible(gamma, pseudo):
            raise ValueError(
                f"|C|/Gamma = {rho} violates Picinbono's condition |C| <= Gamma"
            )

    # The reference a reader would draw from the covariance alone: the circular
    # variable of the same Gamma, whose real covariance is (Gamma/2) I.
    reference = concentration_ellipse(0.5 * gamma * np.eye(2), args.probability)

    samples, ellipses, empirical = [], [], []
    for offset, pseudo in enumerate(pseudos):
        covariance = real_covariance(gamma, pseudo)
        rng = np.random.default_rng(args.seed + offset)
        values, vectors = np.linalg.eigh(covariance)
        factor = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
        data = (factor @ rng.standard_normal((2, args.n_samples))).T
        z = data[:, 0] + 1j * data[:, 1]

        samples.append(data)
        ellipses.append(concentration_ellipse(covariance, args.probability))
        empirical.append((np.mean(np.abs(z) ** 2), np.mean(z * z)))

    # A 2-column grid rather than a single row: the dissertation's text block
    # is narrow, and four panels side by side overflow it.
    n_panels = len(pseudos)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    limit = 1.15 * max(np.quantile(np.abs(data), 0.999) for data in samples)

    for index, (ax, rho, phase, data, ellipse) in enumerate(
        zip(axes, args.rho, phases, samples, ellipses)
    ):
        ax.scatter(
            data[:, 0], data[:, 1],
            marker="o", s=6, facecolors="none", edgecolors="C0", linewidths=0.4,
            zorder=2,
        )
        ax.plot(
            reference[0], reference[1],
            color="C7", linestyle="--", linewidth=1.2, zorder=3,
            label="prédit par $\\Gamma$ seule" if index == 0 else None,
        )
        ax.plot(
            ellipse[0], ellipse[1],
            color="C1", linestyle="-", linewidth=1.4, zorder=4,
            label="concentration réelle" if index == 0 else None,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        if index // n_cols == n_rows - 1:
            ax.set_xlabel(r"$\mathrm{Re}\,z$")
        if index % n_cols == 0:
            ax.set_ylabel(r"$\mathrm{Im}\,z$")
        ax.set_title(panel_title(rho, phase))

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Legend attached to an axis, not to the figure: matplot2tikz exports the
    # former and silently drops the latter.
    axes[0].legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.14),
        ncol=2, frameon=False, fontsize=9,
    )
    fig.tight_layout()

    print(f"Gamma = {gamma:g}, shared by every panel; N = {args.n_samples}")
    for rho, phase, (gamma_hat, pseudo_hat) in zip(args.rho, phases, empirical):
        print(
            f"  |C|/Gamma={rho:<4g} arg C={phase / np.pi:<7.4g}pi  "
            f"Gamma_hat={gamma_hat:.3f}  "
            f"C_hat={abs(pseudo_hat):.3f} exp(j{np.angle(pseudo_hat) / np.pi:.3f}pi)"
        )

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        seed=args.seed, gamma=gamma, n_samples=args.n_samples,
        probability=args.probability,
        rho=np.array(args.rho), phase=np.array(phases),
        samples=np.stack(samples),
        ellipses=np.stack(ellipses),
        reference=reference,
    )

    if args.export:
        save_path = os.path.join(args.storage_path, "circularity.tex")
        save(save_path)
        write_prov_sidecar(save_path, args)
        print(f"Saved circularity panels in {save_path}")

    if args.show_interactive:
        plt.show()
