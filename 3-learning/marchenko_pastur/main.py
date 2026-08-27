# Marchenko-Pastur: what the dimensional regime does to a spectrum
#
# The true covariance is the identity, so every one of its eigenvalues is 1.
# The eigenvalues of the sample covariance matrix are not: they spread over
# [(1-sqrt(c))^2, (1+sqrt(c))^2] with c = d/N, and that spread does not shrink
# when more data is added — it depends on c alone. Adding data at fixed c is
# not the same thing as adding data at fixed d.
#
# The figure is the illustration behind the remark "Ce que cette loi dit
# vraiment" of the learning chapter (ch:learning, subsec:learning-rmt), which
# is the claim the whole RMT correction rests on: the bias is deterministic and
# perfectly described, hence correctable — unlike the shrinkage of
# sec:context-covariance-regularized-estimation, which contracts the spectrum
# without knowing by how much.
#
# Three panels, one per concentration ratio. Each superposes the histogram of
# the pooled eigenvalues of n_trials sample covariance matrices on the
# Marchenko-Pastur density (eq:learning-marchenko-pastur). The dimension d is
# held fixed and N is derived from c, so the three panels differ by the sample
# support alone.
#
# Backend-free: the sampling and the eigenvalue decomposition go through
# hdrlib.core.backend, so the same script runs on numpy, torch, cupy or jax.
# The density itself is evaluated in numpy — it is a scalar formula on a
# plotting grid, not a computation on the data.

import os

import numpy as np
import matplotlib.pyplot as plt

from hdrlib.core.backend import (
    batched_eigh,
    get_backend_module,
    sample_standard_normal,
    to_numpy,
)
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.mc import add_mc_base_args, init_logging, make_mc_parser
from hdrlib.core.plot_style import apply_style
from hdrlib.core.exporter import save_tikz


def marchenko_pastur_density(grid, ratio):
    """Density of eq:learning-marchenko-pastur, zero outside its support.

    Valid for ``ratio < 1``. At ``ratio == 1`` the lower edge reaches the
    origin and the density diverges there like 1/sqrt(x); the value is finite
    everywhere the grid actually samples, so the formula is used as is and the
    plotting range is what keeps the picture readable.
    """
    lower = (1.0 - np.sqrt(ratio)) ** 2
    upper = (1.0 + np.sqrt(ratio)) ** 2
    density = np.zeros_like(grid)
    inside = (grid > lower) & (grid < upper)
    density[inside] = np.sqrt(
        (upper - grid[inside]) * (grid[inside] - lower)
    ) / (2.0 * np.pi * ratio * grid[inside])
    return density, lower, upper


def sample_eigenvalues(n_features, n_samples, n_trials, backend, seed):
    """Pooled eigenvalues of ``n_trials`` sample covariance matrices.

    The data of every trial is drawn in one call and the trials are stacked in
    the leading dimension, so a single batched eigendecomposition covers the
    whole Monte-Carlo. This is the shape ``batched_eigh`` is written for, and
    it is what makes the non-numpy backends worth anything here.
    """
    bm = get_backend_module(backend)
    data = sample_standard_normal(
        n_trials, [n_features, n_samples], backend, seed=seed
    )
    # One SCM per trial: (n_trials, d, N) -> (n_trials, d, d). The true
    # covariance is the identity, so no whitening is needed.
    scm = bm.matmul(data, _transpose(bm, data, backend)) / n_samples
    eigenvalues, _ = batched_eigh(backend, scm)
    return to_numpy(eigenvalues).reshape(-1)


def _transpose(bm, x, backend):
    """Swap the last two axes, whichever backend ``x`` lives on."""
    if hasattr(bm, "swapaxes"):
        return bm.swapaxes(x, -1, -2)
    return bm.transpose(x, -1, -2)  # torch


if __name__ == "__main__":
    parser = make_mc_parser(
        "Marchenko-Pastur law: histogram of the sample covariance eigenvalues "
        "against the theoretical density, for several concentration ratios."
    )
    add_mc_base_args(parser)
    parser.add_argument(
        "--n_features", type=int, default=200,
        help="Dimension d, held fixed across the panels. Large enough for the "
             "asymptotic density to be visible, small enough for the batched "
             "eigendecomposition to stay cheap.",
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.1, 0.5, 1.0],
        help="Concentration ratios c = d/N, one panel each. The number of "
             "samples of a panel is N = round(d / c), so only the sample "
             "support changes from one panel to the next.",
    )
    parser.add_argument(
        "--n_bins", type=int, default=80,
        help="Histogram bins per panel.",
    )
    parser.add_argument(
        "--axis_width", type=str, default="0.31\\textwidth",
        help="Width of a single panel in the exported PGFPlots figure. Set "
             "here rather than patched into the .tex afterwards, so that a "
             "re-sync into the dissertation does not undo it.",
    )
    parser.add_argument(
        "--axis_height", type=str, default="4.4cm",
        help="Height of a single panel in the exported PGFPlots figure.",
    )
    args = parser.parse_args()
    # The base MC parser defaults to 10 000 trials, which is far more than this
    # figure needs: every trial already contributes d eigenvalues to the
    # histogram, so a few hundred trials give a smooth curve.
    args.storage_path = args.export_path

    init_logging(args.backend)
    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)

    d = args.n_features
    fig, axes = plt.subplots(
        1, len(args.ratios), figsize=(3.0 * len(args.ratios), 3.0), sharey=False
    )
    axes = np.atleast_1d(axes)

    saved = {}
    for index, (ax, ratio) in enumerate(zip(axes, args.ratios)):
        n_samples = int(round(d / ratio))
        # Each panel gets its own seed, otherwise the three would share the
        # same underlying draw and the comparison would be between three views
        # of one realisation rather than three independent ones.
        eigenvalues = sample_eigenvalues(
            d, n_samples, args.n_trials, args.backend, args.seed + index
        )

        upper_plot = (1.0 + np.sqrt(ratio)) ** 2 * 1.15
        grid = np.linspace(1e-4, upper_plot, 600)
        density, lower, upper = marchenko_pastur_density(grid, ratio)

        ax.hist(
            eigenvalues, bins=args.n_bins, range=(0.0, upper_plot),
            density=True, color="C0", alpha=0.55, edgecolor="none",
            label="valeurs propres de la scm",
        )
        ax.plot(grid, density, color="C3", linewidth=1.6, label="loi de MP")
        # The true spectrum is a single point; it is worth drawing, because the
        # whole reading of the figure is the gap between it and the histogram.
        ax.axvline(1.0, color="k", linestyle="--", linewidth=1.0,
                   label=r"spectre vrai")
        ax.set_xlim(0.0, upper_plot)
        ax.set_xlabel(r"$\lambda$")
        if index == 0:
            ax.set_ylabel("densité")
        ax.set_title(rf"$c = {ratio:g}$")

        saved[f"eigenvalues_c{index}"] = eigenvalues
        saved[f"ratio_c{index}"] = ratio
        saved[f"n_samples_c{index}"] = n_samples
        print(
            f"c = {ratio:g}: d = {d}, N = {n_samples}, "
            f"support [{lower:.3f}, {upper:.3f}], "
            f"observed [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]"
        )

    # Legend below the panels rather than inside one of them: at export size an
    # inner legend either covers the histogram or spills out of the frame. It
    # is attached to an axis and not to the figure, since matplot2tikz exports
    # axis legends and silently drops figure ones.
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=3,
        frameon=False, fontsize=8,
    )
    fig.tight_layout()

    if args.export:
        np.savez(
            os.path.join(args.storage_path, "results.npz"),
            seed=args.seed, n_features=d, n_trials=args.n_trials,
            ratios=np.asarray(args.ratios), n_bins=args.n_bins,
            **saved,
        )
        save_path = os.path.join(args.storage_path, "marchenko.tex")
        save_tikz(
            save_path, axis_width=args.axis_width, axis_height=args.axis_height
        )
        write_prov_sidecar(save_path, args)
        print(f"Saved Marchenko-Pastur figure in {save_path}")

    if args.show_interactive:
        plt.show()
