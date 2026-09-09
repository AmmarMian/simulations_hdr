# What ReEig does to the spectrum, and what it buys the backward pass.
#
# The CovPool layer of eq:spdnet-covpool is an empirical covariance estimated
# from n_pixels positions for n_filters channels. In the setting of
# rem:spdnet-covpool-regime that ratio is about three (n_filters = 8 x 32 = 256
# against n_pixels = 38 x 20 = 760), which is a *low* sampling regime: the
# smallest eigenvalues of such a matrix are far below those of the covariance
# it estimates, and it is those that the rest of the network has to survive.
#
# Two things are measured against the sampling ratio, both on the same
# simulated matrices:
#
#   (a) the spectrum itself, with the ReEig threshold drawn across it, which
#       says how much of the spectrum the layer actually rectifies;
#   (b) the largest entry of the Loewner matrix of prop:spdnet-diffm for the
#       LogEig that follows, which is the factor by which backpropagating
#       through the spectral layers multiplies the incoming error.
#
# The point of the second panel: that factor is 1/lambda_min, so it diverges as
# the sampling ratio drops, and a ReEig layer of threshold eps caps it at
# exactly 1/eps. This is the sense in which ReEig is a spectral regularisation
# of the same family as the shrinkage of the second part of the dissertation
# (rem:spdnet-reeig-retrecissement) — it pays a bias on the small eigenvalues
# to buy a bound on the gradient.
#
# Simulated data only; the same two measurements are run on the datasets of the
# chapter by real_data.py.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from hdrlib.core.mc import Progress
from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.plot_style import apply_style

from common import (
    decaying_covariance,
    resolve_device,
    sample_covpool,
    spectral_summary,
)


def sweep(args, device, dtype):
    """Measure the summary of common.py at every (decay, ratio) pair.

    Two axes rather than one, because the two of them drive the instability
    independently: the sampling ratio sets how far the empirical spectrum falls
    below the true one, and the decay sets where the true one already was. The
    condition numbers reported on the real datasets of the chapter (from
    9.1e5 on HDM05 to 1.2e7 on Rices90) are far beyond what a mild decay
    reaches at any ratio, so fixing the decay would answer a question the
    chapter is not asking.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    covariances_by_decay = {}
    records = []
    # One step per (decay, ratio) cell: that is the grid the figure draws, and
    # n_trials covariances are sampled inside each of them.
    progress = Progress(
        args.storage_path, len(args.decays) * len(args.ratios),
        description="Decay x ratio", unit="cells",
    )
    for decay in args.decays:
        true_covariance = decaying_covariance(
            args.n_filters, decay, device=device, dtype=dtype
        )
        covariances_by_decay[decay] = true_covariance
        for ratio in args.ratios:
            n_pixels = max(int(round(ratio * args.n_filters)), 2)
            covariances = sample_covpool(
                true_covariance, n_pixels, args.n_trials, generator, device, dtype
            )
            summary = spectral_summary(covariances, args.eps)
            summary["decay"] = decay
            summary["ratio"] = ratio
            summary["n_pixels"] = n_pixels
            records.append(summary)
            progress.step()
    progress.done()
    return covariances_by_decay, records


def median(values):
    return float(torch.median(values).cpu())


def band(values, low=0.05, high=0.95):
    quantiles = torch.quantile(
        values.cpu(), torch.tensor([low, high], dtype=values.dtype)
    )
    return float(quantiles[0]), float(quantiles[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Effect of the ReEig threshold on the spectrum of a CovPool matrix, "
        "and on the Loewner factor of the backward pass."
    )
    parser.add_argument(
        "--n_filters", type=int, default=256,
        help="Number of channels, i.e. the size of the SPD matrix. Default is "
             "the 8 x 32 = 256 of the SRCNet architecture.",
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+",
        default=[0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
        help="Sampling ratios n_pixels / n_filters to sweep. The architecture "
             "of the chapter sits at about 3.",
    )
    parser.add_argument(
        "--spectra_at", type=float, nargs="+", default=[1.0, 3.0, 20.0],
        help="Ratios whose full spectrum is drawn on the left panel. Kept to "
             "three so that the panel stays readable at 355 pt.",
    )
    parser.add_argument(
        "--decays", type=float, nargs="+", default=[1e2, 1e4, 1e6],
        help="Ratios between the largest and the smallest eigenvalue of the "
             "true covariance, which decays geometrically in between. The "
             "chapter's real data sit at the top of this range: the reported "
             "condition numbers run from 9.1e5 to 1.2e7.",
    )
    parser.add_argument(
        "--spectra_decay", type=float, default=1e4,
        help="Which of the decays the left panel draws the spectra for.",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-4,
        help="ReEig rectification threshold, the default of the layer.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=200,
        help="Number of CovPool matrices drawn at each ratio.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/reeig_spectrum",
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
        help="Width of a single panel in the exported PGFPlots figure.",
    )
    parser.add_argument(
        "--axis_height", type=str, default="4.6cm",
        help="Height of a single panel in the exported PGFPlots figure.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: cpu or cuda. MPS is refused, see common.py.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    args = parser.parse_args()

    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)

    device = resolve_device(args.device)
    dtype = torch.float64

    covariances_by_decay, records = sweep(args, device, dtype)

    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4))

    # ---- Panel (a): the spectra --------------------------------------------
    # One decay only, at several sampling ratios: the panel is about the gap
    # the sampling opens between the true spectrum and the estimated one, and
    # about where the ReEig threshold falls in that gap.
    ax = axes[0]
    index = np.arange(1, args.n_filters + 1)
    true_spectrum = (
        torch.linalg.eigvalsh(covariances_by_decay[args.spectra_decay])
        .flip(0).cpu().numpy()
    )
    ax.plot(
        index, true_spectrum, color="k", linewidth=1.2, linestyle=":",
        label="covariance vraie", zorder=4,
    )
    for position, ratio in enumerate(args.spectra_at):
        drawn = [
            record for record in records
            if record["ratio"] == ratio and record["decay"] == args.spectra_decay
        ]
        if not drawn:
            continue
        spectra = drawn[0]["eigenvalues"].flip(-1).cpu().numpy()
        ax.plot(
            index, np.median(spectra, axis=0), color=f"C{position}", linewidth=1.2,
            label=rf"$N_{{pix}}/N_{{filtre}} = {ratio:g}$", zorder=3,
        )
    ax.axhline(
        args.eps, color="C3", linewidth=1.0, linestyle="--", zorder=2,
        label=rf"seuil $\varepsilon = {args.eps:g}$",
    )
    ax.set_yscale("log")
    ax.set_xlabel("rang de la valeur propre")
    ax.set_ylabel("valeur propre")
    ax.set_title(rf"décroissance $10^{{{int(round(np.log10(args.spectra_decay)))}}}$")
    ax.legend()

    # ---- Panel (b): what the backward pass pays ----------------------------
    # One curve per decay without ReEig, all of them flattened onto the same
    # 1/eps ceiling once the layer is applied. Drawing the rectified curves as
    # dashed rather than as a second family of colours keeps the panel readable
    # at 355 pt.
    ax = axes[1]
    all_ratios = sorted({record["ratio"] for record in records})

    # Below a ratio of one the CovPool matrix is rank deficient — the centring
    # of eq:spdnet-covpool caps its rank at n_pixels - 1 — so lambda_min is zero
    # up to rounding, LogEig is undefined and the Loewner factor has no value to
    # plot. Those ratios are shaded rather than drawn, because "no finite value"
    # is the finding, not a large value.
    singular = [
        record["ratio"] for record in records if median(record["lambda_min"]) <= 0
    ]
    if singular:
        ax.axvspan(
            min(all_ratios), max(singular) * 1.05,
            color="0.85", linewidth=0, zorder=0,
        )

    for position, decay in enumerate(args.decays):
        regular = [
            record for record in records
            if record["decay"] == decay and median(record["lambda_min"]) > 0
        ]
        exponent = int(round(np.log10(decay)))
        ax.plot(
            [record["ratio"] for record in regular],
            [median(record["loewner_max"]) for record in regular],
            color=f"C{position}", linewidth=1.4, marker="o", markersize=3, zorder=3,
            label=rf"$10^{{{exponent}}}$, sans \textsc{{reeig}}",
        )
        ax.plot(
            [record["ratio"] for record in regular],
            [median(record["loewner_max_reeig"]) for record in regular],
            color=f"C{position}", linewidth=1.2, linestyle="--", zorder=3,
            label=rf"$10^{{{exponent}}}$, avec \textsc{{reeig}}",
        )
    ax.axhline(
        1.0 / args.eps, color="C3", linewidth=1.0, linestyle="--", zorder=2,
        label=r"borne $1/\varepsilon$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{pix}/N_{filtre}$")
    ax.set_ylabel(r"$\max_{ij}|\mathbf{G}_{ij}|$ de \textsc{logeig}")
    ax.legend()

    fig.tight_layout()

    # ---- A short digest on stdout, which is what qanat keeps ---------------
    print(f"n_filters = {args.n_filters}, eps = {args.eps:g}, "
          f"n_trials = {args.n_trials}")
    print(f"{'decay':>9s} {'ratio':>7s} {'n_pix':>7s} {'lambda_min':>12s} "
          f"{'cond':>11s} {'% ecretees':>11s} {'Loewner':>11s} {'+ReEig':>11s}")
    for record in records:
        lambda_min = median(record["lambda_min"])
        prefix = (
            f"{record['decay']:9.0e} {record['ratio']:7.2f} "
            f"{record['n_pixels']:7d} {lambda_min:12.3e} "
        )
        # A non-positive lambda_min means the matrix is singular; the condition
        # number and the Loewner factor are then meaningless rather than large,
        # and printing a number there would invite reading one.
        if lambda_min <= 0:
            print(
                prefix + f"{'singuliere':>11s} "
                f"{100 * median(record['fraction_clamped']):10.1f}% "
                f"{'non defini':>11s} {median(record['loewner_max_reeig']):11.3e}"
            )
        else:
            print(
                prefix + f"{median(record['condition']):11.3e} "
                f"{100 * median(record['fraction_clamped']):10.1f}% "
                f"{median(record['loewner_max']):11.3e} "
                f"{median(record['loewner_max_reeig']):11.3e}"
            )

    # Raw measurements next to the figure, so that a qanat action can redraw
    # without re-running the sweep. Only the medians and the band are kept: the
    # per-trial spectra are a few hundred megabytes at the default settings and
    # nothing downstream reads them.
    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        decays=np.array([record["decay"] for record in records]),
        ratios=np.array([record["ratio"] for record in records]),
        n_pixels=np.array([record["n_pixels"] for record in records]),
        lambda_min=np.array([median(record["lambda_min"]) for record in records]),
        condition=np.array([median(record["condition"]) for record in records]),
        fraction_clamped=np.array(
            [median(record["fraction_clamped"]) for record in records]
        ),
        loewner_max=np.array([median(record["loewner_max"]) for record in records]),
        loewner_max_reeig=np.array(
            [median(record["loewner_max_reeig"]) for record in records]
        ),
        spectra_median=np.stack(
            [
                np.median(record["eigenvalues"].flip(-1).cpu().numpy(), axis=0)
                for record in records
            ]
        ),
        eps=args.eps,
        n_filters=args.n_filters,
    )

    if args.export:
        figure_path = os.path.join(args.storage_path, "reeig_spectrum.tex")
        save_tikz(
            figure_path,
            axis_width=args.axis_width,
            axis_height=args.axis_height,
        )
        write_prov_sidecar(figure_path, args)

    if args.show_interactive:
        plt.show()
