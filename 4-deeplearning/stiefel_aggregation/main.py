# How far apart are the two aggregations of sec:spdnet-federe-agregation?
#
# prop:spdnet-federe-equivalence states that projavg (eq:spdnet-projavg) and
# rlavg (eq:spdnet-rlavg) coincide up to O(eps^2) when the local weights stay
# within O(eps) of the global iterate. The chapter currently supports that with
# EEG validation curves that lie on top of each other, which shows the two agree
# but says nothing about the *order* at which they do.
#
# This measures the order directly, with no data and no training: draw K local
# weights at geodesic distance eps from a base point of the Stiefel manifold,
# aggregate both ways, and look at ||projavg - rlavg||_F as eps shrinks.
#
# The measured order is three, not two — the proposition is true but
# conservative. Swept over (d0, d1, K) here rather than asserted from one
# geometry, since a single configuration cannot tell an exponent from a
# coincidence.
#
# Reuses stiefel_projection_polar and stiefel_projection_tangent_orthogonal of
# yetanotherspdnet, which are exactly the polarf and Lift of def:spdnet-lift.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.mc import Progress
from hdrlib.core.plot_style import apply_style

from yetanotherspdnet.functions.stiefel import (
    stiefel_projection_polar,
    stiefel_projection_tangent_orthogonal,
)
from yetanotherspdnet.random.stiefel import random_stiefel


def aggregate_both_ways(base, locals_):
    """projavg and rlavg of the same local weights, at the same base point."""
    projavg = stiefel_projection_polar(locals_.mean(dim=0))
    lifted = torch.stack(
        [stiefel_projection_tangent_orthogonal(local - base, base) for local in locals_]
    )
    rlavg = stiefel_projection_polar(base + lifted.mean(dim=0))
    return projavg, rlavg


def local_weights(base, dispersion, n_clients, generator):
    """K points of the manifold at distance ``dispersion`` from ``base``.

    Each is obtained by retracting a tangent vector of norm exactly
    ``dispersion``, so the dispersion of the clients is a controlled quantity
    and not the by-product of a random draw — which is what lets an exponent be
    read off the result.
    """
    weights = []
    for _ in range(n_clients):
        ambient = torch.randn(
            base.shape, generator=generator, device=base.device, dtype=base.dtype
        )
        tangent = stiefel_projection_tangent_orthogonal(ambient, base)
        tangent = tangent / tangent.norm() * dispersion
        weights.append(stiefel_projection_polar(base + tangent))
    return torch.stack(weights)


def measure(dimensions, n_clients, dispersions, n_repeats, generator, device, dtype):
    """Median gap between the two aggregations, at every dispersion."""
    n_in, n_out = dimensions
    gaps, displacements = [], []
    for dispersion in dispersions:
        trial_gaps, trial_displacements = [], []
        for _ in range(n_repeats):
            base = random_stiefel(
                n_in, n_out, 1, generator=generator, device=device, dtype=dtype
            ).squeeze(0)
            locals_ = local_weights(base, dispersion, n_clients, generator)
            projavg, rlavg = aggregate_both_ways(base, locals_)
            trial_gaps.append(float((projavg - rlavg).norm()))
            # How far the aggregate itself moved. The gap is only meaningful
            # against this: two schemes that agree to 1e-8 while both barely
            # moving would not be saying much.
            trial_displacements.append(float((projavg - base).norm()))
        gaps.append(float(np.median(trial_gaps)))
        displacements.append(float(np.median(trial_displacements)))
    return np.array(gaps), np.array(displacements)


def fitted_order(dispersions, gaps, floor):
    """Slope of log(gap) against log(dispersion), above the rounding floor.

    Points at or below ``floor`` are dropped: once the gap reaches machine
    precision it stops following the exponent and flattens, and including that
    tail would drag any fit towards zero.
    """
    dispersions = np.asarray(dispersions)
    usable = gaps > floor
    if usable.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log10(dispersions[usable]), np.log10(gaps[usable]), 1)
    return float(slope)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Order at which the projavg and rlavg aggregations of "
        "prop:spdnet-federe-equivalence coincide."
    )
    parser.add_argument(
        "--dimensions", type=int, nargs="+", action="append", default=None,
        help="A Stiefel geometry as 'n_in n_out'. Repeat the flag for several. "
             "Defaults to (40, 20), (128, 32) and (64, 60).",
    )
    parser.add_argument(
        "--n_clients", type=int, nargs="+", default=[2, 8, 32],
        help="Numbers of clients aggregated per round.",
    )
    parser.add_argument(
        "--dispersions", type=float, nargs="+",
        default=[1e0, 1e-1, 1e-2, 1e-3, 1e-4],
        help="Distances between a local weight and the global iterate. Stops "
             "at 1e-4: below that the gap is at the float64 rounding floor and "
             "carries no exponent.",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=20,
        help="Draws of the base point and the clients at each dispersion.",
    )
    parser.add_argument(
        "--floor", type=float, default=1e-13,
        help="Gaps at or below this are treated as rounding, not signal.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/stiefel_aggregation",
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
        help="Compute device: cpu or cuda.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    args = parser.parse_args()

    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)

    geometries = (
        [tuple(pair) for pair in args.dimensions]
        if args.dimensions
        else [(40, 20), (128, 32), (64, 60)]
    )
    device = torch.device(args.device)
    dtype = torch.float64

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    results = {}
    # One step per (geometry, client count): the unit the sweep is written in,
    # and the one whose cost the user controls through --dimensions and
    # --n_clients.
    with Progress(
        args.storage_path, len(geometries) * len(args.n_clients),
        description="Geometries x clients", unit="fits",
    ) as progress:
        for dimensions in geometries:
            for n_clients in args.n_clients:
                gaps, displacements = measure(
                    dimensions, n_clients, args.dispersions, args.n_repeats,
                    generator, device, dtype,
                )
                results[(dimensions, n_clients)] = {
                    "gaps": gaps,
                    "displacements": displacements,
                    "order": fitted_order(args.dispersions, gaps, args.floor),
                }
                progress.step()

    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4))

    # ---- Panel (a): the gap, with the two candidate orders for reference ----
    ax = axes[0]
    dispersions = np.array(args.dispersions)
    reference = results[(geometries[0], args.n_clients[0])]["gaps"][0]
    for exponent, style in ((2, "--"), (3, ":")):
        ax.plot(
            dispersions, reference * (dispersions / dispersions[0]) ** exponent,
            color="0.5", linewidth=0.9, linestyle=style, zorder=1,
            label=rf"$\varepsilon^{exponent}$",
        )
    for position, dimensions in enumerate(geometries):
        for n_clients in args.n_clients:
            entry = results[(dimensions, n_clients)]
            ax.plot(
                dispersions, entry["gaps"], color=f"C{position}", linewidth=1.2,
                marker="o", markersize=2.5, alpha=0.85, zorder=3,
                label=(
                    rf"$\mathrm{{St}}({dimensions[0]},{dimensions[1]})$"
                    if n_clients == args.n_clients[0] else None
                ),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"dispersion des clients $\varepsilon$")
    ax.set_ylabel(r"$\|\mathrm{projavg}-\mathrm{rlavg}\|_F$")
    ax.legend()

    # ---- Panel (b): the fitted order, per configuration --------------------
    # A bar per (geometry, number of clients), against the O(eps^2) the
    # proposition claims: the panel exists to show the exponent is the same
    # everywhere, and that it is three.
    ax = axes[1]
    labels, orders, colors = [], [], []
    for position, dimensions in enumerate(geometries):
        for n_clients in args.n_clients:
            labels.append(f"({dimensions[0]},{dimensions[1]})\n$K={n_clients}$")
            orders.append(results[(dimensions, n_clients)]["order"])
            colors.append(f"C{position}")
    ax.bar(range(len(orders)), orders, color=colors, width=0.7, zorder=3)
    ax.axhline(2, color="C3", linewidth=1.0, linestyle="--", zorder=4,
               label=r"$O(\varepsilon^2)$ annoncé")
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylim(0, 4)
    ax.set_ylabel("ordre mesuré")
    ax.legend()

    fig.tight_layout()

    # ---- Digest ------------------------------------------------------------
    print(f"n_repeats = {args.n_repeats}, seed = {args.seed}")
    print(f"{'geometrie':>16s} {'K':>4s} {'ordre':>7s} "
          f"{'ecart a eps=1e-2':>17s} {'deplacement':>13s} {'rapport':>9s}")
    index = args.dispersions.index(1e-2) if 1e-2 in args.dispersions else 0
    for dimensions in geometries:
        for n_clients in args.n_clients:
            entry = results[(dimensions, n_clients)]
            gap = entry["gaps"][index]
            displacement = entry["displacements"][index]
            print(
                f"{f'St({dimensions[0]},{dimensions[1]})':>16s} {n_clients:4d} "
                f"{entry['order']:7.2f} {gap:17.3e} {displacement:13.3e} "
                f"{gap / displacement:9.2e}"
            )

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        dispersions=dispersions,
        geometries=np.array(geometries),
        n_clients=np.array(args.n_clients),
        gaps=np.stack([results[key]["gaps"] for key in results]),
        displacements=np.stack([results[key]["displacements"] for key in results]),
        orders=np.array([results[key]["order"] for key in results]),
    )

    if args.export:
        figure_path = os.path.join(args.storage_path, "stiefel_aggregation.tex")
        save_tikz(
            figure_path,
            axis_width=args.axis_width,
            axis_height=args.axis_height,
        )
        write_prov_sidecar(figure_path, args)

    if args.show_interactive:
        plt.show()
