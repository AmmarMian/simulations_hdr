# Does the mean matched to the model win, and by how much?
#
# prop:spdnet-moyennes-frechet identifies each mean of the batch-norm layer with
# the Fréchet mean of a geometry or a divergence: arithmetic with the left
# Kullback-Leibler (Wishart), harmonic with the right one (inverse-Wishart), GAH
# with the symmetrised one. The chapter needs that read as a modelling statement
# rather than a numerical curiosity, and the only support it currently plans for
# is the F1 tables on three real datasets — an indirect argument, on data whose
# law is unknown.
#
# The wishart-inverse grid of eusipco_2026 tests it head on: draw the data from a
# Wishart, then from an inverse-Wishart, and see which mean wins. This adds the
# one axis that grid does not sweep — the degrees of freedom. As df grows the
# Wishart concentrates around its scale matrix and the choice of mean should
# matter less, so the mechanism should appear as a gradient rather than as two
# points. That is what makes it a *statement about the model* and not about one
# particular setting.
#
# Everything is reused from eusipco_2026: run_single_experiment does the data
# generation, the training and the evaluation. This file only sweeps and draws.

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from hdrlib.core.mc import Progress
from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.plot_style import apply_style

try:
    from eusipco_2026.simulation.runner import run_single_experiment
    from spdnet_datasets.synthetic import ExperimentConfig
except ImportError:  # pragma: no cover - environment guidance, not logic
    sys.exit(
        "eusipco_2026 and spdnet-datasets are needed and are not dependencies "
        "of this repository. See README.md in this directory."
    )

# Wishart draws sit at 'large', inverse-Wishart draws at 'small'; the generator
# of spdnet-datasets overloads discriminant_position to select the law.
MODELS = {"large": "Wishart", "small": "Wishart inverse"}

MEANS = {
    "arithmetic": "arithmétique",
    "harmonic": "harmonique",
    "geometric_arithmetic_harmonic": r"\textsc{gah}",
    "adaptive_geometric_arithmetic_harmonic": r"\textsc{armagnac}",
    "affine_invariant": "géométrique",
}


def sweep(args):
    """One experiment per (df, model, mean, seed), all through eusipco_2026."""
    accuracies = defaultdict(list)
    total = len(args.df) * len(MODELS) * len(args.means) * len(args.seeds)
    progress = Progress(
        args.storage_path, total,
        description="df x model x mean x seed", unit="fits",
    )
    done = 0
    for df in args.df:
        for position in MODELS:
            for mean in args.means:
                for seed in args.seeds:
                    config = ExperimentConfig(
                        generation_mode="wishart",
                        structure="full",
                        eigenvalue_mode="random",
                        matrix_size=args.matrix_size,
                        n_classes=args.n_classes,
                        n_samples_per_class=args.n_samples_per_class,
                        conditioning=args.conditioning,
                        n_discriminant=0,
                        discriminant_position=position,
                        class_separation_ratio=0.0,
                        max_value=1.0,
                        df=df,
                        batchnorm_method=mean,
                        seed=seed,
                    )
                    result = run_single_experiment(config, force_cpu=args.cpu)
                    accuracies[(df, position, mean)].append(result["test_acc"])
                    done += 1
                    progress.step()
                    print(
                        f"  [{done:4d}/{total}] df={df:4d} {MODELS[position]:16s} "
                        f"{mean:40s} seed={seed:5d} acc={result['test_acc']:.3f}",
                        flush=True,
                    )
    progress.done()
    return accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Accuracy of each batch-norm mean against the degrees of freedom of the "
        "Wishart / inverse-Wishart model that generated the data."
    )
    parser.add_argument(
        "--df", type=int, nargs="+", default=[64, 96, 160, 320, 640],
        help="Degrees of freedom. Must exceed matrix_size - 1. As df grows the "
             "law concentrates and the choice of mean should matter less.",
    )
    parser.add_argument(
        "--means", type=str, nargs="+", default=sorted(MEANS),
        help="Batch-norm means to compare.",
    )
    parser.add_argument("--matrix-size", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--n-samples-per-class", type=int, default=120)
    parser.add_argument("--conditioning", type=float, default=100.0)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011],
        help="Seeds, as in the wishart-inverse grid of eusipco_2026.",
    )
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument(
        "--storage_path", type=str, default="outputs/wishart_model",
        help="Output directory for LaTeX exports.",
    )
    parser.add_argument("--show-interactive", action="store_true")
    parser.add_argument(
        "--export", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--axis_width", type=str, default="0.45\\textwidth")
    parser.add_argument("--axis_height", type=str, default="4.6cm")
    parser.add_argument("--seed", type=int, default=42, help="Unused; for provenance.")
    args = parser.parse_args()

    if args.show_interactive:
        apply_style()
    os.makedirs(args.storage_path, exist_ok=True)

    accuracies = sweep(args)

    # ---- One panel per model, the means as curves against df ---------------
    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4), sharey=True)
    for axis, (position, title) in zip(axes, MODELS.items()):
        for index, mean in enumerate(args.means):
            medians = [
                100 * np.mean(accuracies[(df, position, mean)]) for df in args.df
            ]
            spread = [
                100 * np.std(accuracies[(df, position, mean)]) for df in args.df
            ]
            axis.errorbar(
                args.df, medians, yerr=spread, color=f"C{index}", linewidth=1.3,
                marker="o", markersize=3, capsize=2, label=MEANS[mean],
            )
        axis.set_xscale("log")
        axis.set_xlabel("degrés de liberté")
        axis.set_title(f"données {title}")
    axes[0].set_ylabel(r"précision de test (\%)")
    axes[0].legend()
    fig.tight_layout()

    # ---- Digest ------------------------------------------------------------
    print()
    header = "  ".join(f"{MEANS[m][:12]:>13s}" for m in args.means)
    print(f"{'df':>6s} {'modèle':>16s}  {header}")
    for df in args.df:
        for position, title in MODELS.items():
            cells = "  ".join(
                f"{100 * np.mean(accuracies[(df, position, m)]):12.1f}%"
                for m in args.means
            )
            print(f"{df:6d} {title:>16s}  {cells}")

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        df=np.array(args.df),
        means=np.array(args.means),
        models=np.array(list(MODELS)),
        accuracies=np.array(
            [
                [[accuracies[(df, p, m)] for m in args.means] for p in MODELS]
                for df in args.df
            ]
        ),
    )

    if args.export:
        figure_path = os.path.join(args.storage_path, "wishart_model.tex")
        save_tikz(
            figure_path, axis_width=args.axis_width, axis_height=args.axis_height
        )
        write_prov_sidecar(figure_path, args)

    if args.show_interactive:
        plt.show()
