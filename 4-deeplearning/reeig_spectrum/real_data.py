# The same two measurements as main.py, on the datasets of the chapter.
#
# main.py establishes the mechanism on simulated CovPool matrices: the largest
# entry of the Loewner matrix of prop:spdnet-diffm for LogEig is 1/lambda_min,
# and a ReEig layer of threshold eps caps it at 1/eps. What that costs and buys
# on real data depends on the spectrum of the real data, which is what this
# script measures.
#
# It cannot be run here — the datasets are not distributable and were not
# available on the machine this was written on — so it is written to be run
# elsewhere, on a machine that has them, with --device cuda. Run --self-test
# first: it exercises every code path on synthetic matrices and needs no data.
#
# Note on the ReEig threshold. eps is an *absolute* floor on the eigenvalues,
# so it is not scale free: the dataset configurations of sigpro_2026 apply a
# scaling_factor (190.0 for HDM05), and multiplying the matrices by a constant
# multiplies the spectrum by it while leaving eps where it is. The digest below
# therefore reports the spectrum's own scale next to the threshold, so that a
# fraction of clamped eigenvalues can be read for what it is.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.plot_style import apply_style

from common import decaying_covariance, resolve_device, sample_covpool, spectral_summary

# Sizes announced in the chapter, used only to check what is loaded against
# what is expected and to warn on a mismatch.
EXPECTED = {
    "hdm05": {"n_features": 93, "condition": 9.1e5},
    "hyperleaf": {"n_features": 204, "condition": 6.0e6},
    "rices90": {"n_features": 256, "condition": 1.2e7},
}


def load_covariances(name, path, max_samples, device, dtype, **kwargs):
    """Stack a dataset's SPD matrices into one tensor.

    Goes through spdnet_datasets rather than reading the files directly, so
    that whatever preprocessing the paper applied — the scaling_factor of the
    dataset configs in particular — is applied here too.
    """
    from spdnet_datasets.manager import DatasetManager

    config = {"name": name, "path": path, "verbose": True, **kwargs}
    dataset = DatasetManager.create_dataset(config)

    matrices = []
    for index in range(len(dataset)):
        if max_samples is not None and index >= max_samples:
            break
        matrix, _ = dataset[index]
        matrices.append(torch.as_tensor(matrix))
    return torch.stack(matrices).to(device=device, dtype=dtype)


def synthetic_covariances(device, dtype, seed=0):
    """Stand-in for --self-test: CovPool matrices in the regime of the chapter."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    true_covariance = decaying_covariance(64, 1e6, device=device, dtype=dtype)
    return sample_covpool(true_covariance, 192, 64, generator, device, dtype)


def median(values):
    return float(torch.median(values).cpu())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Effect of the ReEig threshold on the spectra of the real datasets of "
        "the chapter, and on the Loewner factor of the backward pass."
    )
    parser.add_argument(
        "--dataset", type=str, default="hdm05", choices=sorted(EXPECTED),
        help="Which dataset of the chapter to measure.",
    )
    parser.add_argument(
        "--data-root", type=str, default=os.environ.get("DATA_ROOT"),
        help="Directory holding the dataset. Defaults to $DATA_ROOT.",
    )
    parser.add_argument(
        "--scaling-factor", type=float, default=None,
        help="Passed through to the loader. The sigpro_2026 config uses 190.0 "
             "for HDM05; leaving this unset uses the loader's own default.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap on the number of matrices read, for a quick pass.",
    )
    parser.add_argument(
        "--eps-values", type=float, nargs="+",
        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        help="ReEig thresholds to sweep. The layer's default is 1e-4.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run on synthetic matrices instead of the dataset, to check that "
             "the script works before pointing it at data.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/reeig_real",
        help="Output directory for LaTeX exports.",
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

    if args.self_test:
        covariances = synthetic_covariances(device, dtype, seed=args.seed)
        label = "synthétique (auto-test)"
    else:
        if not args.data_root:
            raise SystemExit(
                "No dataset directory: pass --data-root or set $DATA_ROOT. "
                "Use --self-test to check the script without any data."
            )
        extra = {}
        if args.scaling_factor is not None:
            extra["scaling_factor"] = args.scaling_factor
        covariances = load_covariances(
            args.dataset, args.data_root, args.max_samples, device, dtype, **extra
        )
        label = args.dataset

    n_matrices, n_features = covariances.shape[0], covariances.shape[-1]
    print(f"{label}: {n_matrices} matrices de taille {n_features}")

    if not args.self_test:
        expected = EXPECTED[args.dataset]["n_features"]
        if n_features != expected:
            print(
                f"  ATTENTION: le chapitre annonce des matrices {expected}x{expected} "
                f"pour {args.dataset}, celles-ci sont {n_features}x{n_features}."
            )

    # The spectrum is computed once; the eps sweep only re-reads it. Doing it
    # inside the loop would re-run an eigendecomposition per threshold for
    # nothing, which on Rices90-sized data is the whole runtime.
    eigenvalues = torch.linalg.eigvalsh(covariances)
    lambda_min = eigenvalues.min(dim=-1).values
    lambda_max = eigenvalues.max(dim=-1).values
    condition = lambda_max / lambda_min

    print(f"  lambda_max median  : {median(lambda_max):.3e}")
    print(f"  lambda_min median  : {median(lambda_min):.3e}")
    print(f"  conditionnement    : {median(condition):.3e}", end="")
    if not args.self_test:
        print(f"   (chapitre: {EXPECTED[args.dataset]['condition']:.1e})")
    else:
        print()

    records = []
    print()
    print(f"{'eps':>9s} {'% ecretees':>11s} {'Loewner':>11s} {'+ReEig':>11s} "
          f"{'gain':>9s}")
    for eps in args.eps_values:
        summary = spectral_summary(covariances, eps)
        raw = median(summary["loewner_max"])
        rectified = median(summary["loewner_max_reeig"])
        records.append(
            {
                "eps": eps,
                "fraction": median(summary["fraction_clamped"]),
                "loewner": raw,
                "loewner_reeig": rectified,
            }
        )
        print(
            f"{eps:9.0e} {100 * records[-1]['fraction']:10.1f}% "
            f"{raw:11.3e} {rectified:11.3e} {raw / rectified:8.1f}x"
        )

    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4))

    # ---- Panel (a): the spectrum, with every threshold drawn across it -----
    ax = axes[0]
    index = np.arange(1, n_features + 1)
    spectra = eigenvalues.flip(-1).cpu().numpy()
    ax.fill_between(
        index, np.quantile(spectra, 0.05, axis=0), np.quantile(spectra, 0.95, axis=0),
        color="C0", alpha=0.2, linewidth=0, zorder=1,
    )
    ax.plot(
        index, np.median(spectra, axis=0), color="C0", linewidth=1.3, zorder=3,
        label="spectre médian",
    )
    for eps in args.eps_values:
        ax.axhline(eps, color="C3", linewidth=0.6, linestyle="--", zorder=2)
    ax.axhline(
        args.eps_values[0], color="C3", linewidth=0.6, linestyle="--",
        label=r"seuils $\varepsilon$",
    )
    ax.set_yscale("log")
    ax.set_xlabel("rang de la valeur propre")
    ax.set_ylabel("valeur propre")
    ax.set_title(label)
    ax.legend()

    # ---- Panel (b): what the threshold clamps, and what it buys ------------
    # Two quantities on one panel with twin axes: the fraction of the spectrum
    # rectified (the price, in bias) against the Loewner factor (what is
    # bought). Reading them together is the whole point — a threshold that
    # buys nothing is one that clamps nothing.
    ax = axes[1]
    eps_values = [record["eps"] for record in records]
    ax.plot(
        eps_values, [100 * record["fraction"] for record in records],
        color="C1", linewidth=1.4, marker="o", markersize=3, label="\\% écrêtées",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"seuil $\varepsilon$")
    ax.set_ylabel(r"\% du spectre écrêté")

    twin = ax.twinx()
    twin.plot(
        eps_values, [record["loewner"] for record in records],
        color="C0", linewidth=1.4, linestyle=":", label="sans \\textsc{reeig}",
    )
    twin.plot(
        eps_values, [record["loewner_reeig"] for record in records],
        color="C2", linewidth=1.4, marker="s", markersize=3,
        label="avec \\textsc{reeig}",
    )
    twin.set_yscale("log")
    twin.set_ylabel(r"$\max_{ij}|\mathbf{G}_{ij}|$ de \textsc{logeig}")

    handles, labels = ax.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    ax.legend(handles + twin_handles, labels + twin_labels)

    fig.tight_layout()

    if args.export:
        stem = "reeig_real_selftest" if args.self_test else f"reeig_real_{args.dataset}"
        figure_path = os.path.join(args.storage_path, f"{stem}.tex")
        save_tikz(
            figure_path,
            axis_width=args.axis_width,
            axis_height=args.axis_height,
        )
        write_prov_sidecar(figure_path, args)

    if args.show_interactive:
        plt.show()
