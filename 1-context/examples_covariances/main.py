# Illustation of different regime of covariance matrices

import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from matplot2tikz import save
import os


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Illustration of different covariance matrix regimes."
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="outputs/example_covariances",
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
    _args = parser.parse_args()
    os.makedirs(_args.storage_path, exist_ok=True)

    # Constant(s)
    d = 7
    rho = 0.8

    # Covariance matrices
    cov_id = np.eye(d)
    cov_toeplitz = toeplitz(np.power(rho, np.arange(0, d)))
    cov_rd = np.empty((d, d))
    cov_rd[np.tril_indices(d)] = 2 * np.random.rand(int(d * (d + 1) / 2)) - 1
    cov_rd[np.triu_indices(d)] = cov_rd[np.tril_indices(d)]
    cov_rd[np.diag_indices(d)] = 1
    cov_rd = 0.5 * (cov_rd + cov_rd.T)

    # Single figure with all three matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    matrices = [cov_id, cov_toeplitz, cov_rd]
    titles = [r"$\mathbf{I}_d$", r"Toeplitz($\rho$)", r"Random"]

    for i, (ax, mat, title) in enumerate(zip(axes, matrices, titles)):
        im = ax.imshow(mat, aspect="equal")
        ax.set_xlabel(r"variable $i$")
        if i == 0:
            ax.set_ylabel(r"variable $j$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.15)

    fig.tight_layout()

    if _args.export:
        save(os.path.join(_args.storage_path, "all_covariances.tex"))

    if _args.show_interactive:
        plt.show()
