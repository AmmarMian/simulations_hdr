# Does the correction survive downstream? Segmenting a hyperspectral scene.
#
# The eqm figure of the same chapter measures an *internal* criterion: how far
# the estimated Fréchet mean is from the true one. But the thesis of
# ch:learning is that the criterion has left the model — so the estimate has to
# be judged on the task, not on itself. That is what this experiment does.
#
# The pipeline is the standard one for covariance-based segmentation:
#
#   scene -> remove the global mean -> PCA to n_features bands
#         -> sliding window -> one covariance per pixel
#         -> Riemannian K-means -> compare to the ground truth
#
# and it is where the dimensional regime becomes concrete. A 5x5 window on 5
# principal components gives 25 samples for 5 variables: c = 0.2, and every
# pixel's covariance is estimated from a handful of neighbours. Exactly the
# regime of subsec:learning-rmt, and exactly the regime the second panel of the
# eqm figure describes — many matrices, each badly estimated.
#
# Four methods, differing only in what the centroids minimise: the plain
# Fréchet mean of the SCMs, of the linearly shrunk covariances, of the
# non-linearly shrunk ones, and the corrected mean. Two scores, because they
# disagree informatively: accuracy is dominated by the large classes, mIoU is
# not.
#
# Backend-free, float64 (see hdrlib.core.rmt.require_double), and no
# scikit-learn: the pipeline must be able to run on a GPU backend.

import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from hdrlib.core.clustering import (
    clustering_accuracy,
    match_labels,
    mean_iou,
    riemannian_kmeans,
)
from hdrlib.core.exporter import write_prov_sidecar
from hdrlib.core.hyperspectral import (
    crop_labels,
    download_scene,
    pca_image,
    read_scene,
    remove_global_mean,
    sliding_window_vectorize,
    unvectorize_labels,
)
from hdrlib.core.mc import add_mc_base_args, init_logging, make_mc_parser
from hdrlib.core.plot_style import apply_style


METHODS = ("SCM", "LW", "LW-NL", "RMT")


def prepare(scene, data_path, n_features, window_size, stride, backend):
    """Scene -> one block of neighbouring samples per pixel, plus its truth."""
    cube, labels, n_classes = read_scene(scene, data_path)
    centred = remove_global_mean(cube, backend)
    # Global scale normalisation. The affine-invariant distance is unchanged by
    # a common positive factor, so this is statistically free; it is not free
    # numerically, since raw radiance puts the eigenvalues around 1e8 and the
    # descents lose most of their precision before they start.
    centred = centred / centred.std()
    reduced = pca_image(centred, n_features, backend)
    windows = sliding_window_vectorize(reduced, window_size, stride, backend)
    truth = crop_labels(labels, window_size, stride)
    return windows, truth, labels.shape, n_classes


def main():
    parser = make_mc_parser(
        "Riemannian K-means segmentation of a hyperspectral scene, with and "
        "without the random-matrix-theory correction."
    )
    add_mc_base_args(parser)
    parser.add_argument(
        "--scene", type=str, default="salinas",
        help="Scene to segment: indianpines or salinas.",
    )
    parser.add_argument(
        "--data_path", type=str, default="data/hyperspectral",
        help="Where the .mat files live; downloaded there if missing.",
    )
    parser.add_argument(
        "--n_features", type=int, default=5,
        help="Principal components kept. Five is the configuration of the "
             "published table, and a handful of directions represent these "
             "scenes well.",
    )
    parser.add_argument(
        "--window_size", type=int, default=5,
        help="Side of the square neighbourhood, odd. With n_features=5 this "
             "gives 25 samples for 5 variables, c = 0.2.",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Step between two windows. One segments every pixel; a larger "
             "value trades resolution for time, and the caption must say so.",
    )
    parser.add_argument(
        "--n_init", type=int, default=5,
        help="Restarts of the K-means; the one of least inertia is kept.",
    )
    parser.add_argument(
        "--max_iter", type=int, default=30,
        help="Assignment/re-estimation rounds per restart.",
    )
    parser.add_argument(
        "--mean_iterations", type=int, default=50,
        help="Iteration budget of one Fréchet mean.",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=list(METHODS),
        help="Subset of the four methods to run.",
    )
    parser.add_argument(
        "--figure_width", type=str, default="0.23\\textwidth",
        help="Width of one map in the exported figure.",
    )
    args = parser.parse_args()
    args.storage_path = args.export_path

    init_logging(args.backend)
    if args.show_interactive:
        apply_style()
    os.makedirs(args.storage_path, exist_ok=True)

    download_scene(args.scene, args.data_path)
    windows, truth, image_shape, n_classes = prepare(
        args.scene, args.data_path, args.n_features, args.window_size,
        args.stride, args.backend,
    )
    concentration = args.n_features / args.window_size**2
    print(f"{args.scene}: {windows.shape[0]} fenêtres de "
          f"{windows.shape[1]} échantillons en dimension {windows.shape[2]}, "
          f"c = {concentration:.2f}, {n_classes} classes", flush=True)

    maps, scores = {}, {}
    for method in args.methods:
        start = time.perf_counter()
        labels, inertia, histories = riemannian_kmeans(
            windows, n_classes, method=method, n_init=args.n_init,
            max_iter=args.max_iter, mean_iterations=args.mean_iterations,
            seed=args.seed, backend=args.backend, verbose=True,
        )
        segmented = unvectorize_labels(
            labels, *image_shape, args.window_size, args.stride
        )
        matched = match_labels(segmented, truth)
        accuracy = clustering_accuracy(matched, truth)
        ious, miou = mean_iou(matched, truth)
        elapsed = time.perf_counter() - start

        maps[method] = matched
        scores[method] = {
            "accuracy": accuracy, "mIoU": miou,
            "inertia": inertia, "seconds": elapsed,
            "restarts": histories,
            "worst_moved": max(h["moved"] for h in histories),
        }
        print(f"{method:6s} acc={accuracy:.3f}  mIoU={miou:.3f}  "
              f"({elapsed:.0f}s, worst restart left "
              f"{scores[method]['worst_moved']:.2%} moving)", flush=True)

    # ── the figure: ground truth, then one map per method ────────────────
    panels = ["vérité terrain"] + list(args.methods)
    figure, axes = plt.subplots(
        1, len(panels), figsize=(2.0 * len(panels), 2.4)
    )
    axes = np.atleast_1d(axes)
    # A discrete colormap: these are class labels, not a continuous field, so a
    # perceptual gradient would suggest an order between crops that has none.
    colormap = plt.get_cmap("tab20", n_classes + 1)
    for axis, panel in zip(axes, panels):
        image = truth if panel == "vérité terrain" else maps[panel]
        axis.imshow(image, cmap=colormap, vmin=0, vmax=n_classes,
                    interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        if panel == "vérité terrain":
            axis.set_title(panel, fontsize=8)
        else:
            axis.set_title(
                f"{panel}\\n{scores[panel]['accuracy']:.3f} / "
                f"{scores[panel]['mIoU']:.3f}",
                fontsize=8,
            )
    figure.tight_layout()

    if args.export:
        np.savez(
            os.path.join(args.storage_path, "results.npz"),
            scene=args.scene, seed=args.seed, n_features=args.n_features,
            window_size=args.window_size, stride=args.stride,
            n_init=args.n_init, max_iter=args.max_iter,
            n_classes=n_classes, concentration=concentration,
            truth=truth, **{f"map_{m}": maps[m] for m in args.methods},
        )
        with open(os.path.join(args.storage_path, "scores.json"), "w") as handle:
            json.dump(scores, handle, indent=2)
        # Saved as an image rather than PGFPlots: these are label maps, and a
        # PGFPlots export of a few hundred thousand coloured cells would be
        # unusable both to compile and to open.
        save_path = os.path.join(args.storage_path, "segmentation.pdf")
        figure.savefig(save_path, bbox_inches="tight", dpi=300)
        write_prov_sidecar(save_path, args)
        print(f"Saved segmentation maps in {save_path}")

    if args.show_interactive:
        plt.show()


if __name__ == "__main__":
    main()
