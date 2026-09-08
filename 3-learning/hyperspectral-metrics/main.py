# What does the geometry alone buy? Euclidean against Riemannian K-means.
#
# The sibling experiment (3-learning/hyperspectral-rmt) asks whether the RMT
# correction survives downstream, and answers it by holding the optimiser fixed
# and varying the estimator. This one steps back and asks the prior question,
# the one pyRiemann's image-radar example poses: before any correction, how much
# of the gain is the *metric*? Same scene, same windows, same covariances, same
# alternation — only the geometry in which the centroids are means changes.
#
# Three geometries, in increasing order of what they respect and of what they
# cost:
#
#   euclid     the cone treated as a flat vector space. The centroid is the
#              arithmetic mean of the covariances, closed-form. It is the
#              baseline that a comparison needs and, on its own, the reason the
#              literature bothered with the other two.
#   logeuclid  flat, but on the matrix logarithms. Respects the positivity of
#              the eigenvalues; not affine-invariant. Still closed-form, and it
#              pays for its eigendecompositions once for the whole run.
#   riemann    the affine-invariant metric. The centroid is a Karcher mean, so
#              the only one of the three that iterates. It is what pyRiemann
#              calls ``riemann`` and what the RMT correction of the sibling
#              experiment corrects.
#
# Everything runs on the device — hdrlib.core.clustering.spd_kmeans keeps the
# covariances, the centroids and the labels there, and reads back only the two
# scalars that decide control flow. The windows are dropped before the loop
# starts, since none of these three metrics needs the samples a covariance came
# from; that is the whole reason this experiment fits on a GPU and the corrected
# one does not.
#
# float64 throughout, and enforced rather than assumed: all three metrics end in
# the eigenvalues of a 5x5 covariance built from 25 samples, two of them take the
# logarithm, and in single precision the smallest eigenvalue's sign is not
# reliable — so the answer would be wrong rather than merely imprecise.

import json
import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np

from hdrlib.core.backend import (
    empty_cache,
    get_data_on_device,
    peak_memory_bytes,
    reset_peak_memory,
)
from hdrlib.core.clustering import (
    SPD_METRICS,
    clustering_accuracy,
    match_labels,
    mean_iou,
    require_double,
    spd_kmeans,
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
from hdrlib.core.mc import Progress, add_mc_base_args, make_mc_parser
from hdrlib.core.plot_style import apply_style
from hdrlib.core.rmt import scm

# Same colour per metric everywhere: the figure, the docs export, the tables.
COLOURS = {"euclid": "#c0504d", "logeuclid": "#dea11f", "riemann": "#59bfa3"}
LABELS = {
    "euclid": "euclidien",
    "logeuclid": "log-euclidien",
    "riemann": "riemannien",
}


def describe_device(backend: str) -> dict:
    """Name the hardware the timings were measured on.

    Without this the seconds in ``scores.json`` are unreadable a year later, and
    for this experiment they are unreadable in a specific way: float64 runs at
    half the float32 rate on a datacentre card and at a sixty-fourth of it on a
    workstation one. The affine-invariant metric is the one that notices — it is
    eigendecomposition-bound where the two flat metrics are matmul-bound — so the
    *ranking by time* is a property of the card as much as of the method.
    """
    information = {
        "backend": backend,
        "device": "cpu",
        "host": platform.node(),
        "platform": platform.platform(),
    }
    if backend.startswith("torch"):
        import torch

        information["torch"] = torch.__version__
        if backend == "torch-cuda" and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            information["device"] = properties.name
            information["vram_gb"] = round(properties.total_memory / 1024**3, 1)
            information["capability"] = f"{properties.major}.{properties.minor}"
    return information


def prepare(scene, data_path, n_features, window_size, stride, backend):
    """Scene -> one covariance per pixel, on the device, plus its truth.

    The cube crosses to the device once, here, and nothing crosses back until
    the labels are scored. ``scipy.io.loadmat`` hands back a numpy array whatever
    the backend is, so without the explicit move every step below would be a
    backend call on a host array.
    """
    cube, labels, n_classes = read_scene(scene, data_path)
    if n_features >= cube.shape[-1]:
        # pca_image returns the cube untouched in this case, and the windows
        # would then be (n_pixels, window², n_bands) — 4.4 GB on Salinas at 204
        # bands, against 108 MB at five components. The reduction is not an
        # optimisation here, it is what makes the windowing representable at all.
        raise ValueError(
            f"n_features={n_features} does not reduce a {cube.shape[-1]}-band "
            "cube; the windows would not fit. Choose n_features well below the "
            "number of bands (five is the configuration of the published table)."
        )
    cube = get_data_on_device(cube, backend)
    require_double(cube, "the hyperspectral cube")

    centred = remove_global_mean(cube, backend)
    # Global scale normalisation. The affine-invariant distance is unchanged by
    # a common positive factor, so this is statistically free; it is not free
    # numerically, since raw radiance puts the eigenvalues around 1e8 and the
    # eigensolvers lose most of their precision before they start. The two flat
    # metrics are *not* scale-invariant, so for them this fixes the units the
    # comparison is made in — one more reason to do it before the split.
    centred = centred / centred.std()
    reduced = pca_image(centred, n_features, backend)
    windows = sliding_window_vectorize(reduced, window_size, stride, backend)
    truth = crop_labels(labels, window_size, stride)

    # Formed once, here, rather than inside the K-means: none of these three
    # metrics needs the samples the covariance came from, which is exactly what
    # lets the windows be dropped before the loop starts. On Salinas that is
    # 108 MB returned against 22 MB kept.
    covariances = scm(windows, backend)
    del windows, reduced, centred, cube
    empty_cache(backend)
    return covariances, truth, labels.shape, n_classes


def main():
    parser = make_mc_parser(
        "Euclidean, log-Euclidean and affine-invariant K-means on a "
        "hyperspectral scene, entirely on the device."
    )
    add_mc_base_args(parser)
    parser.add_argument(
        "--scene", type=str, default="salinas",
        help="Scene to segment: indianpines or salinas. Salinas is the one "
             "pyRiemann's example uses.",
    )
    parser.add_argument(
        "--data_path", type=str, default="data/hyperspectral",
        help="Where the .mat files live; downloaded there if missing.",
    )
    parser.add_argument(
        "--n_features", type=int, default=5,
        help="Principal components kept. Five is pyRiemann's setting and the "
             "configuration of the published table.",
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
        "--n_init", type=int, default=10,
        help="Restarts of the K-means; the one of least inertia is kept. Every "
             "metric gets the same starting partitions.",
    )
    parser.add_argument(
        "--max_iter", type=int, default=100,
        help="Assignment/re-estimation rounds per restart.",
    )
    parser.add_argument(
        "--mean_iterations", type=int, default=10,
        help="Iteration budget of one Karcher mean, warm-started on the "
             "previous centroids. Ignored by the two flat metrics.",
    )
    parser.add_argument(
        "--max_batch", type=int, default=16000,
        help="Largest batch handed to the eigensolver at once. Only the "
             "affine-invariant metric is bounded by it: cuSOLVER refuses a "
             "batch of n_clusters x n_pixels matrices outright on a scene the "
             "size of Salinas. Lowering it costs kernel launches, not results.",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=list(SPD_METRICS),
        help="Subset of the three geometries to run.",
    )
    parser.add_argument(
        "--figure_width", type=str, default="0.23\\textwidth",
        help="Width of one map in the exported figure.",
    )
    args = parser.parse_args()
    args.storage_path = args.export_path

    # Not init_logging: its GPU disclaimer is written for the Monte-Carlo
    # experiments, whose batched path is sequential over T checkpoints. Nothing
    # here is a Monte-Carlo trial and the device path is the fast one, so the
    # warning would be actively misleading.
    if args.show_interactive:
        apply_style()
    os.makedirs(args.storage_path, exist_ok=True)

    device = describe_device(args.backend)
    print(f"backend {args.backend} on {device['device']}", flush=True)
    reset_peak_memory(args.backend)

    download_scene(args.scene, args.data_path)
    covariances, truth, image_shape, n_classes = prepare(
        args.scene, args.data_path, args.n_features, args.window_size,
        args.stride, args.backend,
    )
    concentration = args.n_features / args.window_size**2
    print(f"{args.scene}: {covariances.shape[0]} covariances "
          f"{covariances.shape[-1]}x{covariances.shape[-1]}, "
          f"c = {concentration:.2f}, {n_classes} classes", flush=True)

    progress = Progress(args.storage_path, len(args.metrics))
    maps, scores = {}, {}
    for metric in args.metrics:
        start = time.perf_counter()
        labels, inertia, histories = spd_kmeans(
            covariances, n_classes, metric=metric, n_init=args.n_init,
            max_iter=args.max_iter, mean_iterations=args.mean_iterations,
            max_batch=args.max_batch, seed=args.seed, backend=args.backend,
            verbose=True,
        )
        segmented = unvectorize_labels(
            labels, *image_shape, args.window_size, args.stride
        )
        matched = match_labels(segmented, truth)
        accuracy = clustering_accuracy(matched, truth)
        ious, miou = mean_iou(matched, truth)
        elapsed = time.perf_counter() - start

        maps[metric] = matched
        scores[metric] = {
            "accuracy": accuracy, "mIoU": miou,
            # Comparable between restarts of one metric, never between metrics:
            # the three measure lengths in different geometries. The ranking of
            # the metrics is the accuracy and the mIoU, which are on the truth.
            "inertia": inertia, "seconds": elapsed,
            "restarts": histories,
            "worst_moved": max(h["moved"] for h in histories),
        }
        print(f"{LABELS[metric]:15s} acc={accuracy:.3f}  mIoU={miou:.3f}  "
              f"({elapsed:.0f}s, worst restart left "
              f"{scores[metric]['worst_moved']:.2%} moving)", flush=True)
        progress.step()

    peak = peak_memory_bytes(args.backend)
    if peak is not None:
        device["peak_vram_gb"] = round(peak / 1024**3, 2)
        print(f"peak device memory {device['peak_vram_gb']} GB", flush=True)

    # ── the figure: ground truth, then one map per geometry ───────────────
    panels = ["vérité terrain"] + list(args.metrics)
    figure, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.4))
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
                f"{LABELS[panel]}\\n{scores[panel]['accuracy']:.3f} / "
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
            mean_iterations=args.mean_iterations, max_batch=args.max_batch,
            n_classes=n_classes, concentration=concentration,
            metrics=np.array(args.metrics), truth=truth,
            **{f"map_{m}": maps[m] for m in args.metrics},
        )
        with open(os.path.join(args.storage_path, "scores.json"), "w") as handle:
            json.dump({"device": device, "scores": scores}, handle, indent=2)
        # Saved as an image rather than PGFPlots: these are label maps, and a
        # PGFPlots export of a few hundred thousand coloured cells would be
        # unusable both to compile and to open.
        save_path = os.path.join(args.storage_path, "segmentation.pdf")
        figure.savefig(save_path, bbox_inches="tight", dpi=300)
        write_prov_sidecar(save_path, args)
        print(f"Saved segmentation maps in {save_path}")

    progress.done()

    if args.show_interactive:
        plt.show()


if __name__ == "__main__":
    main()
