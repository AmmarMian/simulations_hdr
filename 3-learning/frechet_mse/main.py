# How well is the Fréchet mean of a set of covariances estimated?
#
# The mean of ch:learning is the one every nearest-centroid classifier and
# every Riemannian k-means computes. In practice the true covariances are not
# available, only their sample estimates, and the regime is the one of
# subsec:learning-rmt: p and N comparable. The plain Fréchet mean of the SCMs
# is then biased, and the question is by how much, and what the correction
# buys.
#
# Two panels, and they answer two different questions:
#
#   * against the number of samples N: the gap closes as N grows, which is the
#     signature of a bias of regime and not of a variance;
#   * against the number of matrices K: the gap *widens*. Averaging more
#     matrices reduces the variance of the estimate but not its bias, which is
#     common to every SCM; past some K the bias is all that is left and it is
#     the only thing separating the methods. This is the panel that carries
#     the argument, and it is the one the intuition gets wrong.
#
# Five estimators, as in the paper: the plain Fréchet mean of the SCMs, of the
# linearly shrunk covariances (Ledoit-Wolf, OAS), of the non-linearly shrunk
# ones, and the corrected mean. Note where the correction is applied: the
# shrinkage methods regularise each covariance *before* averaging, the RMT
# method corrects the *distance* the average minimises. It is not the same
# gesture, and the figures are what separates them.
#
# Backend-free through hdrlib.learning.rmt, whose port of the reference
# implementation is checked term by term against the published code by
# validate_against_paper.py. Needs float64 — see rmt.require_double.

import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from hdrlib.core.backend import get_data_on_device, to_numpy
from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.mc import Progress, add_mc_base_args, init_logging, make_mc_parser
from hdrlib.core.plot_style import apply_style
from hdrlib.learning import rmt


METHODS = ("SCM", "LW", "OAS", "LW-NL", "RMT")


def random_spd(rng, n_features, condition_number):
    """SPD matrix with a prescribed condition number.

    Random orthogonal basis, eigenvalues uniform between the two extremes,
    which are pinned so that the condition number is exactly the one asked
    for. Same construction as the reference implementation.
    """
    basis = np.linalg.qr(rng.standard_normal((n_features, n_features)))[0]
    low, high = 1 / np.sqrt(condition_number), np.sqrt(condition_number)
    eigenvalues = rng.uniform(low, high, size=n_features)
    eigenvalues[-2], eigenvalues[-1] = low, high
    return basis @ np.diag(eigenvalues) @ basis.T


def covariances_around(rng, centre, n_matrices, scale):
    """Covariances whose Fréchet mean is exactly ``centre``.

    Tangent vectors at the centre are drawn, then *centred* before being
    pushed onto the manifold: their arithmetic mean is zero, so the centre is
    the exact Fréchet mean of the resulting set rather than its limit for
    large ``n_matrices``. Without that subtraction the MSE would measure the
    sampling of the cloud on top of the estimation error, and the two would be
    impossible to tell apart at small ``n_matrices`` — which is precisely the
    regime the second panel is about.
    """
    n_features = centre.shape[0]
    tangents = rng.standard_normal((n_matrices, n_features, n_features)) * scale
    tangents = (tangents + tangents.swapaxes(-1, -2)) / 2
    tangents -= tangents.mean(axis=0)
    values, vectors = np.linalg.eigh(tangents)
    factor = np.linalg.cholesky(centre)
    exponential = vectors @ (
        np.exp(values)[..., None] * vectors.swapaxes(-1, -2)
    )
    return factor @ exponential @ factor.T


def gaussian_data(rng, covariances, n_samples):
    """Centred Gaussian samples, one block per covariance."""
    n_matrices, n_features, _ = covariances.shape
    factors = np.linalg.cholesky(covariances)
    white = rng.standard_normal((n_matrices, n_samples, n_features))
    return white @ factors.swapaxes(-1, -2)


def squared_distance(reference, estimate):
    """Squared affine-invariant distance — the error the figures report."""
    inverse = np.linalg.inv(np.linalg.cholesky(reference))
    logarithms = np.log(
        np.linalg.eigvalsh(inverse @ estimate @ inverse.swapaxes(-1, -2))
    )
    return float(logarithms @ logarithms)


def estimate_all(data, backend, max_iterations, tol):
    """The five estimates of the Fréchet mean, from one set of data blocks."""
    device_data = get_data_on_device(data, backend)
    estimates = {}

    # The four two-step methods: regularise each covariance, then average.
    for name, estimator in (
        ("SCM", rmt.scm),
        ("LW", rmt.ledoit_wolf_linear),
        ("OAS", rmt.oas),
        ("LW-NL", lambda d, b: rmt.analytical_shrinkage(d, b, shrink=0)),
    ):
        covariances = estimator(device_data, backend)
        mean, _ = rmt.frechet_mean_cholesky(
            covariances, max_iterations=max_iterations, backend=backend
        )
        estimates[name] = to_numpy(mean)

    # The one-step method: correct the distance the average minimises.
    mean, _ = rmt.rmt_frechet_mean(
        device_data, max_iterations=max_iterations, tol=tol, backend=backend
    )
    estimates["RMT"] = to_numpy(mean)
    return estimates


def one_trial(job, rng_seed, centre, n_axis, args):
    """One Monte-Carlo trial: draw a cloud, estimate it five ways, score them.

    Seeded per ``(axis index, trial)`` so that a run is reproducible and a
    single point can be re-drawn without replaying the whole sweep — which
    matters here, because the expensive points are at the ends of the axes.
    """
    index, trial, n_matrices, n_samples = job
    rng = np.random.default_rng([rng_seed, index, trial, n_axis])
    covariances = covariances_around(rng, centre, n_matrices, args.scale)
    data = gaussian_data(rng, covariances, n_samples)
    estimates = estimate_all(
        data, args.backend, args.n_iterations_max, args.tol
    )
    return index, trial, {
        name: squared_distance(centre, estimate)
        for name, estimate in estimates.items()
    }


def sweep(rng_seed, centre, axis_name, axis_values, fixed, args, progress=None):
    """One panel: errors of every method over a Monte-Carlo, per axis value.

    The trials of every axis value are independent, so the whole panel is one
    flat job list. On the numpy backend it is handed to a process pool; the
    other backends already batch internally and are run in this process.

    ``progress`` is owned by the caller and spans both panels, so that the run
    reports one monotone count rather than restarting halfway.
    """
    errors = {name: np.zeros((len(axis_values), args.n_trials)) for name in METHODS}
    jobs = [
        (
            index,
            trial,
            value if axis_name == "n_matrices" else fixed,
            value if axis_name == "n_samples" else fixed,
        )
        for index, value in enumerate(axis_values)
        for trial in range(args.n_trials)
    ]
    worker = partial(
        one_trial, rng_seed=rng_seed, centre=centre,
        n_axis=len(axis_values), args=args,
    )

    n_workers = args.n_workers or os.cpu_count()
    # One step per trial. pool.map would only return once every job was done,
    # which is precisely the hour this experiment takes, so the pool is drained
    # with imap_unordered instead: same jobs, same results, but each one lands
    # as it finishes and can be counted. The order is restored below anyway,
    # since every result carries its own (index, trial).
    def advance():
        if progress is not None:
            progress.step()

    results = []
    if args.backend == "numpy" and n_workers > 1:
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(worker, jobs):
                results.append(result)
                advance()
    else:
        for job in jobs:
            results.append(worker(job))
            advance()

    for index, trial, scores in results:
        for name, score in scores.items():
            errors[name][index, trial] = score

    for index, value in enumerate(axis_values):
        print(
            f"  {axis_name} = {value:5d}: "
            + "  ".join(
                f"{name} {10 * np.log10(errors[name][index].mean()):7.2f} dB"
                for name in METHODS
            )
        )
    return errors


def draw(ax, axis_values, errors, xlabel, first):
    """One panel: mean curve per method, with a 5/95 interpercentile band."""
    colors = {"SCM": "C3", "LW": "C1", "OAS": "C4", "LW-NL": "C0", "RMT": "C2"}
    for name in METHODS:
        decibels = 10 * np.log10(errors[name])
        ax.plot(
            axis_values, decibels.mean(axis=1), color=colors[name],
            linewidth=1.4, label=name if first else None,
        )
        ax.fill_between(
            axis_values,
            np.percentile(decibels, 5, axis=1),
            np.percentile(decibels, 95, axis=1),
            color=colors[name], alpha=0.15, linewidth=0,
        )
    ax.set_xscale("log")
    # Minor tick labels off. Two reasons: on a range this narrow matplotlib
    # labels a dozen minor decades, which is unreadable at export size; and
    # matplot2tikz mis-exports them — it emits the label list without the
    # `minor xticklabels={` key that opens it, which is a syntax error in the
    # generated .tex and stops the dissertation build outright.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    if first:
        ax.set_ylabel("eqm (dB)")


if __name__ == "__main__":
    parser = make_mc_parser(
        "Mean squared error of the Fréchet mean of a set of covariances, "
        "against the number of samples and against the number of matrices."
    )
    add_mc_base_args(parser)
    parser.add_argument(
        "--n_features", type=int, default=64,
        help="Dimension of the covariances. The paper uses 64.",
    )
    parser.add_argument(
        "--n_samples", type=int, nargs="+",
        default=[65, 68, 80, 100, 150, 200, 300],
        help="Sample sizes of the first panel. The smallest is barely above "
             "the dimension, which is where the correction matters most.",
    )
    parser.add_argument(
        "--n_matrices_fixed", type=int, default=10,
        help="Number of matrices held fixed in the first panel.",
    )
    parser.add_argument(
        "--n_matrices", type=int, nargs="+",
        default=[3, 5, 20, 30, 40, 60, 80, 100],
        help="Numbers of matrices of the second panel.",
    )
    parser.add_argument(
        "--n_samples_fixed", type=int, default=128,
        help="Sample size held fixed in the second panel.",
    )
    parser.add_argument(
        "--condition_number", type=float, default=100.0,
        help="Condition number of the true mean.",
    )
    parser.add_argument(
        "--scale", type=float, default=0.1,
        help="Spread of the covariances around their mean, in the tangent "
             "space. Large values quickly cost numerical stability.",
    )
    parser.add_argument(
        "--n_iterations_max", type=int, default=100,
        help="Iteration budget of the Riemannian descents.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6,
        help="Stopping tolerance on the relative change of the iterate.",
    )
    parser.add_argument(
        "--axis_width", type=str, default="0.44\\textwidth",
        help="Width of a single panel in the exported PGFPlots figure.",
    )
    parser.add_argument(
        "--axis_height", type=str, default="5.2cm",
        help="Height of a single panel in the exported PGFPlots figure.",
    )
    args = parser.parse_args()
    args.storage_path = args.export_path

    init_logging(args.backend)
    if args.show_interactive:
        apply_style()
    os.makedirs(args.storage_path, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    centre = random_spd(rng, args.n_features, args.condition_number)

    print(f"d = {args.n_features}, {args.n_trials} trials, "
          f"backend = {args.backend}")
    print("panel 1 — against the number of samples "
          f"(K = {args.n_matrices_fixed}):")
    # One counter over both panels: two Progress objects would write the same
    # progress.txt and the second would truncate the first.
    total_trials = (
        (len(args.n_samples) + len(args.n_matrices)) * args.n_trials
    )
    with Progress(
        args.storage_path, total_trials,
        description="Monte-Carlo trials", unit="trials",
    ) as progress:
        errors_samples = sweep(
            args.seed, centre, "n_samples", args.n_samples,
            args.n_matrices_fixed, args, progress,
        )
        print("panel 2 — against the number of matrices "
              f"(N = {args.n_samples_fixed}):")
        errors_matrices = sweep(
            args.seed + 1, centre, "n_matrices", args.n_matrices,
            args.n_samples_fixed, args, progress,
        )

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    draw(axes[0], args.n_samples, errors_samples,
         r"nombre d'échantillons $N$", first=True)
    draw(axes[1], args.n_matrices, errors_matrices,
         r"nombre de matrices $K$", first=False)
    # Legend below the panels: at export size an inner one covers the curves.
    # Attached to an axis, since matplot2tikz drops figure legends.
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.30), ncol=5,
        frameon=False, fontsize=8,
    )
    fig.tight_layout()

    if args.export:
        np.savez(
            os.path.join(args.storage_path, "results.npz"),
            seed=args.seed, n_features=args.n_features,
            n_trials=args.n_trials, condition_number=args.condition_number,
            scale=args.scale, centre=centre,
            n_samples=np.asarray(args.n_samples),
            n_matrices=np.asarray(args.n_matrices),
            n_matrices_fixed=args.n_matrices_fixed,
            n_samples_fixed=args.n_samples_fixed,
            **{f"samples_{name}": errors_samples[name] for name in METHODS},
            **{f"matrices_{name}": errors_matrices[name] for name in METHODS},
        )
        save_path = os.path.join(args.storage_path, "frechet_mse.tex")
        save_tikz(
            save_path, axis_width=args.axis_width, axis_height=args.axis_height
        )
        write_prov_sidecar(save_path, args)
        print(f"Saved MSE figure in {save_path}")

    if args.show_interactive:
        plt.show()
