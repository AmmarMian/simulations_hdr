# What the hand-written backward of the batch-norm layer costs, and what it saves.
#
# Volet 1 of sec:spdnet-batchnorm-resultats. The figures of the article that
# section reports were produced by code that was never committed — none of the
# five SPDnet repositories contains any timing or memory instrumentation — so
# this is a rewrite rather than a replay. It needs no data.
#
# On memory. The article measures torch.cuda.max_memory_allocated, which needs
# a GPU and mixes the quantity of interest with the allocator's behaviour. What
# prop:spdnet-grad-geo actually predicts is narrower and exactly measurable:
# automatic differentiation has to *retain the n_iterations iterates* of the
# fixed point eq:spdnet-geometrique-iteration, where the hand-written backward
# recomputes what it needs. torch.autograd.graph.saved_tensors_hooks intercepts
# every tensor the graph keeps alive, so summing their sizes measures precisely
# that, on any device. Peak allocator memory is reported too when running on
# CUDA, so the two can be compared against the article.
#
# On time. Measured, and reported without being oversold: on CPU the manual
# backward of the geometric mean is *slower* than autograd. The case for
# deriving by hand is memory and numerical robustness, not speed, and the
# chapter should say so plainly.

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from hdrlib.core.exporter import save_tikz, write_prov_sidecar
from hdrlib.core.plot_style import apply_style

from yetanotherspdnet.nn.batchnorm import BatchNormSPDMean
from yetanotherspdnet.random.spd import random_SPD

# The two means the article compares: the one with no closed form, whose
# backward has to unroll a fixed point, and the closed-form alternative that
# replaces it.
MEANS = {
    "affine_invariant": "géométrique",
    "geometric_arithmetic_harmonic": r"\textsc{gah}",
}


class GraphFootprint:
    """Bytes retained by the autograd graph inside the ``with`` block.

    Counts every tensor saved for backward, once per storage: a tensor saved by
    several nodes is one allocation, and counting it twice would inflate the
    manual path and the automatic one differently.
    """

    def __init__(self):
        self.total = 0
        self._seen = set()

    def __enter__(self):
        def pack(tensor):
            key = tensor.untyped_storage().data_ptr()
            if key not in self._seen:
                self._seen.add(key)
                self.total += tensor.untyped_storage().nbytes()
            return tensor

        self._hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        self._hooks.__exit__(*exc)


def build(n_features, mean_type, use_autograd, depth, n_iterations, device, dtype):
    """A stack of ``depth`` batch-norm layers, which is what the depth sweep varies."""
    options = {"n_iterations": n_iterations} if mean_type == "affine_invariant" else None
    return torch.nn.Sequential(
        *[
            BatchNormSPDMean(
                n_features,
                mean_type=mean_type,
                mean_options=options,
                use_autograd=use_autograd,
                device=device,
                dtype=dtype,
            )
            for _ in range(depth)
        ]
    )


def one_measurement(
    n_features, batch_size, depth, mean_type, use_autograd, args, device, dtype
):
    """Time and retained memory of one forward and backward pass."""
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    data = random_SPD(
        n_features, batch_size, cond=args.cond,
        device=device, dtype=dtype, generator=generator,
    ).clone().requires_grad_(True)

    model = build(
        n_features, mean_type, use_autograd, depth, args.n_iterations, device, dtype
    )

    # One untimed pass first: the first call through a spectral layer pays for
    # lazily initialised buffers and, on CUDA, for the kernels themselves.
    (model(data) ** 2).sum().backward()
    data.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    footprint = GraphFootprint()
    start = time.perf_counter()
    for _ in range(args.n_repeats):
        data.grad = None
        with footprint:
            loss = (model(data) ** 2).sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / args.n_repeats

    return {
        "time": elapsed,
        # The hooks fire on every repeat, so divide back out.
        "graph_bytes": footprint.total / args.n_repeats,
        "peak_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else float("nan")
        ),
    }


SWEEPS = {
    "size": ("taille de matrice", lambda v, a: dict(n_features=v, batch_size=a.batch_size, depth=1)),
    "batch": ("taille de batch", lambda v, a: dict(n_features=a.n_features, batch_size=v, depth=1)),
    "depth": ("profondeur", lambda v, a: dict(n_features=a.n_features, batch_size=a.batch_size, depth=v)),
    "iterations": ("itérations du point fixe", None),
}


def run_sweep(name, values, args, device, dtype):
    """Every (mean, differentiation) combination along one axis."""
    records = []
    for value in values:
        for mean_type in args.means:
            for use_autograd in (False, True):
                if name == "iterations":
                    # Only the geometric mean has a fixed point to unroll; the
                    # closed-form means do not depend on this axis at all.
                    if mean_type != "affine_invariant":
                        continue
                    shape = dict(
                        n_features=args.n_features,
                        batch_size=args.batch_size,
                        depth=1,
                    )
                    args.n_iterations = int(value)
                else:
                    shape = SWEEPS[name][1](value, args)
                measurement = one_measurement(
                    mean_type=mean_type, use_autograd=use_autograd,
                    args=args, device=device, dtype=dtype, **shape,
                )
                measurement.update(
                    value=value, mean_type=mean_type, use_autograd=use_autograd
                )
                records.append(measurement)
    return records


def draw(records, axis_label, args):
    """Two panels: retained memory on the left, wall time on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3.4))
    values = sorted({record["value"] for record in records})

    for position, mean_type in enumerate(args.means):
        for use_autograd, style, marker in ((False, "-", "o"), (True, "--", "s")):
            selected = [
                record for record in records
                if record["mean_type"] == mean_type
                and record["use_autograd"] == use_autograd
            ]
            if not selected:
                continue
            selected.sort(key=lambda record: record["value"])
            label = (
                f"{MEANS[mean_type]}, "
                + ("autograd" if use_autograd else "manuel")
            )
            axes[0].plot(
                [record["value"] for record in selected],
                [record["graph_bytes"] / 2**20 for record in selected],
                color=f"C{position}", linestyle=style, marker=marker, markersize=3,
                linewidth=1.3, label=label,
            )
            axes[1].plot(
                [record["value"] for record in selected],
                [1e3 * record["time"] for record in selected],
                color=f"C{position}", linestyle=style, marker=marker, markersize=3,
                linewidth=1.3, label=label,
            )

    for axis, ylabel in zip(axes, ("mémoire retenue (Mio)", "temps (ms)")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(axis_label)
        axis.set_ylabel(ylabel)
    axes[0].legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Time and memory of the batch-norm layer, hand-written backward against "
        "automatic differentiation."
    )
    parser.add_argument(
        "--sweep", type=str, default="size", choices=sorted(SWEEPS),
        help="Which axis to vary. 'iterations' is the mechanism behind the "
             "other three and is not in the article.",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", default=None,
        help="Points along the swept axis. Defaults per sweep: 8..512 for "
             "size and batch, 1..32 for depth, 1..20 for iterations.",
    )
    parser.add_argument(
        "--means", type=str, nargs="+", default=sorted(MEANS),
        help="Means to compare. The article compares the geometric one with GAH.",
    )
    parser.add_argument(
        "--n_features", type=int, default=64,
        help="Matrix size, when it is not the swept axis.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size, when it is not the swept axis.",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=5,
        help="Fixed-point iterations of the geometric mean, when not swept.",
    )
    parser.add_argument(
        "--cond", type=float, default=1e5,
        help="Condition number of the drawn matrices, as in the article.",
    )
    parser.add_argument(
        "--n_repeats", type=int, default=5,
        help="Passes averaged at each point.",
    )
    parser.add_argument(
        "--storage_path", type=str, default="outputs/batchnorm_cost",
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
        help="Compute device: cpu or cuda. On cuda the allocator peak is "
             "reported alongside the retained-graph measurement.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    args = parser.parse_args()

    if args.show_interactive:
        apply_style()

    os.makedirs(args.storage_path, exist_ok=True)

    if args.device == "mps":
        raise SystemExit(
            "MPS cannot run this study: no float64, and linalg.eigh is not "
            "implemented for it. Use --device cpu or --device cuda."
        )
    device = torch.device(args.device)
    dtype = torch.float64

    defaults = {
        "size": [8, 16, 32, 64, 128, 256, 512],
        "batch": [8, 16, 32, 64, 128, 256, 512],
        "depth": [1, 2, 4, 8, 16, 32],
        "iterations": [1, 2, 5, 10, 20],
    }
    values = args.values if args.values is not None else defaults[args.sweep]
    values = [int(value) for value in values]

    records = run_sweep(args.sweep, values, args, device, dtype)

    axis_label, _ = SWEEPS[args.sweep]
    figure = draw(records, axis_label, args)

    # ---- Digest ------------------------------------------------------------
    print(f"sweep = {args.sweep}, device = {device}, cond = {args.cond:g}, "
          f"n_repeats = {args.n_repeats}")
    print(f"{'valeur':>8s} {'moyenne':>32s} {'mem manuel':>12s} {'mem auto':>11s} "
          f"{'rapport':>8s} {'t manuel':>10s} {'t auto':>9s}")
    for value in values:
        for mean_type in args.means:
            pair = {
                record["use_autograd"]: record
                for record in records
                if record["value"] == value and record["mean_type"] == mean_type
            }
            if len(pair) != 2:
                continue
            manual, auto = pair[False], pair[True]
            print(
                f"{value:8d} {mean_type:>32s} "
                f"{manual['graph_bytes'] / 2**20:11.2f}M "
                f"{auto['graph_bytes'] / 2**20:10.2f}M "
                f"{auto['graph_bytes'] / manual['graph_bytes']:7.2f}x "
                f"{1e3 * manual['time']:9.1f}ms {1e3 * auto['time']:8.1f}ms"
            )

    np.savez(
        os.path.join(args.storage_path, "results.npz"),
        sweep=args.sweep,
        values=np.array([record["value"] for record in records]),
        mean_types=np.array([record["mean_type"] for record in records]),
        use_autograd=np.array([record["use_autograd"] for record in records]),
        graph_bytes=np.array([record["graph_bytes"] for record in records]),
        peak_bytes=np.array([record["peak_bytes"] for record in records]),
        times=np.array([record["time"] for record in records]),
    )

    if args.export:
        figure_path = os.path.join(
            args.storage_path, f"batchnorm_cost_{args.sweep}.tex"
        )
        save_tikz(
            figure_path,
            axis_width=args.axis_width,
            axis_height=args.axis_height,
        )
        write_prov_sidecar(figure_path, args)

    if args.show_interactive:
        plt.show()
