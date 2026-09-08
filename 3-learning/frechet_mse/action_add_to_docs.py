#!/usr/bin/env python
"""Qanat action: export the two MSE panels as Plotly JSON.

Reads results.npz — the per-trial squared distances of every method, for both
sweeps — and rebuilds the two panels: mean curve in dB with a 5/95
interpercentile band, against the number of samples and against the number of
matrices.

Output: docs/docs/assets/data/learning_frechet_mse.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly not found — run: uv add plotly --dev")

from hdrlib.core.plotly_style import (
    BG, GRID, AXIS_LINE, MUTED, INK2,
    FONT_SANS, FONT_MONO, hex_to_rgba,
)
from hdrlib.core.exporter import write_docs_provenance

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--storage_path", required=True,
    help="Run directory injected by qanat.")
parser.add_argument("--name", default=None,
    help="Experiment name (output filename). Defaults to grandparent dir.")
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[2]
storage = Path(args.storage_path)
name = args.name or storage.parent.name

npz_path = storage / "results.npz"
if not npz_path.exists():
    sys.exit(f"No results.npz in {storage}")

r = np.load(npz_path)
n_samples = r["n_samples"]
n_matrices = r["n_matrices"]

# Same order and same colours as the exported figure, so that the interactive
# version and the one in the dissertation read as the same picture.
METHODS = [
    ("SCM", "#c0504d"),
    ("LW", "#dea11f"),
    ("OAS", "#9a6fb0"),
    ("LW-NL", "#4a7c9e"),
    ("RMT", "#59bfa3"),
]

TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.09,
    subplot_titles=[
        f"contre le nombre d'échantillons (K = {int(r['n_matrices_fixed'])})",
        f"contre le nombre de matrices (N = {int(r['n_samples_fixed'])})",
    ],
)

for column, (axis, prefix) in enumerate(
    ((n_samples, "samples"), (n_matrices, "matrices")), start=1
):
    for name_method, colour in METHODS:
        decibels = 10 * np.log10(r[f"{prefix}_{name_method}"])
        low = np.percentile(decibels, 5, axis=1)
        high = np.percentile(decibels, 95, axis=1)

        # Band first, as one closed polygon: going out along the low edge and
        # back along the high one.
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([axis, axis[::-1]]),
                y=np.concatenate([low, high[::-1]]),
                fill="toself", fillcolor=hex_to_rgba(colour, 0.13),
                line=dict(width=0), hoverinfo="skip",
                showlegend=False, legendgroup=name_method,
            ),
            row=1, col=column,
        )
        fig.add_trace(
            go.Scatter(
                x=axis, y=decibels.mean(axis=1), mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(size=5, color=colour),
                name=name_method, legendgroup=name_method,
                showlegend=column == 1,
                hovertemplate="%{x}<br>%{y:.2f} dB<extra>"
                              + name_method + "</extra>",
            ),
            row=1, col=column,
        )

axis_common = dict(
    tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=70, r=20, b=60, l=70),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=430, width=980,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1,
        font=dict(family=FONT_SANS, size=12, color=MUTED),
    ),
    xaxis={**axis_common, "type": "log",
           "title": dict(text="nombre d'échantillons N", font=TITLE, standoff=12)},
    yaxis={**axis_common,
           "title": dict(text="eqm (dB)", font=TITLE, standoff=12)},
    xaxis2={**axis_common, "type": "log",
            "title": dict(text="nombre de matrices K", font=TITLE, standoff=12)},
    yaxis2={**axis_common,
            "title": dict(text="eqm (dB)", font=TITLE, standoff=12)},
    title=dict(
        text=f"d = {int(r['n_features'])}, {int(r['n_trials'])} tirages, "
             f"conditionnement {float(r['condition_number']):g} · "
             "bandes : interpercentiles 5/95",
        font=dict(family=FONT_SANS, size=12, color=MUTED),
        x=0.0, xanchor="left", y=0.015, yanchor="bottom",
    ),
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
write_docs_provenance(out.parent, name, npz_path)
print(f"Written {out.relative_to(repo_root)}")
