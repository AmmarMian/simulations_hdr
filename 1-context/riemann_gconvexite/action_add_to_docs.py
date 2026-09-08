#!/usr/bin/env python
"""Qanat action: export Tyler's cost along both paths as Plotly JSON.

Reads results.npz — the cost sampled along the Euclidean segment and along
the affine-invariant geodesic joining the same two points of the cone.

Output: docs/docs/assets/data/context_riemann_gconvexite.json
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
    FONT_SANS, FONT_MONO, C_ON,
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
times = r["times"]
CURVES = [
    ("segment euclidien", "cost_euclidean", C_ON),
    ("géodésique", "cost_geodesic", "#59bfa3"),
]


def local_minima(values):
    """Indices of the strict local minima, ends included — as in the figure."""
    interior = [
        index for index in range(1, len(values) - 1)
        if values[index] < values[index - 1] and values[index] < values[index + 1]
    ]
    if values[0] < values[1]:
        interior.insert(0, 0)
    if values[-1] < values[-2]:
        interior.append(len(values) - 1)
    return interior


TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.09, shared_yaxes=True,
    subplot_titles=[label for label, _, _ in CURVES],
)

for index, (label, key, color) in enumerate(CURVES):
    values = r[key]
    fig.add_trace(go.Scatter(
        x=times.tolist(), y=values.tolist(),
        mode="lines", name=label,
        line=dict(color=color, width=2),
        hovertemplate=f"{label}<br>t=%{{x:.2f}}<br>L=%{{y:.3f}}<extra></extra>",
        showlegend=False,
    ), row=1, col=index + 1)

    minima = local_minima(values)
    fig.add_trace(go.Scatter(
        x=times[minima].tolist(), y=values[minima].tolist(),
        mode="markers", name="minima locaux", legendgroup="minima",
        marker=dict(size=8, color=MUTED),
        hovertemplate="minimum local<br>t=%{x:.2f}<extra></extra>",
        showlegend=(index == 0),
    ), row=1, col=index + 1)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

axis_common = dict(
    tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=60, r=20, b=60, l=60),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=420, width=880,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1,
        font=dict(family=FONT_SANS, size=12, color=MUTED),
    ),
    xaxis={**axis_common, "title": dict(text="<i>t</i>", font=TITLE, standoff=12)},
    yaxis={**axis_common, "title": dict(text="<i>L</i>", font=TITLE, standoff=12)},
    xaxis2={**axis_common, "title": dict(text="<i>t</i>", font=TITLE, standoff=12)},
    yaxis2={**axis_common},
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
write_docs_provenance(out.parent, name, npz_path)
print(f"Written {out.relative_to(repo_root)}")
