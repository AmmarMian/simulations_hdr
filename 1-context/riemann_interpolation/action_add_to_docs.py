#!/usr/bin/env python
"""Qanat action: export the three interpolation paths as Plotly JSON.

Reads results.npz — the two endpoints, the ellipses of the three paths and
the determinant along each of them. Produces a 2x2 grid: one panel per
metric, plus the determinants.

Output: docs/docs/assets/data/context_riemann_interpolation.json
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
    FONT_SANS, FONT_MONO, C_OFF, C_ON,
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
radius = float(r["radius"])
fine_times = r["fine_times"]

# Same three paths, same order and same colour roles as the exported figure.
PATHS = [
    ("euclidienne", "path_euclidienne", "det_euclidienne", C_ON),
    ("affine invariante", "path_affine_invariante", "det_affine_invariante", "#59bfa3"),
    ("log-euclidienne", "path_log_euclidienne", "det_log_euclidienne", "#d4a535"),
]


def ellipse(shape, n_points=200):
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    values, vectors = np.linalg.eigh(shape)
    return vectors @ np.diag(np.sqrt(values)) @ circle


TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=2, cols=2,
    horizontal_spacing=0.09, vertical_spacing=0.12,
    subplot_titles=[label for label, _, _, _ in PATHS] + ["déterminant"],
)

limit = 0.0
for index, (label, path_key, _, color) in enumerate(PATHS):
    row, col = index // 2 + 1, index % 2 + 1
    path = r[path_key]
    for step, matrix in enumerate(path):
        curve = ellipse(matrix)
        limit = max(limit, float(np.abs(curve).max()))
        endpoint = step in (0, len(path) - 1)
        fig.add_trace(go.Scatter(
            x=curve[0].tolist(), y=curve[1].tolist(),
            mode="lines", name="extrémités" if endpoint else label,
            legendgroup="extrémités" if endpoint else label,
            line=dict(
                color=MUTED if endpoint else color,
                width=1.6 if endpoint else 1.1,
                dash="dash" if endpoint else "solid",
            ),
            hovertemplate=f"{'extrémité' if endpoint else label}<extra></extra>",
            showlegend=(index == 0 and step in (0, 1)),
        ), row=row, col=col)

for label, _, det_key, color in PATHS:
    fig.add_trace(go.Scatter(
        x=fine_times.tolist(), y=r[det_key].tolist(),
        mode="lines", name=label, legendgroup=label,
        line=dict(color=color, width=2),
        hovertemplate=f"{label}<br>t=%{{x:.2f}}<br>det=%{{y:.3f}}<extra></extra>",
        showlegend=True,
    ), row=2, col=2)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

axis_common = dict(
    tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

limit *= 1.15
layout_axes = {}
for index in range(3):
    suffix = "" if index == 0 else str(index + 1)
    layout_axes[f"xaxis{suffix}"] = {
        **axis_common, "range": [-limit, limit],
        "title": dict(text="<i>x</i>₁", font=TITLE, standoff=12),
    }
    layout_axes[f"yaxis{suffix}"] = {
        **axis_common, "range": [-limit, limit], "scaleanchor": f"x{suffix}",
        "title": dict(text="<i>x</i>₂", font=TITLE, standoff=12),
    }
layout_axes["xaxis4"] = {
    **axis_common, "title": dict(text="<i>t</i>", font=TITLE, standoff=12),
}
layout_axes["yaxis4"] = {
    **axis_common, "title": dict(text="det", font=TITLE, standoff=12),
}

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=50, r=20, b=60, l=60),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=780, width=800,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1,
        font=dict(family=FONT_SANS, size=12, color=MUTED),
    ),
    **layout_axes,
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
write_docs_provenance(out.parent, name, npz_path)
print(f"Written {out.relative_to(repo_root)}")
