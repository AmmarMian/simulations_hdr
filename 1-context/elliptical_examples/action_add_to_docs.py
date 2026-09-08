#!/usr/bin/env python
"""Qanat action: export the elliptical distribution panels as Plotly JSON.

Reads results.npz — the shared scatter matrix, the draws and the isodensity
curves for each distribution. Produces a 2x2 grid of subplots.

Output: docs/docs/assets/data/context_elliptical_examples.json
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
    FONT_SANS, FONT_MONO, C_OFF, C_ON, hex_to_rgba,
)

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
samples = r["samples"]
contours = r["contours"]
names = [str(n) for n in r["names"]]
probabilities = r["probabilities"]
rho = float(r["rho"])
n_samples = int(r["n_samples"])

TITLES = {
    "gaussian": "𝒩",
    "student": f"t, ν = {float(r['dof_student']):.0f}",
    "k": f"K, ν = {float(r['dof_k']):.0f}",
    "gengauss": f"𝒢𝒢, s = {float(r['shape_gengauss'])}",
}

TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

# 2x2 grid: squarer than a single row, and it matches the dissertation figure.
n_panels = len(names)
n_cols = min(2, n_panels)
n_rows = int(np.ceil(n_panels / n_cols))

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    horizontal_spacing=0.09, vertical_spacing=0.11,
    subplot_titles=[TITLES.get(n, n) for n in names],
)

limit = 1.1 * float(np.abs(contours[:, -1]).max())

for index, name_key in enumerate(names):
    row, col = index // n_cols + 1, index % n_cols + 1
    data = samples[index]

    fig.add_trace(go.Scatter(
        x=data[:, 0].tolist(), y=data[:, 1].tolist(),
        mode="markers", name="samples",
        marker=dict(
            size=6, color="rgba(0,0,0,0)",
            line=dict(width=1.1, color=hex_to_rgba(C_OFF, 0.85)),
        ),
        hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    for level, probability in enumerate(probabilities):
        curve = contours[index, level]
        fig.add_trace(go.Scatter(
            x=curve[0].tolist(), y=curve[1].tolist(),
            mode="lines",
            line=dict(color=C_ON, width=1.5),
            hovertemplate=f"isodensity {probability:.0%}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

axis_x = dict(
    range=[-limit, limit], tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)
axis_y = dict(axis_x)

layout_axes = {}
for index in range(n_panels):
    suffix = "" if index == 0 else str(index + 1)
    layout_axes[f"xaxis{suffix}"] = {
        **axis_x,
        "title": dict(
            text="<i>x</i>₁" if index // n_cols == n_rows - 1 else "",
            font=TITLE, standoff=12,
        ),
    }
    layout_axes[f"yaxis{suffix}"] = {
        **axis_y,
        "scaleanchor": f"x{suffix}",
        "title": dict(
            text="<i>x</i>₂" if index % n_cols == 0 else "",
            font=TITLE, standoff=12,
        ),
    }

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=50, r=20, b=60, l=60),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=760, width=760,
    **layout_axes,
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)} (rho={rho}, {n_samples} samples per panel)")
