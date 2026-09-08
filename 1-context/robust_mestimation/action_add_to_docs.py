#!/usr/bin/env python
"""Qanat action: export the estimated concentration ellipses as Plotly JSON.

Reads results.npz — the true shape matrix, one draw per model, and the SCM,
model-MLE and Tyler estimates. Produces a 2x2 grid of subplots.

Output: docs/docs/assets/data/context_robust_mestimation.json
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
names = [str(n) for n in r["names"]]
samples = r["samples"]
shape_true = r["shape_true"]
radius = float(r["radius"])
n_samples = int(r["n_samples"])

TITLES = {
    "gaussian": "𝒩",
    "student": f"t, ν = {float(r['dof_student']):g}",
    "k": f"K, ν = {float(r['dof_k']):g}",
    "gengauss": f"𝒢𝒢, s = {float(r['shape_gengauss']):g}",
}

# Same colour roles as the exported figure: truth dashed, then the estimators.
CURVES = [
    ("true", "vraie ξ", MUTED, "dash"),
    ("scm", "scm", C_ON, "solid"),
    ("mle", "mle", "#59bfa3", "solid"),
    ("tyler", "Tyler", "#d4a535", "solid"),
]


def ellipse(shape, n_points=300):
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    return np.linalg.cholesky(shape) @ circle


TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

n_panels = len(names)
n_cols = min(2, n_panels)
n_rows = int(np.ceil(n_panels / n_cols))

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    horizontal_spacing=0.09, vertical_spacing=0.11,
    subplot_titles=[TITLES.get(n, n) for n in names],
)

limit = 1.1 * max(np.quantile(np.abs(data), 0.995) for data in samples)

for index, model in enumerate(names):
    row, col = index // n_cols + 1, index % n_cols + 1
    data = samples[index]

    fig.add_trace(go.Scatter(
        x=data[:, 0].tolist(), y=data[:, 1].tolist(),
        mode="markers", name="observations",
        marker=dict(
            size=5, color="rgba(0,0,0,0)",
            line=dict(width=1.0, color=hex_to_rgba(C_OFF, 0.8)),
        ),
        hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    for key, label, color, dash in CURVES:
        shape = shape_true if key == "true" else r[f"{key}_{model}"]
        curve = ellipse(shape)
        fig.add_trace(go.Scatter(
            x=curve[0].tolist(), y=curve[1].tolist(),
            mode="lines", name=label, legendgroup=key,
            line=dict(color=color, width=1.8, dash=dash),
            hovertemplate=f"{label}<extra></extra>",
            showlegend=(index == 0),
        ), row=row, col=col)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

axis_common = dict(
    range=[-limit, limit], tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

layout_axes = {}
for index in range(n_panels):
    suffix = "" if index == 0 else str(index + 1)
    layout_axes[f"xaxis{suffix}"] = {
        **axis_common,
        "title": dict(
            text="<i>x</i>₁" if index // n_cols == n_rows - 1 else "",
            font=TITLE, standoff=12,
        ),
    }
    layout_axes[f"yaxis{suffix}"] = {
        **axis_common,
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
print(f"Written {out.relative_to(repo_root)} (N={n_samples} per panel)")
