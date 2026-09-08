#!/usr/bin/env python
"""Qanat action: export the cloud of covariances and its three means as Plotly JSON.

Reads results.npz — the cloud, the arithmetic, Fréchet and log-Euclidean
means. Produces two panels: the concentration ellipses and the determinants.

Output: docs/docs/assets/data/context_riemann_moyennes.json
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
cloud = r["cloud"]
radius = float(r["radius"])

# Drawing order is the order of the exported figure: the dashed log-Euclidean
# mean comes last so that it stays visible where it lands on the Fréchet one.
MEANS = [
    ("arithmétique", "mean_arithmetique", C_ON, "solid"),
    ("de Fréchet", "mean_de_Frechet", "#59bfa3", "solid"),
    ("log-euclidienne", "mean_log_euclidienne", "#d4a535", "dash"),
]


def ellipse(shape, n_points=200):
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    values, vectors = np.linalg.eigh(shape)
    return vectors @ np.diag(np.sqrt(values)) @ circle


TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.11,
    subplot_titles=["ellipses de concentration", "déterminants"],
)

for index, matrix in enumerate(cloud):
    curve = ellipse(matrix)
    fig.add_trace(go.Scatter(
        x=curve[0].tolist(), y=curve[1].tolist(),
        mode="lines", name="échantillon", legendgroup="échantillon",
        line=dict(color=hex_to_rgba(C_OFF, 0.7), width=1),
        hovertemplate="échantillon<extra></extra>",
        showlegend=(index == 0),
    ), row=1, col=1)

for label, key, color, dash in MEANS:
    curve = ellipse(r[key])
    fig.add_trace(go.Scatter(
        x=curve[0].tolist(), y=curve[1].tolist(),
        mode="lines", name=label, legendgroup=label,
        line=dict(color=color, width=2.2, dash=dash),
        hovertemplate=f"{label}<extra></extra>",
    ), row=1, col=1)

determinants = np.sort(np.array([np.linalg.det(matrix) for matrix in cloud]))
fig.add_trace(go.Scatter(
    x=np.arange(1, len(determinants) + 1).tolist(), y=determinants.tolist(),
    mode="markers", name="échantillon", legendgroup="échantillon",
    marker=dict(size=6, color=hex_to_rgba(C_OFF, 0.9)),
    hovertemplate="det=%{y:.3f}<extra></extra>", showlegend=False,
), row=1, col=2)

for label, key, color, dash in MEANS:
    value = float(np.linalg.det(r[key]))
    fig.add_trace(go.Scatter(
        x=[1, len(determinants)], y=[value, value],
        mode="lines", name=label, legendgroup=label,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"{label}<br>det=%{{y:.3f}}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

axis_common = dict(
    tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

limit = 1.1 * float(
    np.quantile(np.abs(np.stack([ellipse(matrix) for matrix in cloud])), 0.99)
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
    xaxis={**axis_common, "range": [-limit, limit],
           "title": dict(text="<i>x</i>₁", font=TITLE, standoff=12)},
    yaxis={**axis_common, "range": [-limit, limit], "scaleanchor": "x",
           "title": dict(text="<i>x</i>₂", font=TITLE, standoff=12)},
    xaxis2={**axis_common,
            "title": dict(text="matrices, triées par déterminant",
                          font=TITLE, standoff=12)},
    yaxis2={**axis_common, "type": "log",
            "title": dict(text="det", font=TITLE, standoff=12)},
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)}")
