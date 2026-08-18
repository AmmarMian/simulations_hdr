#!/usr/bin/env python
"""Qanat action: export the circularity panels as Plotly JSON.

Reads results.npz — one draw per pseudo-covariance, the matching concentration
ellipses, and the single reference circle predicted by the covariance alone.
Produces a 2-column grid of subplots.

Output: docs/docs/assets/data/context_complex_circularity.json
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
ellipses = r["ellipses"]
reference = r["reference"]
rho = r["rho"]
phase = r["phase"]
n_samples = int(r["n_samples"])
gamma = float(r["gamma"])


def title(index):
    if rho[index] == 0:
        return "C = 0"
    return f"|C|/Γ = {rho[index]:g}, arg C = {phase[index] / np.pi:.2f}π"


TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

n_panels = len(rho)
n_cols = min(2, n_panels)
n_rows = int(np.ceil(n_panels / n_cols))
fig = make_subplots(
    rows=n_rows, cols=n_cols,
    horizontal_spacing=0.09, vertical_spacing=0.13,
    subplot_titles=[title(i) for i in range(n_panels)],
)

limit = 1.15 * max(np.quantile(np.abs(data), 0.999) for data in samples)

for index in range(n_panels):
    row, col = index // n_cols + 1, index % n_cols + 1
    data = samples[index]

    fig.add_trace(go.Scatter(
        x=data[:, 0].tolist(), y=data[:, 1].tolist(),
        mode="markers", name="observations",
        marker=dict(
            size=4, color="rgba(0,0,0,0)",
            line=dict(width=0.8, color=hex_to_rgba(C_OFF, 0.7)),
        ),
        hovertemplate="Re z=%{x:.2f}<br>Im z=%{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=reference[0].tolist(), y=reference[1].tolist(),
        mode="lines", name="prédit par Γ seule", legendgroup="reference",
        line=dict(color=MUTED, width=1.6, dash="dash"),
        hovertemplate="prédit par Γ seule<extra></extra>",
        showlegend=(index == 0),
    ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=ellipses[index][0].tolist(), y=ellipses[index][1].tolist(),
        mode="lines", name="concentration réelle", legendgroup="actual",
        line=dict(color=C_ON, width=2.0),
        hovertemplate="concentration réelle<extra></extra>",
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
            text="Re <i>z</i>" if index // n_cols == n_rows - 1 else "",
            font=TITLE, standoff=12,
        ),
    }
    layout_axes[f"yaxis{suffix}"] = {
        **axis_common,
        "scaleanchor": f"x{suffix}",
        "title": dict(
            text="Im <i>z</i>" if index % n_cols == 0 else "",
            font=TITLE, standoff=12,
        ),
    }

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=70, r=20, b=60, l=60),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=760, width=800,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1,
        font=dict(family=FONT_SANS, size=12, color=MUTED),
    ),
    **layout_axes,
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)} (Γ={gamma:g}, N={n_samples} per panel)")
