#!/usr/bin/env python
"""Qanat action: export LWF under-regime results as Plotly JSON for the docs.

Reads results.npz from the run directory — condition number of SCM vs
LWF-regularised covariance as dimension d grows past N.

Output: docs/docs/assets/data/context_lwf_underregime.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly not found — run: uv add plotly --dev")

from hdrlib.core.plotly_style import (
    BG, GRID, AXIS_LINE, MUTED, INK2, ANNO,
    FONT_SANS, FONT_MONO, C_OFF, C_ON, hex_to_rgba,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--storage_path", required=True)
parser.add_argument("--name", default=None)
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[2]
storage = Path(args.storage_path)
name = args.name or storage.parent.name

npz_path = storage / "results.npz"
if not npz_path.exists():
    sys.exit(f"No results.npz in {storage}")

d = np.load(npz_path)
d_vec = d["d_vec"]
N = int(d["N"])
n_trials = int(d["n_trials"])
alpha = float(d["alpha"])
cond_scm = d["cond_scm_mean"]
cond_lwf = d["cond_lwf_mean"]

TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=d_vec.tolist(), y=cond_scm.tolist(),
    mode="markers", name="SCM",
    marker=dict(symbol="circle-open", size=7, color=C_OFF, line=dict(width=1.5)),
    hovertemplate="<b>SCM</b><br>d=%{x}<br>Cond=%{y:.1f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=d_vec.tolist(), y=cond_lwf.tolist(),
    mode="markers", name=f"LWF (α={alpha})",
    marker=dict(symbol="square-open", size=7, color=C_ON, line=dict(width=1.5)),
    hovertemplate="<b>LWF</b><br>d=%{x}<br>Cond=%{y:.1f}<extra></extra>",
))
fig.add_vline(
    x=N, line=dict(color=MUTED, width=1.2, dash="dash"),
    annotation_text=f"d = N = {N}",
    annotation=dict(
        font=dict(family=FONT_MONO, size=11, color=ANNO),
        showarrow=False, yanchor="top", yshift=-6,
    ),
)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=20, r=40, b=60, l=66),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    legend=dict(
        x=0.04, y=0.97, xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(family=FONT_SANS, size=13, color=INK2),
    ),
    xaxis=dict(
        title=dict(text="Dimension <i>d</i>", font=TITLE, standoff=14),
        type="log", tickfont=TICK,
        range=[np.log10(d_vec.min() * 0.8), np.log10(d_vec.max() * 1.2)],
        ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
        linecolor=AXIS_LINE, linewidth=1, showline=True,
        showgrid=False, zeroline=False,
    ),
    yaxis=dict(
        title=dict(text="Condition number", font=TITLE, standoff=14),
        type="log", tickfont=TICK,
        range=[np.log10(max(cond_lwf.min(), 1) * 0.5),
               np.log10(cond_scm.max() * 2)],
        ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
        linecolor=AXIS_LINE, linewidth=1, showline=True,
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
    ),
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)}")
