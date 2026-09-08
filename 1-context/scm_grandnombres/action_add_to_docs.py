#!/usr/bin/env python
"""Qanat action: export SCM grand-nombre convergence as Plotly JSON for the docs.

Reads results.npz — mean and covariance estimation error as N → ∞.
Produces a two-panel figure (subplots).

Output: docs/docs/assets/data/context_scm_grandnombres.json
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

r = np.load(npz_path)
N_vec = r["N_vec"]
d = int(r["d"])
n_trials = int(r["n_trials"])
err_mean = r["error_mean_mean"]
err_mean_std = r["error_mean_std"]
err_cov = r["error_cov_mean"]
err_cov_std = r["error_cov_std"]

TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.12,
    subplot_titles=["Mean estimation error", "Covariance estimation error"],
)

for i, (y, y_std, label, col) in enumerate([
    (err_mean, err_mean_std, "Mean error", C_OFF),
    (err_cov, err_cov_std, "Cov error", C_ON),
]):
    fig.add_trace(go.Scatter(
        x=N_vec.tolist(), y=y.tolist(),
        mode="markers", name=label,
        marker=dict(symbol="circle-open", size=7, color=col, line=dict(width=1.5)),
        hovertemplate=f"<b>{label}</b><br>N=%{{x}}<br>err=%{{y:.4f}}<extra></extra>",
        showlegend=False,
    ), row=1, col=i + 1)
    # Error bars as band
    upper = (y + y_std).tolist()
    lower = np.clip(y - y_std, 0, None).tolist()
    fig.add_trace(go.Scatter(
        x=N_vec.tolist() + N_vec[::-1].tolist(),
        y=upper + lower[::-1],
        fill="toself",
        fillcolor=hex_to_rgba(col, 0.12),
        line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=i + 1)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=12, color=MUTED)

axis_x = dict(
    title=dict(text="Number of samples <i>N</i>", font=TITLE, standoff=14),
    type="log", tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=False, zeroline=False,
)
axis_y = dict(
    tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=40, r=20, b=60, l=66),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=400, width=900,
    xaxis=axis_x, xaxis2=axis_x,
    yaxis={**axis_y, "title": dict(text="‖μ̂ − μ‖₂", font=TITLE, standoff=14)},
    yaxis2={**axis_y, "title": dict(text="‖Σ̂ − Σ‖_F", font=TITLE, standoff=14)},
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)}")
