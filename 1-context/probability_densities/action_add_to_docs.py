#!/usr/bin/env python
"""Qanat action: export Gaussian isodensity contours as Plotly JSON for the docs.

Reads results.npz — the three covariance regimes and their samples.
Produces a three-panel figure (subplots) with samples and isodensity ellipses.

Output: docs/docs/assets/data/context_gaussian_isocontours.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

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
covariances = r["covariances"]
samples = r["samples"]
probabilities = r["probabilities"]
n_samples = int(r["n_samples"])
rho = float(r["rho"])
condition = float(r["condition"])

titles = ["Σ = I₂", f"correlated (ρ = {rho})", f"ill-conditioned (k = {condition:.0f})"]


def ellipse(cov, probability, n_points=200):
    """Isodensity ellipse enclosing a given probability mass."""
    radius = np.sqrt(chi2.ppf(probability, df=2))
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = radius * np.stack([np.cos(angles), np.sin(angles)])
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ circle


TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=3,
    horizontal_spacing=0.07,
    subplot_titles=titles,
)

limit = 1.15 * float(np.abs(samples).max())

for i, (cov, data) in enumerate(zip(covariances, samples)):
    fig.add_trace(go.Scatter(
        x=data[:, 0].tolist(), y=data[:, 1].tolist(),
        mode="markers", name="samples",
        marker=dict(size=4, color=hex_to_rgba(C_OFF, 0.45), line=dict(width=0)),
        hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=i + 1)

    for probability in probabilities:
        curve = ellipse(cov, probability)
        fig.add_trace(go.Scatter(
            x=curve[0].tolist(), y=curve[1].tolist(),
            mode="lines", name=f"{probability:.0%}",
            line=dict(color=C_ON, width=1.5),
            hovertemplate=f"isodensity {probability:.0%}<extra></extra>",
            showlegend=False,
        ), row=1, col=i + 1)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=12, color=MUTED)

axis_x = dict(
    title=dict(text="<i>x</i>₁", font=TITLE, standoff=12),
    range=[-limit, limit], tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)
axis_y = dict(
    range=[-limit, limit], tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=40, r=20, b=60, l=60),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=380, width=1000,
    xaxis=axis_x, xaxis2=axis_x, xaxis3=axis_x,
    yaxis={**axis_y, "scaleanchor": "x",
           "title": dict(text="<i>x</i>₂", font=TITLE, standoff=12)},
    yaxis2={**axis_y, "scaleanchor": "x2"},
    yaxis3={**axis_y, "scaleanchor": "x3"},
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
write_docs_provenance(out.parent, name, npz_path)
print(f"Written {out.relative_to(repo_root)} ({n_samples} samples per panel)")
