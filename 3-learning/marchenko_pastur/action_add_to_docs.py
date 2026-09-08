#!/usr/bin/env python
"""Qanat action: export the Marchenko-Pastur panels as Plotly JSON.

Reads results.npz — the pooled eigenvalues of every concentration ratio — and
rebuilds one panel per ratio: histogram of the sample covariance eigenvalues,
the theoretical density on top, and the true spectrum as a vertical line.

The density is recomputed here rather than stored: it is a closed-form scalar
formula of the ratio alone, so storing it would only risk it drifting out of
step with the eigenvalues it is drawn against.

Output: docs/docs/assets/data/learning_marchenko_pastur.json
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
ratios = [float(x) for x in r["ratios"]]
d = int(r["n_features"])
n_bins = int(r["n_bins"])


def density(grid, ratio):
    """Marchenko-Pastur density, zero outside its support."""
    lower = (1.0 - np.sqrt(ratio)) ** 2
    upper = (1.0 + np.sqrt(ratio)) ** 2
    out = np.zeros_like(grid)
    inside = (grid > lower) & (grid < upper)
    out[inside] = np.sqrt(
        (upper - grid[inside]) * (grid[inside] - lower)
    ) / (2.0 * np.pi * ratio * grid[inside])
    return out


TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=len(ratios), horizontal_spacing=0.07,
    subplot_titles=[f"c = {ratio:g}" for ratio in ratios],
)

for index, ratio in enumerate(ratios):
    column = index + 1
    eigenvalues = r[f"eigenvalues_c{index}"]
    n_samples = int(r[f"n_samples_c{index}"])
    upper_plot = (1.0 + np.sqrt(ratio)) ** 2 * 1.15

    counts, edges = np.histogram(
        eigenvalues, bins=n_bins, range=(0.0, upper_plot), density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig.add_trace(
        go.Bar(
            x=centers, y=counts, width=edges[1] - edges[0],
            marker=dict(color=hex_to_rgba(C_OFF, 0.55), line=dict(width=0)),
            name="valeurs propres de la SCM",
            legendgroup="hist", showlegend=index == 0,
            hovertemplate="λ = %{x:.3f}<br>densité = %{y:.3f}<extra></extra>",
        ),
        row=1, col=column,
    )

    grid = np.linspace(1e-4, upper_plot, 600)
    fig.add_trace(
        go.Scatter(
            x=grid, y=density(grid, ratio), mode="lines",
            line=dict(color=C_ON, width=2),
            name="loi de Marchenko-Pastur",
            legendgroup="mp", showlegend=index == 0,
            hovertemplate="λ = %{x:.3f}<br>ρ = %{y:.3f}<extra></extra>",
        ),
        row=1, col=column,
    )

    # The true spectrum is one point; the whole reading of the figure is the
    # gap between it and the histogram, so it is drawn rather than described.
    fig.add_trace(
        go.Scatter(
            x=[1.0, 1.0], y=[0.0, 1.25 * float(counts.max())], mode="lines",
            line=dict(color=INK2, width=1.2, dash="dash"),
            name="spectre vrai", legendgroup="truth", showlegend=index == 0,
            hoverinfo="skip",
        ),
        row=1, col=column,
    )

    axis_common = dict(
        tickfont=TICK,
        ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
        linecolor=AXIS_LINE, linewidth=1, showline=True,
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
    )
    fig.update_xaxes(
        **axis_common, range=[0.0, upper_plot],
        title=dict(text="λ", font=TITLE, standoff=10),
        row=1, col=column,
    )
    # At c = 1 the density diverges at the origin; the frame follows the
    # histogram so that the three panels stay comparable.
    fig.update_yaxes(
        **axis_common, range=[0.0, 1.25 * float(counts.max())],
        title=dict(text="densité" if index == 0 else None,
                   font=TITLE, standoff=10),
        row=1, col=column,
    )

    fig.layout.annotations[index].text = (
        f"c = {ratio:g}  ·  N = {n_samples}"
    )

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=70, r=20, b=60, l=60),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=400, width=980, bargap=0,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.14, xanchor="right", x=1,
        font=dict(family=FONT_SANS, size=12, color=MUTED),
    ),
    title=dict(
        text=f"d = {d}, spectre vrai constant égal à 1",
        font=dict(family=FONT_SANS, size=12, color=MUTED),
        x=0.0, xanchor="left", y=0.02, yanchor="bottom",
    ),
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
write_docs_provenance(out.parent, name, npz_path)
print(f"Written {out.relative_to(repo_root)}")
