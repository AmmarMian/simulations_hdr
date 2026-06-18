#!/usr/bin/env python
"""Qanat action: export covariance matrix examples as Plotly JSON for the docs.

Deterministic and cheap — matrices are recomputed directly.

Output: docs/docs/assets/data/context_example_covariances.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import toeplitz

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly not found — run: uv add plotly --dev")

from hdrlib.core.plotly_style import BG, MUTED, INK2, FONT_SANS, FONT_MONO

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--storage_path", required=True,
    help="Run directory injected by qanat.")
parser.add_argument("--name", default=None,
    help="Experiment name (output filename). Defaults to grandparent dir.")
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[2]
storage = Path(args.storage_path)
name = args.name or storage.parent.name

# ── Covariance matrices (deterministic) ──────────────────────────────────────
d = 7
rho = 0.8

cov_id = np.eye(d)
cov_toeplitz = toeplitz(np.power(rho, np.arange(0, d)))

np.random.seed(0)
cov_rd = np.empty((d, d))
cov_rd[np.tril_indices(d)] = 2 * np.random.rand(int(d * (d + 1) / 2)) - 1
cov_rd[np.triu_indices(d)] = cov_rd[np.tril_indices(d)]
cov_rd[np.diag_indices(d)] = 1
cov_rd = 0.5 * (cov_rd + cov_rd.T)

matrices = [cov_id, cov_toeplitz, cov_rd]
titles = ["Identity", "Toeplitz(ρ)", "Random"]

# ── Figure ───────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=3,
    horizontal_spacing=0.08,
    subplot_titles=titles,
)

for i, mat in enumerate(matrices):
    fig.add_trace(
        go.Heatmap(
            z=mat[::-1],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                len=0.9,
                thickness=12,
                tickfont=dict(family=FONT_MONO, size=10, color=MUTED),
            ),
            hovertemplate="i=%{x}, j=%{y}<br>value=%{z:.3f}<extra></extra>",
        ),
        row=1, col=i + 1,
    )

axis_common = dict(
    showgrid=False,
    zeroline=False,
    constrain="domain",
    scaleanchor="y" if True else None,
    tickfont=dict(family=FONT_MONO, size=11, color=MUTED),
    title=dict(text="variable i", font=dict(family=FONT_SANS, size=12, color=INK2)),
)
yaxis_common = dict(
    showgrid=False,
    zeroline=False,
    tickfont=dict(family=FONT_MONO, size=11, color=MUTED),
)

for i in range(3):
    xref = f"xaxis{i + 1}" if i > 0 else "xaxis"
    yref = f"yaxis{i + 1}" if i > 0 else "yaxis"
    fig.update_layout(**{
        xref: axis_common,
        yref: {
            **yaxis_common,
            "scaleanchor": f"x{i + 1}" if i > 0 else "x",
            "title": dict(
                text="variable j" if i == 0 else "",
                font=dict(family=FONT_SANS, size=12, color=INK2),
            ),
        },
    })

fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    margin=dict(t=40, r=10, b=50, l=50),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=350,
    width=950,
)
for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=12, color=MUTED)

# ── Export ───────────────────────────────────────────────────────────────────
out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"deterministic\n")
print(f"Written {out.relative_to(repo_root)}")
