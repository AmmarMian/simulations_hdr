#!/usr/bin/env python
"""Qanat action: export the three convergence curves as Plotly JSON.

Reads results.npz — the cost, gradient norm and elapsed time recorded at
every iteration by the fixed point, the Riemannian descent and the projected
Euclidean descent. Produces two panels: iterations and seconds.

Output: docs/docs/assets/data/context_riemann_convergence.json
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
    FONT_SANS, FONT_MONO, C_ON,
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
optimum = float(r["optimum"])
FLOOR = 1e-14

ALGORITHMS = [
    ("point fixe", "point_fixe", C_ON),
    ("gradient riemannien", "gradient_riemannien", "#59bfa3"),
    ("gradient euclidien projeté", "gradient_euclidien_projete", "#d4a535"),
]

TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.09, shared_yaxes=True,
    subplot_titles=["par itération", "par temps de calcul"],
)

for label, key, color in ALGORITHMS:
    gap = np.maximum(r[f"cost_{key}"] - optimum, FLOOR)
    time = r[f"time_{key}"]
    for column, abscissa in enumerate([np.arange(len(gap)), time]):
        fig.add_trace(go.Scatter(
            x=abscissa.tolist(), y=gap.tolist(),
            mode="lines", name=label, legendgroup=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>%{{x}}<br>écart=%{{y:.2e}}<extra></extra>",
            showlegend=(column == 0),
        ), row=1, col=column + 1)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=13, color=MUTED)

axis_common = dict(
    tickfont=TICK,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
    linecolor=AXIS_LINE, linewidth=1, showline=True,
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=60, r=20, b=60, l=70),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=420, width=880,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1,
        font=dict(family=FONT_SANS, size=12, color=MUTED),
    ),
    xaxis={**axis_common,
           "title": dict(text="itération", font=TITLE, standoff=12)},
    yaxis={**axis_common, "type": "log",
           "title": dict(text="<i>L</i> − <i>L</i>*", font=TITLE, standoff=12)},
    xaxis2={**axis_common, "type": "log",
            "title": dict(text="temps (s)", font=TITLE, standoff=12)},
    yaxis2={**axis_common, "type": "log"},
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)}")
