#!/usr/bin/env python
"""Qanat action: export the two error curves as Plotly JSON.

Reads results.npz — the per-trial estimation error of the SCM and of Tyler's
estimator, in Frobenius norm and in Rao distance, for each sample size.

Output: docs/docs/assets/data/context_riemann_frobenius_rao.json
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
n_samples = r["n_samples"]
ESTIMATORS = [("scm", "scm", C_ON, "circle"), ("Tyler", "Tyler", "#59bfa3", "square")]
METRICS = [("norme de Frobenius", "frobenius"), ("distance de Rao", "rao")]

TICK = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE = dict(family=FONT_SANS, size=13, color=INK2)

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.11,
    subplot_titles=[title for title, _ in METRICS],
)

for column, (_, metric) in enumerate(METRICS):
    for label, key, color, symbol in ESTIMATORS:
        errors = r[f"{key}_{metric}"].mean(axis=1)
        fig.add_trace(go.Scatter(
            x=n_samples.tolist(), y=errors.tolist(),
            mode="lines+markers", name=label, legendgroup=label,
            line=dict(color=color, width=2),
            marker=dict(size=7, symbol=symbol, color=color),
            hovertemplate=f"{label}<br>N=%{{x}}<br>erreur=%{{y:.3f}}<extra></extra>",
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
    xaxis={**axis_common, "type": "log",
           "title": dict(text="<i>N</i>", font=TITLE, standoff=12)},
    yaxis={**axis_common, "type": "log",
           "title": dict(text="erreur moyenne", font=TITLE, standoff=12)},
    xaxis2={**axis_common, "type": "log",
            "title": dict(text="<i>N</i>", font=TITLE, standoff=12)},
    yaxis2={**axis_common, "type": "log"},
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)}")
