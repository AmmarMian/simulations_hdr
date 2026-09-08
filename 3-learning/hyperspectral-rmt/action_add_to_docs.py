#!/usr/bin/env python
"""Qanat action: export the segmentation maps as Plotly JSON.

Not a copy of the sibling experiment's action: this results.npz stores no list
of the methods it ran (they are recovered from the map_* keys) and its
scores.json is the mapping itself rather than one nested under "scores".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly not found — run: uv add plotly --dev")

from hdrlib.core.plotly_style import (
    BG, AXIS_LINE, MUTED, INK2,
    FONT_SANS, FONT_MONO,
)

# The order the chapter compares them in: the two shrinkage baselines between
# the uncorrected mean and the corrected one, so the correction is read against
# its neighbours rather than across the figure.
METHOD_ORDER = ("SCM", "LW", "LW-NL", "RMT")

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

r = np.load(npz_path, allow_pickle=False)
n_classes = int(r["n_classes"])
truth = r["truth"]

# main.py saves one ``map_<method>`` array per method it ran and no list of the
# methods themselves, so the run's own keys are the only record of what was in
# it. Sorting by METHOD_ORDER keeps the panels in the chapter's order whatever
# subset --methods selected; anything unrecognised is kept, after, rather than
# dropped silently.
present = [key[len("map_"):] for key in r.files if key.startswith("map_")]
methods = [m for m in METHOD_ORDER if m in present]
methods += sorted(m for m in present if m not in METHOD_ORDER)
if not methods:
    sys.exit(f"No map_* arrays in {npz_path}")

# Unlike the sibling experiment, this one writes scores.json as the mapping
# itself, with no enclosing "scores" key.
scores = {}
scores_path = storage / "scores.json"
if scores_path.exists():
    scores = json.loads(scores_path.read_text())

# A discrete scale: these are class labels, not a field, so a perceptual
# gradient would suggest an order between crops that has none. Plotly wants the
# scale as breakpoints in [0, 1], so each class gets an equal band of it.
palette = [
    "#f4f1ea", "#c0504d", "#dea11f", "#59bfa3", "#4a7c9e", "#9a6fb0",
    "#7d9a4a", "#c77c4e", "#5c8fa8", "#b8556f", "#8a7f5c", "#6b9e78",
    "#a8683f", "#4f6d8c", "#bf8fa0", "#7a6a92", "#93a05a",
]
colours = [palette[i % len(palette)] for i in range(n_classes + 1)]
step = 1.0 / (n_classes + 1)
colorscale = []
for index, colour in enumerate(colours):
    colorscale.append([index * step, colour])
    colorscale.append([(index + 1) * step, colour])

panels = ["vérité terrain"] + methods
titles = []
for panel in panels:
    if panel == "vérité terrain":
        titles.append("vérité terrain")
    else:
        entry = scores.get(panel, {})
        if entry:
            titles.append(
                f"{panel} — {entry['accuracy']:.3f} / {entry['mIoU']:.3f}"
            )
        else:
            titles.append(panel)

fig = make_subplots(
    rows=1, cols=len(panels), horizontal_spacing=0.02, subplot_titles=titles,
)

for column, panel in enumerate(panels, start=1):
    image = truth if panel == "vérité terrain" else r[f"map_{panel}"]
    fig.add_trace(
        go.Heatmap(
            z=image,
            colorscale=colorscale,
            zmin=0, zmax=n_classes,
            showscale=False,
            hovertemplate="ligne %{y}, colonne %{x}<br>classe %{z}<extra></extra>",
        ),
        row=1, col=column,
    )

axis_common = dict(
    showgrid=False, zeroline=False, showticklabels=False,
    ticks="", linecolor=AXIS_LINE, linewidth=1, showline=False,
    constrain="domain",
)

for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=12, color=MUTED)

fig.update_xaxes(**axis_common)
# The maps are images: one square pixel, and the row axis pointing down.
fig.update_yaxes(**axis_common, autorange="reversed", scaleanchor="x", scaleratio=1)

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    margin=dict(t=60, r=20, b=30, l=20),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=340, width=240 * len(panels),
    title=dict(
        text=f"{str(r['scene'])} · fenêtre {int(r['window_size'])}×"
             f"{int(r['window_size'])} sur {int(r['n_features'])} composantes, "
             f"c = {float(r['concentration']):.2f}, {n_classes} classes · "
             f"pas {int(r['stride'])}, {int(r['n_init'])} redémarrages · "
             "titres : exactitude / mIoU",
        font=dict(family=FONT_MONO, size=11, color=MUTED),
        x=0.0, xanchor="left", y=0.015, yanchor="bottom",
    ),
)

out = repo_root / "docs" / "docs" / "assets" / "data" / f"{name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{name}.source.txt").write_text(f"{npz_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)}")
