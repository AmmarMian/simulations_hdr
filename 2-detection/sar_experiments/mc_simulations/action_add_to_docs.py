#!/usr/bin/env python
"""Qanat action: export power-curve results as a styled Plotly JSON for the docs.

Which run is used is controlled by the experiment YAML:

  docs_figure: results/sar_mc_gauss_h1/run_8   # ← pin a specific run

If docs_figure is set, that run is always used regardless of --storage_path.
If absent, the run passed via --storage_path is used (i.e. whichever run
triggered the action).

Output: docs/docs/assets/data/{name}.json
A sidecar docs/docs/assets/data/{name}.source.txt records the run used.

The docs generator embeds the figure automatically in the experiment page.
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--storage_path", required=True,
    help="Run directory injected by qanat.")
parser.add_argument("--name", default=None,
    help="Experiment name (used as output filename). "
         "Defaults to the grandparent directory name of storage_path.")
parser.add_argument("--label", default=None,
    help="Label for this run (e.g. 'n1000'). "
         "Output: {name}.{label}.json. Without a label: {name}.json.")
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[3]
storage   = Path(args.storage_path)

# Derive experiment name: results/sar_mc_gauss_h1/run_3 → sar_mc_gauss_h1
name    = args.name  or storage.parent.name
label   = args.label
run_dir = storage

# ── Find NPZ ─────────────────────────────────────────────────────────────────
candidates = sorted(run_dir.glob("*.npz"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
npz_path = next(
    (p for p in candidates
     if set(np.load(p).keys()) >= {"T", "power_online", "power_offline", "pfa"}),
    None,
)
if npz_path is None:
    print(f"No power-curve NPZ found under {run_dir}")
    sys.exit(0)

d      = np.load(npz_path)
T      = d["T"].astype(int)
p_on   = d["power_online"]
p_off  = d["power_offline"]
pfa    = float(d["pfa"])
n_tri  = int(d["n_trials"]) if "n_trials" in d else None

print(f"Loaded {npz_path.name}  (T: {T[0]}…{T[-1]}, PFA={pfa:.0e})")

# ── Design tokens (from shared/plotly_style.py) ───────────────────────────────

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure()

TICK_FONT  = dict(family=FONT_MONO, size=12, color=MUTED)
TITLE_FONT = dict(family=FONT_SANS, size=13, color=INK2)

def _line(x, y, name, color, dash="solid", width=2.2):
    return go.Scatter(
        x=x, y=y,
        mode="lines",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate=(
            f"<b>{name}</b><br>"
            "T = %{x}<br>"
            "P<sub>D</sub> = %{y:.3f}"
            "<extra></extra>"
        ),
    )

fig.add_trace(_line(T, p_off, "Offline GLRT", C_OFF))
fig.add_trace(_line(T, p_on,  "Online GLRT",  C_ON, dash="dash", width=1.9))

# SE bands
if n_tri is not None:
    se_off = np.sqrt(np.clip(p_off * (1 - p_off), 0, None) / n_tri)
    se_on  = np.sqrt(np.clip(p_on  * (1 - p_on),  0, None) / n_tri)
    for arr, se, color in [(p_off, se_off, C_OFF), (p_on, se_on, C_ON)]:
        fig.add_trace(go.Scatter(
            x=np.concatenate([T, T[::-1]]).tolist(),
            y=np.concatenate([
                np.clip(arr + se, 0, 1),
                np.clip(arr[::-1] - se[::-1], 0, 1),
            ]).tolist(),
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.12),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

# PFA reference
pfa_exp = int(round(np.log10(pfa)))
fig.add_hline(
    y=pfa,
    line=dict(color=MUTED, width=1.2, dash="dash"),
    annotation_text=f"<i>P</i><sub>FA</sub> = 10<sup>{pfa_exp}</sup>",
    annotation_position="right",
    annotation=dict(
        font=dict(family=FONT_MONO, size=11, color=ANNO),
        showarrow=False, xanchor="left", yanchor="middle", xshift=6,
    ),
)

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    margin=dict(t=20, r=110, b=60, l=66),
    dragmode="pan",
    showlegend=True,
    legend=dict(
        x=0.04, y=0.97,
        xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(family=FONT_SANS, size=13, color=INK2),
        itemsizing="constant",
    ),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    xaxis=dict(
        title=dict(text="Number of time steps <i>T</i>", font=TITLE_FONT, standoff=14),
        type="log",
        tickformat="d",
        tickfont=TICK_FONT,
        ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
        linecolor=AXIS_LINE, linewidth=1,
        showline=True, mirror=False,
        showgrid=False, zeroline=False,
    ),
    yaxis=dict(
        title=dict(text="Detection probability <i>P</i><sub>D</sub>", font=TITLE_FONT, standoff=14),
        range=[-0.03, 1.06],
        tickformat=".1f",
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        tickfont=TICK_FONT,
        ticks="outside", ticklen=5, tickwidth=1, tickcolor=AXIS_LINE,
        linecolor=AXIS_LINE, linewidth=1,
        showline=True, mirror=False,
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False,
    ),
)

# ── Export ────────────────────────────────────────────────────────────────────
stem = f"{name}.{label}" if label else name
out  = repo_root / "docs" / "docs" / "assets" / "data" / f"{stem}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
# Sidecar records which run produced this figure
npz_abs = npz_path.resolve()
(out.parent / f"{stem}.source.txt").write_text(f"{npz_abs}\n")
print(f"Written {out.relative_to(repo_root)}")
try:
    print(f"Source:  {npz_abs.relative_to(repo_root)}")
except ValueError:
    print(f"Source:  {npz_abs}")
