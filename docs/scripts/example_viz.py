#!/usr/bin/env python
"""Example Plotly figure styled to match the HDR docs design system.

Run:
    uv run python docs/scripts/example_viz.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    raise SystemExit("plotly not found — run: uv add plotly --dev")

# ── Design tokens ─────────────────────────────────────────────────────────────
BG        = "rgba(0,0,0,0)"
GRID      = "rgba(187,170,148,0.35)"
AXIS_LINE = "rgba(150,130,108,0.55)"
MUTED     = "#9a8a75"
INK2      = "#5c4a35"
ANNO      = "#9a8a75"

FONT_SERIF = "EB Garamond, Georgia, serif"
FONT_SANS  = "IBM Plex Sans, system-ui, sans-serif"
FONT_MONO  = "IBM Plex Mono, ui-monospace, monospace"

C1 = "#b85c2a"   # online  — warm rust
C2 = "#4a7c9e"   # offline — steel blue
C3 = "#7a9e5c"   # scm     — sage green

# ── Data ──────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
T   = np.logspace(np.log10(5), np.log10(600), 45).astype(int)

def sigmoid(t, t50, slope=2.6):
    return 1 / (1 + np.exp(-slope * (np.log(t) - np.log(t50))))

def noise(n):
    return rng.normal(0, 0.012, n)

P_online  = np.clip(sigmoid(T, 60)  + noise(len(T)), 0, 1)
P_offline = np.clip(sigmoid(T, 38)  + noise(len(T)), 0, 1)
P_scm     = np.clip(sigmoid(T, 110) + noise(len(T)), 0, 1)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure()

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

fig.add_trace(_line(T, P_online,  "Online Kronecker GLRT",  C1))
fig.add_trace(_line(T, P_offline, "Offline Kronecker GLRT", C2))
fig.add_trace(_line(T, P_scm,     "SCM GLRT (baseline)",    C3, dash="dot", width=1.8))

# PFA reference
fig.add_hline(
    y=0.01,
    line=dict(color=MUTED, width=1.2, dash="dash"),
    annotation_text="<i>P</i><sub>FA</sub> = 10<sup>−2</sup>",
    annotation_position="right",
    annotation=dict(
        font=dict(family=FONT_MONO, size=11, color=ANNO),
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=6,
    ),
)

# ── Layout ────────────────────────────────────────────────────────────────────
TICK_FONT  = dict(family=FONT_MONO,  size=12, color=MUTED)
TITLE_FONT = dict(family=FONT_SANS,  size=13, color=INK2)

fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    margin=dict(t=20, r=110, b=60, l=66),

    dragmode="pan",   # default interaction is pan; scroll = zoom

    showlegend=True,
    legend=dict(
        x=0.04, y=0.97,
        xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(family=FONT_SANS,  size=13, color=INK2),
        itemsizing="constant",
    ),

    font=dict(family=FONT_SANS,  size=13, color=INK2),

    xaxis=dict(
        title=dict(text="Number of time steps <i>T</i>", font=TITLE_FONT, standoff=14),
        type="log",
        tickformat="d",
        tickvals=[5, 10, 20, 50, 100, 200, 500],
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
OUT = Path(__file__).parent.parent / "docs" / "assets" / "data" / "example_power_curve.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(fig.to_json())
print(f"Written {OUT}")
