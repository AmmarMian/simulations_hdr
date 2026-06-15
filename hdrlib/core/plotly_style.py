"""Plotly design tokens for the HDR docs design system.

Import from action_add_to_docs.py scripts via:

    from hdrlib.core.plotly_style import BG, GRID, AXIS_LINE, MUTED, INK2, ANNO
    from hdrlib.core.plotly_style import FONT_SANS, FONT_MONO, C_OFF, C_ON, hex_to_rgba
"""

BG        = "rgba(0,0,0,0)"
GRID      = "rgba(187,170,148,0.35)"
AXIS_LINE = "rgba(150,130,108,0.55)"
MUTED     = "#9a8a75"
INK2      = "#5c4a35"
ANNO      = "#9a8a75"
FONT_SANS = "IBM Plex Sans, system-ui, sans-serif"
FONT_MONO = "IBM Plex Mono, ui-monospace, monospace"
C_OFF     = "#4a7c9e"   # offline — steel blue
C_ON      = "#b85c2a"   # online  — warm rust


def hex_to_rgba(h: str, a: float) -> str:
    """Convert a hex colour string to an rgba(...) CSS string."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"
