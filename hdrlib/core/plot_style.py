"""
Dark serif theme for all project figures.

Call :func:`apply_style` once per process to configure matplotlib globally.

    from hdrlib.core.plot_style import apply_style
    apply_style()                  # dark theme, no LaTeX
    apply_style(use_latex=True)    # dark theme + LaTeX (requires a LaTeX install)

The module also exports :data:`DARK_STYLE_DICT` — a plain dict of rcParams —
so auto-generated standalone scripts can embed the theme without importing
this module.
"""
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_BG      = "#14141a"   # very dark blue-black
_TEXT    = "#dde3f0"   # cool off-white
_GRID    = "#272733"   # barely-there grid
_SPINE   = "#48485e"   # muted spine / tick marks
_LEGEND  = "#1c1c27"   # legend box fill

# Vivid colours tuned for dark backgrounds — replaces the dull default cycle.
_CYCLE = [
    "#5ca8d3",   # steel-sky blue
    "#e06b6b",   # coral-red
    "#59bfa3",   # teal-green
    "#d4a535",   # warm gold
    "#a57bc5",   # violet
    "#6bbf7a",   # mint green
    "#e08a5a",   # copper-orange
]

# ---------------------------------------------------------------------------
# The rcParams dict (importable for use in templates)
# ---------------------------------------------------------------------------
DARK_STYLE_DICT: dict = {
    # backgrounds
    "figure.facecolor":   _BG,
    "axes.facecolor":     _BG,
    "savefig.facecolor":  _BG,

    # text & ticks
    "text.color":         _TEXT,
    "axes.labelcolor":    _TEXT,
    "axes.titlecolor":    _TEXT,
    "xtick.color":        _SPINE,
    "ytick.color":        _SPINE,
    "xtick.labelcolor":   _TEXT,
    "ytick.labelcolor":   _TEXT,

    # spines
    "axes.edgecolor":     _SPINE,
    "axes.spines.top":    False,
    "axes.spines.right":  False,

    # grid
    "axes.grid":          True,
    "axes.grid.which":    "both",
    "grid.color":         _GRID,
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "grid.alpha":         1.0,

    # font
    "font.family":        "serif",
    "font.serif":         ["STIXTwoText", "STIX Two Text", "DejaVu Serif",
                           "Times New Roman", "serif"],
    "mathtext.fontset":   "stix",
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,

    # lines & markers
    "lines.linewidth":    1.8,
    "lines.markersize":   5,
    "patch.linewidth":    0.8,

    # legend
    "legend.facecolor":   _LEGEND,
    "legend.edgecolor":   _SPINE,
    "legend.framealpha":  0.9,
    "legend.borderpad":   0.6,

    # colour cycle
    "axes.prop_cycle":    mpl.cycler(color=_CYCLE),

    # saving
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,

    # LaTeX off by default
    "text.usetex":        False,
}

# ---------------------------------------------------------------------------
# Compact string for embedding in auto-generated standalone scripts
# ---------------------------------------------------------------------------
# A literal Python code block (no imports needed beyond matplotlib) that can
# be pasted verbatim into a generated plot script.
EMBEDDED_STYLE_CODE: str = """\
import matplotlib as _mpl
_mpl.rcParams.update({
    "figure.facecolor": "#14141a", "axes.facecolor": "#14141a",
    "savefig.facecolor": "#14141a",
    "text.color": "#dde3f0", "axes.labelcolor": "#dde3f0", "axes.titlecolor": "#dde3f0",
    "xtick.color": "#48485e", "ytick.color": "#48485e",
    "xtick.labelcolor": "#dde3f0", "ytick.labelcolor": "#dde3f0",
    "axes.edgecolor": "#48485e", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.which": "both",
    "grid.color": "#272733", "grid.linewidth": 0.6, "grid.linestyle": "--",
    "font.family": "serif",
    "font.serif": ["STIXTwoText", "STIX Two Text", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 1.8, "lines.markersize": 5,
    "legend.facecolor": "#1c1c27", "legend.edgecolor": "#48485e",
    "legend.framealpha": 0.9, "savefig.dpi": 200, "savefig.bbox": "tight",
})
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_style(use_latex: bool = False) -> None:
    """Apply the dark serif theme globally.

    Parameters
    ----------
    use_latex : bool
        Enable LaTeX rendering (requires a working LaTeX installation).
        With LaTeX the font becomes Computer Modern; math is typeset by
        LaTeX.  Without it, STIX Two Text provides a near-identical serif
        look through matplotlib's mathtext engine.
    """
    params = dict(DARK_STYLE_DICT)   # shallow copy — don't mutate the module dict

    if use_latex:
        params["text.usetex"] = True
        params["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    mpl.rcParams.update(params)
