#!/usr/bin/env python
"""Qanat action: export a GLRT detection map as a styled Plotly JSON for the docs.

Finds the latest *.npy detection result in the run directory, loads the matching
ground truth, and builds a two-panel Plotly figure (GLRT map + GT overlay) that
matches the HDR docs design system.

Output: docs/docs/assets/data/{name}[.{label}].json
        docs/docs/assets/data/{name}[.{label}].source.txt

Run:
    uv run python action_add_to_docs.py --storage_path <run_dir> [--name sar_det_off_gauss] [--label scene1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import base64
import io

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("plotly not found — run: uv add plotly --dev")

try:
    from PIL import Image as PILImage
except ImportError:
    sys.exit("Pillow not found — run: uv add Pillow --dev")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--storage_path", required=True,
    help="Run directory injected by qanat.")
parser.add_argument("--name", default=None,
    help="Experiment name. Defaults to grandparent dir of storage_path.")
parser.add_argument("--label", default=None,
    help="Label for this run (e.g. 'scene1_w7'). Output: {name}.{label}.json.")
parser.add_argument("--ground-truth", default=None,
    help="Path to ground truth .npy (binary mask). Auto-detected from provenance if omitted.")
args = parser.parse_args()

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "shared"))
from plotly_style import BG, MUTED, INK2, FONT_SANS, FONT_MONO  # noqa: E402
storage   = Path(args.storage_path)
name      = args.name  or storage.parent.name
label     = args.label
run_dir   = storage

# ── Find NPY result ───────────────────────────────────────────────────────────
candidates = sorted(
    [p for p in run_dir.glob("*.npy") if "ground" not in p.name],
    key=lambda p: p.stat().st_mtime, reverse=True,
)
if not candidates:
    print(f"No .npy result found under {run_dir}")
    sys.exit(0)
npy_path = candidates[0]
print(f"Loaded {npy_path.name}")

det_map = np.load(npy_path).astype(np.float32)   # (H, W)
H, W    = det_map.shape

# ── Load provenance to find data_path / ground truth ─────────────────────────
prov_path = npy_path.with_suffix(".json")
gt_path   = None

if args.ground_truth:
    gt_path = repo_root / args.ground_truth
else:
    if prov_path.exists():
        import json
        prov = json.loads(prov_path.read_text())
        data_path = prov.get("args", {}).get("data_path", "")
        if data_path:
            # e.g. 2-detection/data/SAR/scene1.npy → ground_truth_scene_1.npy
            scene_file = Path(data_path).name          # scene1.npy
            scene_stem = Path(data_path).stem          # scene1
            gt_name    = f"ground_truth_{scene_stem.replace('scene', 'scene_')}.npy"
            gt_path    = repo_root / Path(data_path).parent / gt_name

if gt_path and gt_path.exists():
    gt_full = np.load(gt_path).astype(np.float32)   # (H_full, W_full)
    # Crop to match detection map (window border = (H_full - H) // 2)
    dh = (gt_full.shape[0] - H) // 2
    dw = (gt_full.shape[1] - W) // 2
    gt = gt_full[dh:dh+H, dw:dw+W] if dh > 0 or dw > 0 else gt_full[:H, :W]
    print(f"Ground truth: {gt_path.name}  crop ±{dh},±{dw}")
else:
    gt = None
    print("No ground truth found — showing detection map only")

# ── Pixel spacing (metres) ────────────────────────────────────────────────────
DX = 1.0   # horizontal — range resolution
DY = 0.6   # vertical   — azimuth resolution

# ── Design tokens (from shared/plotly_style.py) ───────────────────────────────

# Warm dark colorscale stops (RGB 0-255)
def _make_colorscale():
    stops = [
        (0.00, (13,  10,   8)),
        (0.30, (61,  37,  16)),
        (0.60, (184, 92,  42)),
        (0.85, (232, 160, 96)),
        (1.00, (245, 234, 220)),
    ]
    return stops

def _apply_colorscale(arr: np.ndarray) -> np.ndarray:
    """Map a 2D float array (0-1 normalised) to uint8 RGB via the warm colorscale."""
    stops = _make_colorscale()
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        mask = (arr >= t0) & (arr <= t1)
        alpha = np.where(mask, (arr - t0) / (t1 - t0), 0.0)
        for ch in range(3):
            out[..., ch] = np.where(
                mask,
                np.clip(c0[ch] + alpha * (c1[ch] - c0[ch]), 0, 255).astype(np.uint8),
                out[..., ch],
            )
    return out


def _to_png_b64(rgb: np.ndarray) -> str:
    img = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Render panels to PNG ──────────────────────────────────────────────────────
vmax = float(np.percentile(det_map, 99))
det_norm = np.clip(det_map / vmax, 0, 1)
det_rgb  = _apply_colorscale(det_norm)
det_src  = _to_png_b64(det_rgb)
print(f"Detection map PNG: {H}×{W}")

gt_src = None
if gt is not None:
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    gt_rgb[gt == 1] = [184, 92, 42]   # rust for changed pixels
    gt_src = _to_png_b64(gt_rgb)
    print(f"Ground truth PNG:  {H}×{W}")

# ── Figure ────────────────────────────────────────────────────────────────────
n_panels       = 2 if gt_src is not None else 1
subplot_titles = ["GLRT statistic", "Ground truth"] if n_panels == 2 else ["GLRT statistic"]

fig = make_subplots(
    rows=1, cols=n_panels,
    column_widths=[1] * n_panels,
    horizontal_spacing=0.04,
    subplot_titles=subplot_titles,
)

def _image_trace(src):
    return go.Image(
        source=src,
        x0=0, y0=0, dx=DX, dy=DY,
        hoverinfo="skip",
    )

fig.add_trace(_image_trace(det_src), row=1, col=1)
if gt_src:
    fig.add_trace(_image_trace(gt_src), row=1, col=2)

# ── Axis styling — equal physical scaling ────────────────────────────────────
# Physical extent: W*DX wide, H*DY tall
phys_w = W * DX
phys_h = H * DY

axis_common = dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[0, phys_w])
yaxis_common = dict(showgrid=False, zeroline=False, showticklabels=False,
                    range=[phys_h, 0],          # top-to-bottom
                    scaleanchor="x", scaleratio=1)

fig.update_xaxes(axis_common)
fig.update_yaxes(yaxis_common)
if n_panels == 2:
    fig.update_xaxes(dict(axis_common, range=[0, phys_w]), row=1, col=2)
    fig.update_yaxes(dict(yaxis_common, scaleanchor="x2"), row=1, col=2)

# Display height: cap at 700px, user can pan/zoom for full resolution
panel_h = 700

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    margin=dict(t=36, r=10, b=10, l=10),
    font=dict(family=FONT_SANS, size=13, color=INK2),
    height=panel_h,
)
for ann in fig.layout.annotations:
    ann.font = dict(family=FONT_SANS, size=12, color=MUTED)
    ann.y    = 1.02

# ── Export ────────────────────────────────────────────────────────────────────
stem = f"{name}.{label}" if label else name
out  = repo_root / "docs" / "docs" / "assets" / "data" / f"{stem}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(fig.to_json())
(out.parent / f"{stem}.source.txt").write_text(f"{npy_path.resolve()}\n")
print(f"Written {out.relative_to(repo_root)}")
