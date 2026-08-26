#!/usr/bin/env python3
"""Render the landing-page hero field as a static SVG.

The docs landing page shows a still portrait of the same vortex field the live
WebGL hero animates: tracers advected along the curl of a 3-D simplex-noise
field on the unit sphere. A frozen frame of the animation is a fuzz of very
short streaks, so the poster instead draws long streamlines of the field held
at one instant — a long exposure of the same flow. Integrating here rather than
screenshotting keeps it crisp at any size and a few hundred kilobytes.

The output is committed; regenerate with

    just docs-hero

Everything below is a port of the ``Vortex`` mode of ``hero-flow.js``. The noise
permutation uses the same seed and the same Lehmer shuffle, so the flow lines
here are the flow lines the animation draws.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ── output geometry ──────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1600, 900
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img" / "hero-field.svg"

# ── field parameters (mirrors the locked-in values in hero-flow.js) ──────────
N_TRAILS = 480
TRAIL_LEN = 60          # points per streamline
STEP = 0.012            # radians of arc between points — sets the exposure
NOISE_FREQ = 1.75
FIELD_TIME = 21.0       # which instant of the slowly-drifting field to freeze

# ── camera (three.js PerspectiveCamera(42, aspect, …) at (0, 0.45, 3.35)) ────
EYE = np.array([0.0, 0.45, 3.35])
FOV_DEG = 42.0

# ── Plasma palette ───────────────────────────────────────────────────────────
BG = "#0b0306"
TRAIL_A = np.array([1.00, 0.18, 0.52])
TRAIL_B = np.array([1.00, 0.78, 0.34])
HEAD = np.array([1.00, 0.74, 0.50])
RIM = np.array([1.00, 0.46, 0.72])
FILL = "#16060d"

# Opacity of the three constant-alpha chunks each tracer is split into, head
# first. A single polyline cannot taper, so the taper is quantised instead.
CHUNK_ALPHA = (0.72, 0.40, 0.18)
BACK_FACTOR = 0.20      # tracers on the far side of the glass shell


# ── 3-D simplex noise ────────────────────────────────────────────────────────
_GRAD3 = np.array(
    [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]],
    dtype=np.float64,
)


def _build_perm() -> tuple[np.ndarray, np.ndarray]:
    """The Lehmer-shuffled permutation table, seed 1337, as in the JS port."""
    p = list(range(256))
    seed = 1337

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 16807) % 2147483647
        return seed / 2147483647

    for i in range(255, 0, -1):
        j = int(rnd() * (i + 1))
        p[i], p[j] = p[j], p[i]
    perm = np.array([p[i & 255] for i in range(512)], dtype=np.int64)
    return perm, perm % 12


_PERM, _PMOD = _build_perm()

_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0


def noise3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Vectorised 3-D simplex noise, value in roughly [-1, 1]."""
    s = (x + y + z) * _F3
    i, j, k = np.floor(x + s), np.floor(y + s), np.floor(z + s)
    t = (i + j + k) * _G3
    x0, y0, z0 = x - (i - t), y - (j - t), z - (k - t)

    # Which of the six simplices within the cell we fell into: the ranking of
    # x0, y0, z0 picks the two corners traversed between (0,0,0) and (1,1,1).
    conds = [
        (x0 >= y0) & (y0 >= z0),
        (x0 >= y0) & (y0 < z0) & (x0 >= z0),
        (x0 >= y0) & (y0 < z0) & (x0 < z0),
        (x0 < y0) & (y0 < z0),
        (x0 < y0) & (y0 >= z0) & (x0 < z0),
        (x0 < y0) & (y0 >= z0) & (x0 >= z0),
    ]
    corner1 = [(1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 1, 0)]
    corner2 = [(1, 1, 0), (1, 0, 1), (1, 0, 1), (0, 1, 1), (0, 1, 1), (1, 1, 0)]
    pick = lambda table, axis: np.select(conds, [c[axis] for c in table], 0.0)
    i1, j1, k1 = (pick(corner1, a) for a in range(3))
    i2, j2, k2 = (pick(corner2, a) for a in range(3))

    x1, y1, z1 = x0 - i1 + _G3, y0 - j1 + _G3, z0 - k1 + _G3
    x2, y2, z2 = x0 - i2 + 2 * _G3, y0 - j2 + 2 * _G3, z0 - k2 + 2 * _G3
    x3, y3, z3 = x0 - 1 + 3 * _G3, y0 - 1 + 3 * _G3, z0 - 1 + 3 * _G3

    ii, jj, kk = i.astype(np.int64) & 255, j.astype(np.int64) & 255, k.astype(np.int64) & 255

    def contrib(xa, ya, za, io, jo, ko):
        tt = 0.6 - xa * xa - ya * ya - za * za
        gi = _PMOD[ii + io + _PERM[jj + jo + _PERM[kk + ko]]]
        g = _GRAD3[gi]
        dot = g[:, 0] * xa + g[:, 1] * ya + g[:, 2] * za
        return np.where(tt < 0, 0.0, np.maximum(tt, 0.0) ** 4 * dot)

    n0 = contrib(x0, y0, z0, 0, 0, 0)
    n1 = contrib(x1, y1, z1, i1.astype(np.int64), j1.astype(np.int64), k1.astype(np.int64))
    n2 = contrib(x2, y2, z2, i2.astype(np.int64), j2.astype(np.int64), k2.astype(np.int64))
    n3 = contrib(x3, y3, z3, 1, 1, 1)
    return 32.0 * (n0 + n1 + n2 + n3)


# ── the vortex field ─────────────────────────────────────────────────────────
def vortex_velocity(p: np.ndarray, now: float) -> np.ndarray:
    """Curl of a noise field, projected onto the sphere: p × ∇n."""
    tt = now * 0.06
    e = 0.12
    f = NOISE_FREQ
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    n0 = noise3(x * f + tt, y * f, z * f - tt)
    nx = noise3((x + e) * f + tt, y * f, z * f - tt) - n0
    ny = noise3(x * f + tt, (y + e) * f, z * f - tt) - n0
    nz = noise3(x * f + tt, y * f, (z + e) * f - tt) - n0
    n = np.stack([nx, ny, nz], axis=1)
    return np.cross(p, n)


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def integrate() -> np.ndarray:
    """Trace streamlines of the frozen field; shape (N, TRAIL_LEN, 3)."""
    rng = np.random.default_rng(20260826)
    u = rng.uniform(-1.0, 1.0, N_TRAILS)
    th = rng.uniform(0.0, 2 * math.pi, N_TRAILS)
    r = np.sqrt(1.0 - u * u)
    p = np.stack([r * np.cos(th), u, r * np.sin(th)], axis=1)

    def advance(pos: np.ndarray) -> np.ndarray:
        v = vortex_velocity(pos, FIELD_TIME)
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        # A degenerate velocity means the tracer sits on a critical point of the
        # field; nudge it along the local azimuth so the streamline still moves.
        fallback = np.stack([-pos[:, 2], np.zeros(len(pos)), pos[:, 0]], axis=1)
        v = np.where(speed < 1e-6, fallback, v)
        return normalise(pos + normalise(v) * STEP)

    trails = np.empty((N_TRAILS, TRAIL_LEN, 3))
    for k in range(TRAIL_LEN):
        trails[:, k] = p
        p = advance(p)
    # Index 0 is the head, so the taper runs head -> tail.
    return trails[:, ::-1]


# ── projection ───────────────────────────────────────────────────────────────
def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-project world points; also report which face the camera sees."""
    forward = normalise((-EYE)[None, :])[0]
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    rel = points - EYE
    depth = (rel * forward).sum(axis=1)
    xv, yv = (rel * right).sum(axis=1), (rel * up).sum(axis=1)
    tan_half = math.tan(math.radians(FOV_DEG) / 2)
    aspect = WIDTH / HEIGHT
    ndc_x = xv / (depth * tan_half * aspect)
    ndc_y = yv / (depth * tan_half)
    screen = np.stack(
        [(ndc_x * 0.5 + 0.5) * WIDTH, (0.5 - ndc_y * 0.5) * HEIGHT], axis=-1
    )
    # |p| = 1, so the near hemisphere is exactly where p·eye > 1.
    front = (points * EYE).sum(axis=1) > 1.0
    return screen, front


# ── SVG assembly ─────────────────────────────────────────────────────────────
def to_hex(rgb: np.ndarray) -> str:
    v = np.clip(rgb, 0.0, 1.0) * 255
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c)) for c in v))


def polyline(pts: np.ndarray) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def build_svg() -> str:
    trails = integrate()
    flat = trails.reshape(-1, 3)
    screen, front = project(flat)
    screen = screen.reshape(N_TRAILS, TRAIL_LEN, 2)
    front = front.reshape(N_TRAILS, TRAIL_LEN)

    rng = np.random.default_rng(7)
    mix = rng.uniform(0.0, 1.0, N_TRAILS)

    bounds = TRAIL_LEN // len(CHUNK_ALPHA)
    back_parts: list[str] = []
    front_parts: list[str] = []
    heads: list[str] = []

    for i in range(N_TRAILS):
        colour = to_hex(TRAIL_A + (TRAIL_B - TRAIL_A) * mix[i])
        for c, alpha in enumerate(CHUNK_ALPHA):
            lo = c * bounds
            hi = TRAIL_LEN if c == len(CHUNK_ALPHA) - 1 else (c + 1) * bounds + 1
            seg, seg_front = screen[i, lo:hi], front[i, lo:hi]
            # A chunk that straddles the silhouette is split so the hidden part
            # dims rather than the whole chunk flipping brightness.
            start = 0
            for end in range(1, len(seg) + 1):
                if end == len(seg) or seg_front[end] != seg_front[start]:
                    run = seg[start : end + 1] if end < len(seg) else seg[start:]
                    if len(run) >= 2:
                        visible = bool(seg_front[start])
                        a = alpha if visible else alpha * BACK_FACTOR
                        line = (
                            f'<polyline points="{polyline(run)}" '
                            f'stroke="{colour}" stroke-opacity="{a:.3f}"/>'
                        )
                        (front_parts if visible else back_parts).append(line)
                    start = end
        if front[i, 0]:
            hx, hy = screen[i, 0]
            heads.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="1.5"/>')

    head_hex = to_hex(HEAD)
    rim_hex = to_hex(RIM)
    # The camera looks at the origin, so the sphere's silhouette is a circle
    # centred on the frame; its radius follows from the tangent cone.
    cx, cy = WIDTH / 2, HEIGHT / 2
    dist = float(np.linalg.norm(EYE))
    tan_half = math.tan(math.radians(FOV_DEG) / 2)
    radius = (HEIGHT / 2) / tan_half / math.sqrt(dist * dist - 1.0)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" \
width="{WIDTH}" height="{HEIGHT}" role="img" \
aria-label="Tracer particles flowing along a vortex field on the unit sphere">
<title>Riemannian flow field</title>
<defs>
<radialGradient id="shell" cx="50%" cy="46%" r="52%">
<stop offset="0%" stop-color="{FILL}" stop-opacity="0.30"/>
<stop offset="62%" stop-color="{FILL}" stop-opacity="0.26"/>
<stop offset="88%" stop-color="{rim_hex}" stop-opacity="0.18"/>
<stop offset="100%" stop-color="{rim_hex}" stop-opacity="0.55"/>
</radialGradient>
<radialGradient id="bloom" cx="50%" cy="42%" r="62%">
<stop offset="0%" stop-color="{rim_hex}" stop-opacity="0.16"/>
<stop offset="100%" stop-color="{rim_hex}" stop-opacity="0"/>
</radialGradient>
</defs>
<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>
<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#bloom)"/>
<g fill="none" stroke-width="0.9" stroke-linecap="round" stroke-linejoin="round" \
style="mix-blend-mode:screen">
{chr(10).join(back_parts)}
</g>
<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="url(#shell)"/>
<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="none" \
stroke="{rim_hex}" stroke-opacity="0.45" stroke-width="1.1"/>
<g fill="none" stroke-width="0.9" stroke-linecap="round" stroke-linejoin="round" \
style="mix-blend-mode:screen">
{chr(10).join(front_parts)}
</g>
<g fill="{head_hex}" fill-opacity="0.65" style="mix-blend-mode:screen">
{chr(10).join(heads)}
</g>
</svg>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(Path.cwd())} ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
