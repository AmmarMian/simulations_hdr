# context_riemann_interpolation

Euclidean, affine-invariant and log-Euclidean paths between two covariance matrices

**Tags:** `context`  `riemann`  `geometry`  `illustration`

## Run

```sh
uv run python 1-context/riemann_interpolation/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--condition` | float | `16.0` | Ratio of the two eigenvalues of each endpoint. The larger it is, the more elongated the ellipses and the more visible the swelling of the Euclidean path. |
| `--angle` | float | `0.35` | Angle between the principal directions of the two endpoints, in units of pi. A quarter turn, 0.5, is the worst case for the Euclidean path, but it makes the two endpoints commute, and the two Riemannian paths then coincide exactly; a value away from it keeps the swelling and separates them. |
| `--n_steps` | int | `7` | Number of ellipses drawn along each path, endpoints included. |
| `--radius` | float | `1.0` | Radius of the drawn ellipses, in units of the Mahalanobis distance. |
| `--storage_path` | str | `outputs/riemann_interpolation` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figure (.tex) (default: True). |
| `--axis_width` | str | `0.45\textwidth` | Width of a single panel in the exported PGFPlots figure. Set here rather than patched into the .tex afterwards, so that a re-sync into the dissertation does not undo it. |
| `--axis_height` | str | `4.6cm` | Height of a single panel in the exported PGFPlots figure. |
| `--backend` | str | `numpy` | Compute backend (numpy, torch-cpu, torch-mps, ...). |
| `--seed` | int | `42` | random seed generation base seed |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-08-19</span><br>
  <code>--condition</code> <span class='mn-default'>16.0</span><br>
  <code>--angle</code> <span class='mn-default'>0.35</span><br>
  <code>--n_steps</code> <span class='mn-default'>7</span><br>
  <code>--radius</code> <span class='mn-default'>1.0</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--axis_width</code> <span class='mn-default'>0.45\textwidth</span><br>
  <code>--axis_height</code> <span class='mn-default'>4.6cm</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_interpolation.json" data-title="context_riemann_interpolation"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Endpoints of determinant 1.000 and 1.000, condition number 16
  euclidienne        det at t=1/2: 3.791   max: 3.791
  affine invariante  det at t=1/2: 1.000   max: 1.000
  log-euclidienne    det at t=1/2: 1.000   max: 1.000
Saved interpolation paths in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_riemann_interpolation/run_29/interpolation.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_riemann_interpolation.yaml`
