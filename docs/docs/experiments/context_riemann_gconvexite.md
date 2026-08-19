# context_riemann_gconvexite

Tyler's cost read along a Euclidean segment and along an affine-invariant geodesic

**Tags:** `context`  `riemann`  `robust`  `illustration`

## Run

```sh
uv run python 1-context/riemann_gconvexite/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_features` | int | `3` | Dimension of the observations. |
| `--n_samples` | int | `10` | Number of observations. A short sample makes the cost surface sharper, hence the effect easier to see; the phenomenon itself does not depend on it. |
| `--dof` | float | `3.0` | Degrees of freedom of the Student data. |
| `--condition` | float | `10000.0` | Condition number of the two endpoints. The larger it is, the more pronounced the interior maximum of the Euclidean reading. |
| `--n_points` | int | `201` | Number of points at which the cost is evaluated on each path. |
| `--storage_path` | str | `outputs/riemann_gconvexite` | Output directory for LaTeX exports (injected by qanat, or set manually). |
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
  <code>--n_features</code> <span class='mn-default'>3</span><br>
  <code>--n_samples</code> <span class='mn-default'>10</span><br>
  <code>--dof</code> <span class='mn-default'>3.0</span><br>
  <code>--condition</code> <span class='mn-default'>10000.0</span><br>
  <code>--n_points</code> <span class='mn-default'>201</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--axis_width</code> <span class='mn-default'>0.45\textwidth</span><br>
  <code>--axis_height</code> <span class='mn-default'>4.6cm</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_gconvexite.json" data-title="context_riemann_gconvexite"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">d = 3, N = 10, Student nu = 3, condition 10000, geodesic distance between the endpoints 13.025
  segment euclidien  min curvature    -43.54   local minima at t = 0.01, 0.96
  géodésique         min curvature     +8.15   local minima at t = 0.45
Saved cost profiles in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_riemann_gconvexite/run_26/gconvexite.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_riemann_gconvexite.yaml`
