# context_riemann_moyennes

Arithmetic, log-Euclidean and Fréchet means of a cloud of covariance matrices

**Tags:** `context`  `riemann`  `geometry`  `illustration`

## Run

```sh
uv run python 1-context/riemann_moyennes/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_matrices` | int | `15` | Number of matrices averaged. Kept small enough for the cloud to remain readable as a set of ellipses. |
| `--dispersion` | float | `0.9` | Standard deviation of the geodesic distance between a matrix of the cloud and its centre, in the affine-invariant metric. |
| `--condition` | float | `4.0` | Ratio of the eigenvalues of the centre of the cloud. |
| `--radius` | float | `1.0` | Radius of the drawn ellipses, in units of the Mahalanobis distance. |
| `--iter_max` | int | `100` | Maximum number of Riemannian gradient steps for the Fréchet mean. |
| `--tol` | float | `1e-10` | Stopping tolerance on the gradient norm of the Fréchet mean. |
| `--storage_path` | str | `outputs/riemann_moyennes` | Output directory for LaTeX exports (injected by qanat, or set manually). |
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
  <code>--n_matrices</code> <span class='mn-default'>15</span><br>
  <code>--dispersion</code> <span class='mn-default'>0.9</span><br>
  <code>--condition</code> <span class='mn-default'>4.0</span><br>
  <code>--radius</code> <span class='mn-default'>1.0</span><br>
  <code>--iter_max</code> <span class='mn-default'>100</span><br>
  <code>--tol</code> <span class='mn-default'>1e-10</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--axis_width</code> <span class='mn-default'>0.45\textwidth</span><br>
  <code>--axis_height</code> <span class='mn-default'>4.6cm</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_moyennes.json" data-title="context_riemann_moyennes"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">N = 15 matrices, dispersion 0.9, Fréchet mean in 5 iterations (gradient norm 7.25e-11)
  geometric mean of the determinants: 1.031
  arithmétique     det =   1.323   distance to the centre = 0.288
  de Fréchet       det =   1.031   distance to the centre = 0.081
  log-euclidienne  det =   1.031   distance to the centre = 0.077
Saved means in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_riemann_moyennes/run_25/moyennes.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_riemann_moyennes.yaml`
