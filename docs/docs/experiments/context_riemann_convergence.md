# context_riemann_convergence

Fixed point, Riemannian descent and projected Euclidean descent on Tyler's cost

**Tags:** `context`  `riemann`  `optimisation`  `robust`

## Run

```sh
uv run python 1-context/riemann_convergence/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_features` | int | `10` | Dimension of the observations. |
| `--n_samples` | int | `100` | Number of observations used by the three algorithms. |
| `--dof` | float | `3.0` | Degrees of freedom of the Student data. |
| `--condition` | float | `100.0` | Condition number of the true scatter matrix. The larger it is, the further the identity — the common starting point — sits from the solution. |
| `--iter_max` | int | `150` | Maximum number of iterations granted to each algorithm. |
| `--tol` | float | `1e-12` | Stopping tolerance on the Riemannian gradient norm. Deliberately unreachable, so that every algorithm spends its whole budget and the curves can be compared over their full length. |
| `--floor` | float | `1e-14` | Smallest optimality gap shown; below it the cost is dominated by rounding rather than by the algorithm. |
| `--storage_path` | str | `outputs/riemann_convergence` | Output directory for LaTeX exports (injected by qanat, or set manually). |
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
  <code>--n_features</code> <span class='mn-default'>10</span><br>
  <code>--n_samples</code> <span class='mn-default'>100</span><br>
  <code>--dof</code> <span class='mn-default'>3.0</span><br>
  <code>--condition</code> <span class='mn-default'>100.0</span><br>
  <code>--iter_max</code> <span class='mn-default'>150</span><br>
  <code>--tol</code> <span class='mn-default'>1e-12</span><br>
  <code>--floor</code> <span class='mn-default'>1e-14</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--axis_width</code> <span class='mn-default'>0.45\textwidth</span><br>
  <code>--axis_height</code> <span class='mn-default'>4.6cm</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_convergence.json" data-title="context_riemann_convergence"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">d = 10, N = 100, Student nu = 3, condition 100
  point fixe                    29 iterations   final gap 5.33e-15   gradient 4.29e-13   0.003 s   distance to the fixed point 5.42e-15
  gradient riemannien           28 iterations   final gap 0.00e+00   gradient 3.05e-09   0.006 s   distance to the fixed point 4.88e-09
  gradient euclidien projeté   150 iterations   final gap 5.79e-02   gradient 3.06e-01   0.067 s   distance to the fixed point 3.87e-01
Saved convergence curves in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_riemann_convergence/run_22/convergence.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_riemann_convergence.yaml`
