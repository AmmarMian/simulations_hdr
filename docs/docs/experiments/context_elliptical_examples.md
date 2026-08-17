# context_elliptical_examples

Isodensity contours and draws for elliptical distributions sharing one scatter matrix

**Tags:** `context`  `elliptical`  `distributions`  `illustration`

## Run

```sh
uv run python 1-context/elliptical_examples/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--distributions` | str | `['gaussian', 'student', 'k', 'gengauss']` | Distributions to show, one panel each. The Gaussian acts as the reference against which the tails are read. |
| `--n_samples` | int | `50` | Number of samples drawn per panel. Drawn as hollow markers so the isodensity contours stay readable underneath. |
| `--rho` | float | `0.8` | Correlation coefficient of the shared scatter matrix. |
| `--dof_student` | float | `3.0` | Degrees of freedom of the t distribution (>2 for a finite covariance). |
| `--dof_k` | float | `2.0` | Texture shape of the K distribution. |
| `--shape_gengauss` | float | `0.5` | Exponent s of the generalized Gaussian (s<1 gives heavier tails). |
| `--storage_path` | str | `outputs/elliptical_examples` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figure (.tex) (default: True). |
| `--backend` | str | `numpy` | Compute backend for the draws (numpy, torch-cpu, torch-mps, ...). |
| `--seed` | int | `42` | random seed generation base seed |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-08-17</span><br>
  <code>--distributions</code> <span class='mn-default'>['gaussian', 'student', 'k', 'gengauss']</span><br>
  <code>--n_samples</code> <span class='mn-default'>50</span><br>
  <code>--rho</code> <span class='mn-default'>0.8</span><br>
  <code>--dof_student</code> <span class='mn-default'>3.0</span><br>
  <code>--dof_k</code> <span class='mn-default'>2.0</span><br>
  <code>--shape_gengauss</code> <span class='mn-default'>0.5</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_elliptical_examples.json" data-title="context_elliptical_examples"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Saved elliptical examples in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_elliptical_examples/run_13/elliptical_tails.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_elliptical_examples.yaml`
