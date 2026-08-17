# context_robust_mestimation

Concentration ellipses of the SCM, the model MLE and Tyler's estimator on a single draw

**Tags:** `context`  `robust`  `m-estimation`  `illustration`

## Run

```sh
uv run python 1-context/robust_mestimation/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--distributions` | str | `['gaussian', 'student', 'k', 'gengauss']` | Models to show, one panel each. |
| `--n_samples` | int | `50` | Number of observations of the single estimation shown. Kept small on purpose: with a large support every estimator is accurate and the ellipses become indistinguishable. |
| `--rho` | float | `0.8` | Correlation of the shared true shape matrix. |
| `--dof_student` | float | `2.1` | Degrees of freedom of the t model. Just above 2, where the covariance still exists but the tails are very heavy. |
| `--dof_k` | float | `0.1` | Texture shape of the K model; the smaller, the heavier. |
| `--shape_gengauss` | float | `0.15` | Exponent s of the generalized Gaussian; s<1 gives heavier tails. |
| `--iter_max` | int | `100` | Fixed-point iterations. |
| `--tol` | float | `1e-08` | Fixed-point tolerance. |
| `--storage_path` | str | `outputs/robust_mestimation` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figure (.tex) (default: True). |
| `--backend` | str | `numpy` | Compute backend (numpy, torch-cpu, torch-mps, ...). |
| `--seed` | int | `42` | random seed generation base seed |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-08-17</span><br>
  <code>--distributions</code> <span class='mn-default'>['gaussian', 'student', 'k', 'gengauss']</span><br>
  <code>--n_samples</code> <span class='mn-default'>50</span><br>
  <code>--rho</code> <span class='mn-default'>0.8</span><br>
  <code>--dof_student</code> <span class='mn-default'>2.1</span><br>
  <code>--dof_k</code> <span class='mn-default'>0.1</span><br>
  <code>--shape_gengauss</code> <span class='mn-default'>0.15</span><br>
  <code>--iter_max</code> <span class='mn-default'>100</span><br>
  <code>--tol</code> <span class='mn-default'>1e-08</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_robust_mestimation.json" data-title="context_robust_mestimation"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Shape estimation error (Frobenius), N = 50:
  gaussian  scm=0.025  mle=0.025  tyler=0.045
  student   scm=0.504  mle=0.089  tyler=0.048
  k         scm=0.694  mle=0.144  tyler=0.158
  gengauss  scm=0.322  mle=0.151  tyler=0.103
Saved ellipses in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_robust_mestimation/run_14/scmvstyler.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_robust_mestimation.yaml`
