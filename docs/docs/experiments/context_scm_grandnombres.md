# context_scm_grandnombres

MC convergence of SCM mean/covariance estimators as N → ∞ (grand-nombre regime)

**Tags:** `context`  `scm`  `monte-carlo`  `convergence`

## Run

```sh
uv run python 1-context/scm_grandnombres/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--d` | int | `7` | Dimension of vector. |
| `--n_trials` | int | `10000` | Number of MC-trials. |
| `--storage_path` | str | `outputs/error_estimation_scm` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figures (.tex) (default: True). |
| `--seed` | int | `42` | random seed generation base seed |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-07-02</span><br>
  <code>--d</code> <span class='mn-default'>7</span><br>
  <code>--n_trials</code> <span class='mn-default'>10000</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_scm_grandnombres.json" data-title="context_scm_grandnombres"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Launching simulation
Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Done.
Saved mean error in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_scm_grandnombres/run_4/mean.tex
Saved cov error in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_scm_grandnombres/run_4/cov.tex
</div>
</details>
<details class="exp-log">
<summary>stderr</summary>
<div class="exp-log-text">/Users/ammarmian/Research/HDR/simulations_hdr/.venv/lib/python3.12/site-packages/matplot2tikz/_cleanfigure.py:149: UserWarning: Cleaning Line Collections (scatter plot) is not supported yet.
  _recursive_cleanfigure(child, target_resolution, scale_precision)
/Users/ammarmian/Research/HDR/simulations_hdr/.venv/lib/python3.12/site-packages/matplot2tikz/_cleanfigure.py:149: UserWarning: Cleaning Line Collections (scatter plot) is not supported yet.
  _recursive_cleanfigure(child, target_resolution, scale_precision)
</div>
</details>
</div>

## Config

`1-context/experiments/context_scm_grandnombres.yaml`
