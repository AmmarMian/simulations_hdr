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

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-06-18</span><br>
  <code>--d</code> <span class='mn-default'>7</span><br>
  <code>--n_trials</code> <span class='mn-default'>10000</span><br>
  <code>--show-interactive</code><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
</span>
<div class="plotly-wrap" data-src="../../assets/data/context_scm_grandnombres.json" data-title="context_scm_grandnombres"></div>

## Config

`1-context/experiments/context_scm_grandnombres.yaml`
