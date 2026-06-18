# context_lwf_underregime

Condition number of SCM vs LWF-regularised covariance in the under-regime (d > N)

**Tags:** `context`  `lwf`  `regularisation`  `under-regime`  `monte-carlo`

## Run

```sh
uv run python 1-context/lwf_underregime/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--N` | int | `30` | Number of observations. |
| `--n_trials` | int | `50` | Number of MC-trials. |
| `--storage_path` | str | `outputs/error_estimation_lwf_underregime` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--alpha` | float | `0.1` | Coefficient of regularization. |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figures (.tex) (default: True). |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-06-18</span><br>
  <code>--N</code> <span class='mn-default'>30</span><br>
  <code>--n_trials</code> <span class='mn-default'>50</span><br>
  <code>--alpha</code> <span class='mn-default'>0.1</span><br>
  <code>--show-interactive</code><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
</span>
<div class="plotly-wrap" data-src="../../assets/data/context_lwf_underregime.json" data-title="context_lwf_underregime"></div>

## Config

`1-context/experiments/context_lwf_underregime.yaml`
