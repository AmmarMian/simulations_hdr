# context_lwf_underregime

Condition number of SCM vs LWF-regularised covariance in the under-regime (d > N)

**Tags:** `context`  `lwf`  `regularisation`  `under-regime`  `monte-carlo`

## Run

```sh
uv run python 1-context/lwf_underregime.py
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

## Config

`1-context/experiments/context_lwf_underregime.yaml`
