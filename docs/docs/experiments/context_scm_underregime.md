# context_scm_underregime

SCM estimator behavior in the under-regime (d > N) — mean, covariance, condition number errors

**Tags:** `context`  `scm`  `under-regime`  `monte-carlo`

## Run

```sh
uv run python 1-context/scm_underregime/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--N` | int | `30` | Number of observations. |
| `--n_trials` | int | `50` | Number of MC-trials. |
| `--storage_path` | str | `outputs/error_estimation_scm_underregime` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figures (.tex) (default: True). |

## Config

`1-context/experiments/context_scm_underregime.yaml`
