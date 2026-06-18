# context_scm_grandnombres

MC convergence of SCM mean/covariance estimators as N → ∞ (grand-nombre regime)

**Tags:** `context`  `scm`  `monte-carlo`  `convergence`

## Run

```sh
uv run python 1-context/scm_grandnombres.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--d` | int | `7` | Dimension of vector. |
| `--n_trials` | int | `10000` | Number of MC-trials. |
| `--storage_path` | str | `outputs/error_estimation_scm` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figures (.tex) (default: True). |

## Config

`1-context/experiments/context_scm_grandnombres.yaml`
