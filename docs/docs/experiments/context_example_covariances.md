# context_example_covariances

PGFPlots matrix visualisations of covariance regimes — identity, Toeplitz, random

**Tags:** `context`  `covariance`  `illustration`

## Run

```sh
uv run python 1-context/examples_covariances/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--storage_path` | str | `outputs/example_covariances` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figure (.tex) (default: True). |
| `--seed` | int | `42` | random seed generation base seed |

## Config

`1-context/experiments/context_example_covariances.yaml`
