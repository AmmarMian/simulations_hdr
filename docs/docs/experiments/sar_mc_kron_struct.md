# sar_mc_kron_struct

Ce que la structure Kronecker achete : erreur vs taille de fenetre N

**Tags:** `detection`  `kronecker`  `estimation`  `structure`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_kron_structure_vs_n.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--a` | int | `3` | Size of the first Kronecker factor. |
| `--b` | int | `4` | Size of the second Kronecker factor. |
| `--T` | int | `25` | Number of dates, held fixed while N varies (default 25). |
| `--N-list` | int | `[4, 6, 9, 13, 20, 30, 45, 70]` | Patch sizes to sweep (default 4 6 9 13 20 30 45 70; p = a*b = 12). |
| `--nu` | float | `1.0` | Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0). |
| `--rho-a` | str | `0.3+0.7j` | Toeplitz coefficient of A. |
| `--rho-b` | str | `0.3+0.6j` | Toeplitz coefficient of B. |
| `--offline` | str | `gd` | Structured estimator: 'gd' Riemannian gradient descent, term for term comparable with the unstructured one (default), or 'mm'. |
| `--mm-iter-max` | int | `50` | Max MM iterations. |
| `--mm-tol` | float | `1e-08` | MM tolerance. |
| `--gd-iter-max` | int | `200` | Max GD iterations. |
| `--gd-tol` | float | `1e-08` | GD tolerance. |
| `--debug` | — | — | Tiny configuration (6 trials, 3 values of N, T=8) to validate the pipeline in seconds. Results are NOT publication grade. |

## Config

`2-detection/experiments/sar/sar_mc_kron_struct.yaml`
