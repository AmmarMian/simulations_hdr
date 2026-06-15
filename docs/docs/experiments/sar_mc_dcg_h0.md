# sar_mc_dcg_h0

MC convergence test — OnlineDCGGLRT telescopes to DCGGLRT as T grows (H0)

**Tags:** `detection`  `dcg`  `H0`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_dcg_h0.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iter-max` | int | `20` | Max iterations for Tyler fixed-point and online estimator (default 20). |
| `--tol-online` | float | `0.0001` | Convergence tolerance for online H0/H1 estimators (default 1e-4). |
| `--tol-offline` | float | `0.0001` | Convergence tolerance for offline Tyler estimators (default 1e-4). |
| `--tau-shape` | float | `1.0` | Shape parameter of Gamma(shape, scale) texture distribution (default 1.0). |
| `--tau-scale` | float | `1.0` | Scale parameter of Gamma(shape, scale) texture distribution (default 1.0). |
| `--n-features` | int | `8` | Feature dimension p; n_samples is fixed to 2*p+1 (default 8). |
| `--T-max` | int | `1000` | Maximum number of time steps (default 1000). |
| `--T-min` | int | `5` | Minimum number of time steps (default 5). |
| `--n-T` | int | `30` | Number of T values in log scale (default 30). |
| `--sigma-seed` | int | `0` | Seed for Sigma_true generation, independent from --seed (default 0). |
| `--n-trials` | int | `10000` | Number of Monte-Carlo trials (default 10000). |
| `--seed` | int | `42` | RNG seed for data generation (default 42). |
| `--backend` | str | `numpy` | Compute backend. numpy → multiprocessing.Pool (one worker per trial); all others → trials stacked in leading batch dimension (default numpy). |
| `--n-workers` | int | — | Pool workers for numpy backend (default: os.cpu_count()). |
| `--export` | — | `True` | Save .npz results + provenance sidecar + plot script (default: True). |
| `--storage-path` / `--storage_path` / `--export-path` | str | `./exports` | Directory for exported results; --storage-path is the qanat alias (default: ./exports). |
| `--show-interactive` | — | — | Display figures interactively at the end of the simulation. |

## Config

`2-detection/experiments/sar/sar_mc_dcg_h0.yaml`
