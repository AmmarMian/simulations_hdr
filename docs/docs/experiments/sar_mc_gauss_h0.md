# sar_mc_gauss_h0

MC convergence test — OnlineGaussianGLRT telescopes to GaussianGLRT as T grows (H0)

**Tags:** `detection`  `gaussian`  `H0`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_gaussian_h0.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
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

`2-detection/experiments/sar/sar_mc_gauss_h0.yaml`
