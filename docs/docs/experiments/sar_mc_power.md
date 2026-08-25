# sar_mc_power

Puissance vs T des quatre detecteurs de changements, hors ligne et en ligne

**Tags:** `detection`  `kronecker`  `puissance`  `H1`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_power_detectors.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--a` | int | `3` | Size of the first Kronecker factor. |
| `--b` | int | `4` | Size of the second Kronecker factor. |
| `--n-samples` | int | — | Samples per date (default: p+1 = a*b+1 = 13). |
| `--texture` | str | `k` | 'k' for K-distributed data with shape --nu (default), 'gaussian' for tau = 1. |
| `--nu` | float | `1.0` | Shape of the K-distribution texture when --texture k (default 1.0). |
| `--rho-a0` | str | `0.3+0.7j` | Toeplitz coefficient of A under H0. |
| `--rho-b0` | str | `0.3+0.6j` | Toeplitz coefficient of B under H0. |
| `--rho-a1` | str | `0.3+0.5j` | Toeplitz coefficient of A after the change. |
| `--rho-b1` | str | `0.4+0.5j` | Toeplitz coefficient of B after the change. |
| `--step-rule` | str | `fixed` | Step of the recursive estimators (default 'fixed', the schedule of eq. 19). |
| `--init-mode` | str | `mm` | Initialisation of the recursive Kronecker estimator (default 'mm'). |
| `--alpha-0` | float | `1.0` | Initial step (default 1.0). |
| `--iter-max` | int | `30` | Max fixed-point / MM iterations. |
| `--tol` | float | `0.0001` | Convergence tolerance. |
| `--with-gaussian` | — | — | Add the Gaussian covariance equality GLRT as a fifth baseline. |
| `--detectors` | str | — | Comma-separated subset of detectors to run (e.g. 'SG-O'). Default: all four. Use it to iterate on one curve without paying for the others; the thresholds are per detector anyway, so a subset gives the same numbers. |
| `--debug` | — | — | Tiny configuration (60 trials, T up to 10, PFA 0.1) to validate the pipeline in seconds. Results are NOT publication grade. |
| `--sigma2-seed` | int | `1` | Seed for Sigma_2 (H1 distribution, default 1 — different from --sigma-seed). |
| `--change-fraction` | float | `0.5` | Change point as a fraction of T, so n_change_dates = max(2, int(T * change_fraction)). Default 0.5 — change at midpoint, ensuring equal pre/post evidence at every T. |
| `--pfa` | float | `0.001` | Target false alarm probability for power estimation (default 1e-3). Reliable threshold estimation requires at least 10/PFA H0 trials. |
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

`2-detection/experiments/sar/sar_mc_power.yaml`
