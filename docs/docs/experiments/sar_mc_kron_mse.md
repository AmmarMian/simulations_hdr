# sar_mc_kron_mse

MSE des estimateurs Kronecker (hors ligne vs recursif) face aux ICRB

**Tags:** `detection`  `kronecker`  `estimation`  `icrb`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_kron_mse_icrb.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--a` | int | `3` | Size of first Kronecker factor (default 3, as in the released config). |
| `--b` | int | `4` | Size of second Kronecker factor (default 4). |
| `--n-samples` | int | — | Samples per date (default: p+1 = a*b+1 = 13). |
| `--nu` | float | `1.0` | Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0). Large nu approaches the Gaussian case. |
| `--rho-a` | str | `0.3+0.7j` | Toeplitz coefficient of A, as a complex literal (default 0.3+0.7j). |
| `--rho-b` | str | `0.3+0.6j` | Toeplitz coefficient of B, as a complex literal (default 0.3+0.6j). |
| `--offline` | str | `mm` | Offline reference: 'mm' majorisation-minimisation (fast, default) or 'gd' Riemannian gradient descent (the reference of the paper). Both target the same MLE and agree numerically. |
| `--step-rule` | str | `fixed` | Step of the recursive estimator: 'fixed' is alpha_0/t, the schedule of equation (19) (default); 'armijo' is a line search at each update. |
| `--init-mode` | str | `mm` | Initialisation of the recursive estimator: 'mm' warm-starts on the first date (default), 'identity' starts from (I, I, 1) as the released code does. |
| `--alpha-0` | float | — | Initial step (default: 1.0 for 'fixed', 0.1 for 'armijo'). |
| `--mm-iter-max` | int | `50` | Max MM iterations (default 50). |
| `--mm-tol` | float | `1e-08` | MM convergence tolerance (default 1e-8). |
| `--gd-iter-max` | int | `200` | Max iterations of the offline Riemannian gradient descent (default 200). |
| `--gd-tol` | float | `1e-08` | Tolerance of the offline Riemannian gradient descent (default 1e-8). |
| `--debug` | — | — | Tiny configuration (8 trials, T up to 50) to validate the pipeline in seconds. Results are NOT publication grade. |
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

`2-detection/experiments/sar/sar_mc_kron_mse.yaml`
