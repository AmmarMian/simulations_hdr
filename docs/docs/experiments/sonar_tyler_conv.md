# sonar_tyler_conv

2TYL fixed-point convergence — relative Frobenius deviation vs iteration

**Tags:** `sonar`  `estimation`  `monte-carlo`  `convergence`

## Run

```sh
uv run python 2-detection/sonar_experiments/mc_simulations/tyler_convergence.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iter-max` | int | `500` | Number of fixed-point iterations to track (default 500). |
| `--K-conv` | int | — | Secondary samples for convergence experiment (default = K_secondary). |
| `--m` | int | `64` | Per-array dimension (total = 2m, default 64). |
| `--beta` | float | `0.0003` | Covariance scale factor beta (default 3e-4). |
| `--rho1` | float | `0.4` | Array-1 correlation coefficient rho1 (default 0.4). |
| `--rho2` | float | `0.9` | Array-2 correlation coefficient rho2 (default 0.9). |
| `--K` | int | — | Secondary data size K (default = 2*2m = 4m). |
| `--theta1` | float | `45.0` | Array-1 steering angle in degrees (default 45). |
| `--theta2` | float | `45.0` | Array-2 steering angle in degrees (default 45). |
| `--gaussian` | — | `gaussian` | Gaussian clutter (default). |
| `--k-dist` | — | — | K-distributed clutter. |
| `--nu` | float | `0.5` | K-distribution shape parameter nu (default 0.5). |
| `--snr-min` | float | `-25.0` | Minimum SNR in dB (default -25). |
| `--snr-max` | float | `5.0` | Maximum SNR in dB (default 5). |
| `--n-snr` | int | `150` | Number of SNR values (default 150). |
| `--pfa` | float | `0.01` | Nominal PFA for PD curves (default 1e-2). |
| `--n-trials` | int | `10000` | Number of Monte-Carlo trials (default 10000). |
| `--seed` | int | `42` | RNG seed for data generation (default 42). |
| `--backend` | str | `numpy` | Compute backend. numpy → multiprocessing.Pool (one worker per trial); all others → trials stacked in leading batch dimension (default numpy). |
| `--n-workers` | int | — | Pool workers for numpy backend (default: os.cpu_count()). |
| `--export` | — | `True` | Save .npz results + provenance sidecar + plot script (default: True). |
| `--storage-path` / `--storage_path` / `--export-path` | str | `./exports` | Directory for exported results; --storage-path is the qanat alias (default: ./exports). |
| `--show-interactive` | — | — | Display figures interactively at the end of the simulation. |

## Config

`2-detection/experiments/sonar/sonar_tyler_conv.yaml`
