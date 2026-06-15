# sar_det_on_kron

Online Kronecker GLRT change detection on real SAR data

**Tags:** `detection`  `kronecker`  `online`  `real-data`  `SAR`

## Run

```sh
uv run python 2-detection/sar_experiments/compute_detection_real_data/online_kronecker.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iter-max` | int | `5` | Maximum MM iterations for H0 warm-start and H1 per-date estimates (default 5). |
| `--tol` | float | `0.0001` | Convergence tolerance for MM algorithms (default 1e-4). |
| `data_path` | str | — | Path to the numpy data file (.npy). |
| `window_size` | int | — | Sliding window size. |
| `--backend` | str | `numpy` | Computation backend (default: numpy). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save result (.npy), provenance sidecar (.json), and plot script (_plot.py) (default: True). |
| `--export-tikz` | — | — | Also save a TikZ/PGFPlots figure (.tex) alongside the exported data. |
| `--storage-path` / `--storage_path` / `--export-path` | str | `./exports` | Directory for exported plots; --storage-path is the qanat alias (default: ./exports). |
| `--debug` | — | — | Crop data to 100×100 for fast debugging. |
| `--splitting` | str | — | Grid splitting '(r,c)'. Default: (1,1) CPU, (5,5) GPU. |
| `--wavelet` | — | — | Apply wavelet decomposition before detection. |
| `--wavelet-R` | int | `2` | Range sub-bands (default 2). |
| `--wavelet-L` | int | `2` | Azimuth sub-bands (default 2). |
| `--wavelet-no-decimate` | — | — | Disable sub-band decimation. Output is R*L times larger than input (full-resolution redundant representation). Default: decimation ON. |
| `--quiet` | — | — | Suppress verbose output. |
| `--log-debug` | — | — | Enable debug-level logging. |
| `--report-memory` | — | — | Print peak GPU memory at the end (torch-cuda only). |
| `--repeat-times` | int | `1` | Repeat the time axis N times using a palindrome bounce (e.g. T=68, repeat=2 → 136 frames: 0..67, 66..1, 0..1, ...). Materialises the full repeated array in RAM. |

## Config

`2-detection/experiments/sar/sar_det_on_kron.yaml`
