# sar_det_off_gauss

Offline Gaussian GLRT change detection on real SAR data

**Tags:** `detection`  `gaussian`  `offline`  `real-data`  `SAR`

## Run

```sh
uv run python 2-detection/sar_experiments/compute_detection_real_data/offline_gaussian.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
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

## Results

<span class="marginnote">
  <span class="mn-label">Run · scene1_w7</span>
  <span class="mn-date">Generated: 2026-06-13</span><br>
  <code>data_path</code><br>
  <code>window_size</code><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--show-interactive</code><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--export-tikz</code><br>
  <code>--debug</code><br>
  <code>--splitting</code><br>
  <code>--wavelet</code><br>
  <code>--wavelet-R</code> <span class='mn-default'>2</span><br>
  <code>--wavelet-L</code> <span class='mn-default'>2</span><br>
  <code>--wavelet-no-decimate</code><br>
  <code>--quiet</code><br>
  <code>--log-debug</code><br>
  <code>--report-memory</code><br>
  <code>--repeat-times</code> <span class='mn-default'>1</span><br>
</span>
<div class="plotly-wrap" data-src="../../assets/data/sar_det_off_gauss.scene1_w7.json" data-title="sar_det_off_gauss — scene1_w7"></div>

## Config

`2-detection/experiments/sar/sar_det_off_gauss.yaml`
