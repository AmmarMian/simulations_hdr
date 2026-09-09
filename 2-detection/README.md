# HDR Chapter 2

Code to reproduce the results of chapter 2.

## Installation

```sh
uv sync                                  # base deps (numpy, torch-cpu, scipy…)
uv sync --extra cupy                     # CuPy / CUDA GPU
uv sync --extra jax                      # JAX CPU
uv sync --extra jax-cuda                 # JAX CUDA GPU
uv sync --extra cupy --extra jax         # combine extras freely
```

## Data

Some figures need real sonar and SAR data:

* The sonar data cannot be distributed.
* For the SAR data, go to `./data/` and run `bash download_sar.sh`.

### Preparing the data for the experiment scripts

`compute_cd_online.py`, `compute_cd_offline.py` and
`compute_cd_kronecker_offline.py` expect the data **time-first**, as
`(n_times, n_rows, n_cols, n_features)`, for efficient memory access. Once the
SAR data is downloaded, convert each file with `prepare_data.py`:

```bash
uv run sar_experiments/prepare_data.py data/SAR/scene1.npy
uv run sar_experiments/prepare_data.py data/SAR/scene2.npy
uv run sar_experiments/prepare_data.py data/SAR/Scene4_cropped.npy
```

Each command writes a `<name>_time_first.npy` file next to the original. The
experiment scripts check for that file and print these instructions if it is
missing.
