# HDR Simulations

Companion code for the HDR dissertation. Each chapter directory reproduces a set of figures; this site documents the structure, explains how to run experiments, and provides a searchable reference for all ~100s of experiment configurations and the underlying library API.

## Chapter map

| Directory | Chapter | Stack |
|-----------|---------|-------|
| `1-context/` | Context & background figures | numpy, scipy |
| `2-detection/` | Change detection in SAR imagery | numpy · torch · cupy · jax |
| `3-machinelearning/` | *(forthcoming)* | scikit-learn · torch |
| `4-deeplearning/` | *(forthcoming)* | torch |

## Shared utilities

`shared/plot_style.py` — dark serif theme applied across all chapters for consistent dissertation aesthetics.

## Installation

Each chapter declares its own extras. Install only what you need:

```sh
# base (numpy, torch-cpu, scipy)
uv sync

# chapter 2 — optional compute backends
uv sync --extra cupy        # NVIDIA CUDA
uv sync --extra jax         # JAX CPU
uv sync --extra jax-cuda    # JAX CUDA
uv sync --extra jax-metal   # Apple Silicon
```

## Running an experiment

Experiments are managed by [qanat](https://github.com/ammarmian/qanat). Each YAML file in a chapter's `experiments/` directory defines one experiment.

```sh
# list all experiments for chapter 2
qanat experiment list

# run a specific experiment
qanat experiment run sar_mc_kron_h1

# plot results from the last run
qanat action run sar_mc_kron_h1 plot
```

Results land in `results/<experiment_name>/run_<N>/` alongside a self-contained `_plot.py` that can be run standalone.
