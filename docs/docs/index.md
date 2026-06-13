# HDR Simulations

Cette page regroupe la documentation qui accompagne ma dissertation d'HDR intitulée:
> Matrices de covariances : des statistiques multivariées à l'apprentissage profond



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
