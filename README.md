# HDR Simulations

Code reproducing the figures and numerical results of my HDR dissertation:

> **Matrices de covariances : des statistiques multivariées à l'apprentissage profond**
> Ammar Mian

The dissertation is in French; this repository and its documentation are in
English, being useful to a broader community.

📖 **[Full documentation](https://ammarmian.fr/simulations_hdr/)** — start
there. It covers the tooling, every experiment, and the provenance chain from a
run to a figure.

## What is here

`hdrlib/` is a small library for working with covariance matrices — estimation,
Riemannian geometry on the SPD cone, random-matrix corrections, detection. It
runs on numpy, torch (CPU/CUDA), cupy or jax through one backend layer, so the
same script runs on a laptop and on a GPU.

The numbered directories hold the experiments, one per dissertation chapter:

| | Chapter | Topic |
|---|---|---|
| `1-context/` | 1 | Background illustrations |
| `2-detection/` | 2 | Change detection in SAR and sonar |
| `3-learning/` | 3 | Learning under a corrected model |
| `4-deeplearning/` | 4 | Deep learning on the SPD manifold |

## Quick start

Needs Python ≥ 3.12, [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems).

```sh
git clone https://github.com/AmmarMian/simulations_hdr
cd simulations_hdr
uv sync --extra chapters     # environment for all four chapters, including qanat
just init-qanat              # create the local experiment database
just register-experiments    # register the 40 experiments with qanat
```

Run one experiment directly:

```sh
uv run python 3-learning/marchenko_pastur/main.py --no-export --show-interactive
```

or through qanat, which tracks the run and its outputs:

```sh
uv run qanat experiment run learning_marchenko_pastur
uv run qanat experiment status learning_marchenko_pastur
```

`just --list` shows every recipe. Extras come on two axes that compose: one per
chapter (`--extra context`, `detection`, `learning`, `deeplearning`) and one per
GPU backend (`--extra cupy`, `--extra jax-cuda`); see
[Getting started](https://ammarmian.fr/simulations_hdr/getting-started/)
for the full matrix and for how the chapters differ.

## Tests

```sh
uv run pytest tests/ -q
```

Backends that are not installed skip themselves, so the suite is meaningful on
any machine; CUDA-specific tests run only where a GPU is present.

## Caveats

- Real sonar and GPR datasets are not redistributable, so those experiments are
  documented but not runnable from this repository.
- Hyperspectral scenes (Indian Pines, Salinas, ~30 MB) download on first use.
