# Does the mean matched to the model win, and by how much?

Serves the central point of the section on means in batch normalisation. No
real data.

## Why

Each mean is identified with the Fréchet mean of a geometry or of a divergence:
arithmetic ↔ left KL (Wishart), harmonic ↔ right KL (inverse Wishart), GAH ↔
symmetrised KL. The intent is to make this **a modelling statement**, not a
numerical curiosity. But the only support available so far is F1 tables on
three real datasets: an *indirect* argument, on data whose law is unknown.

The `wishart-inverse` grid tests the statement **head-on**: draw from a
Wishart, then from an inverse Wishart, and see which mean wins.

```sh
python -m eusipco_2026.simulation --wishart-inverse --cpu --n-jobs 8
```

**This directory adds one axis only**, which that grid does not sweep: the
degrees of freedom. As $df$ grows, the law concentrates around its scale matrix
and the choice of mean must matter less and less. The mechanism then appears as
a **gradient** rather than as two points — which is what makes it a statement
about the model rather than about one particular setting.

Everything else is reused: `run_single_experiment` from `eusipco_2026` does the
generation, the training and the evaluation. This file only sweeps and plots.

## The original grid's result ($df = 64$, $64\times64$ matrices)

| mean | Wishart data | inverse Wishart data |
|---|---|---|
| arithmetic (left KL) | **90.6 % ± 4.2** | 55.3 % ± 22.1 |
| harmonic (right KL) | 46.1 % ± 8.5 | **88.9 % ± 5.7** |
| GAH (symmetrised KL) | 76.7 % ± 5.6 | 81.4 % ± 5.9 |
| ARMAGNAC (adaptive GAH) | 86.7 % ± 2.3 | 87.2 % ± 6.2 |
| geometric (affine invariant) | 90.0 % ± 4.9 | 88.1 % ± 5.3 |

The mean matched to the model wins and the opposite one collapses — and the
22 % standard deviation of the arithmetic mean on inverse Wishart data says the
same thing another way: under the wrong model, training is not even
reproducible from one seed to the next.

## Two caveats to respect

1. On this data, **the geometric mean is the best of the five almost
   everywhere** (90.0 / 88.1). So this figure does *not* say "GAH beats the
   geometric mean" — it says "the mean matched to the model wins, and the
   symmetric means are robust to the model". That is a stronger and more
   defensible statement. The claim that "the geometric mean is never the best"
   remains what it is: a fact **about the three real datasets**, to be kept
   separate and on no account mixed with this one. Kept apart, the two
   reinforce each other: the simulation establishes the mechanism, the real
   data shows that on those datasets the Gaussian assumption is not the right
   one.
2. The scale matrices of the classes differ by a 25 % perturbation: the
   separation between classes is a **setting**, not a property. Say so.

## Running it

`eusipco_2026` and `spdnet-datasets` are not dependencies of this repository:

```sh
uv pip install git+https://github.com/Yet-Another-Research-Organisation/spdnet-datasets.git
uv pip install --no-deps git+https://github.com/Yet-Another-Research-Organisation/eusipco_2026.git
python df_sweep.py
```

`--df 64 96 160 320 640` by default; $df$ must exceed `matrix_size - 1`.
