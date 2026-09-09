# Estimation error of the Fréchet mean

Serves the MSE figure. No real data.

## What it measures

The Fréchet mean of a set of covariances is what every nearest-centroid
classifier and every Riemannian $K$-means computes. In practice the true
covariances are not available, only their estimates, in the regime where $d$
and $N$ are comparable — and there the Fréchet mean of the SCMs is biased.

Five estimators, two sweeps:

| | |
|---|---|
| `SCM` | Fréchet mean of the sample covariances |
| `LW`, `OAS` | of the linearly shrunk covariances |
| `LW-NL` | of the non-linearly shrunk covariances |
| `RMT` | mean corrected by random matrix theory |

The first four regularise each covariance **before** averaging; the last
corrects the **distance** the average minimises. These are not the same
gesture, and separating them is the point of the figure.

## The two panels

**Against the number of samples $N$** ($K = 10$): the gap closes as $N$ grows —
8.1 dB of gain at $N = 65$, 2.2 dB at $N = 300$. The signature of a
regime-induced bias, not of a variance.

**Against the number of matrices $K$** ($N = 128$): the gap *widens*. The SCM
plateaus (12.7 dB at $K = 3$, 8.2 dB at $K = 100$, most of it reached by
$K = 20$) while the corrected mean keeps descending to −4.3 dB. Averaging
reduces variance, not bias: the bias is common to every SCM and survives the
average intact.

The second panel is the one that carries the argument, and the one intuition
gets wrong.

## Running it

```sh
uv run qanat experiment run learning_frechet_mse --n_features 64 --n-trials 100
```

About fifty minutes on ten cores; the $K = 100$ point dominates the cost. The
trials are independent and spread over a pool (`--n-workers`), each seeded by
`(axis index, trial)`, so a single point can be replayed without replaying the
whole sweep.

## Constraints

**float64 is mandatory.** The corrected gradient divides by differences of
eigenvalues, and several of its terms are built so that a diagonal entry
evaluates the finite limit of an expression that is $0/0$ elsewhere. In single
precision those terms lose every significant digit without reporting anything:
the descent still returns a matrix, and it is wrong.
`hdrlib.learning.rmt.require_double` therefore refuses to start.

The practical consequence is that **`torch-mps` cannot run this code**, Metal
having no float64. On Apple Silicon, use `--backend torch-cpu`.

## Display convention

Decibels are `10*log10(mse)` — the power convention, which is the right one for
a squared error.
