# Hyperspectral clustering: what the geometry buys

Follows the protocol of pyRiemann's `image-radar` example — same scene, same
window, same estimator — without pyRiemann, and on the device.

## The question

The neighbouring experiment (`3-learning/hyperspectral-rmt`) asks whether the
RMT correction survives downstream. This one asks the prior question: **before
any correction, how much of the gain is due to the metric?** Same scene, same
windows, same covariances, same alternation — only the geometry in which the
centroids are means changes.

| metric | centroid | closed form | cost |
|---|---|---|---|
| `euclid` | arithmetic mean of the covariances | yes | one matrix product |
| `logeuclid` | mean of the matrix logarithms | yes | N decompositions, **once** |
| `riemann` | Karcher mean (affine-invariant) | no | N·K decompositions **per iteration** |

`logeuclid` is not in the pyRiemann example. It is added because it costs
almost nothing once the logarithms are cached, and because it separates two
effects that `euclid` against `riemann` conflates: respecting the positivity of
the eigenvalues, and being invariant under affine transformation.

## The pipeline

```
scene → remove the global mean → scale normalisation
      → PCA to n_features bands → sliding window → one covariance per pixel
      → K-means (euclid | logeuclid | riemann) → comparison against the ground truth
```

A 5×5 window on 5 components gives **25 samples for 5 variables**, so
$c = 0.2$: many matrices, each of them poorly estimated. That is the regime the
neighbouring experiment's correction targets; here it is the setting, not the
subject.

## Everything on the device

The cube crosses the bus once and the labels come back once. In between,
`hdrlib.learning.clustering.spd_kmeans` reads back only two scalars per
iteration — the fraction of points that changed group, which decides when to
stop, and the inertia, which picks the best restart. Those are branches of the
program: they have to become Python numbers.

What makes this possible is the absence of a correction. None of the three
metrics needs the samples the covariance came from, so the covariances are
formed **once**, before the restarts, and the windows are freed before the loop
begins — 108 MB released against 22 MB kept on Salinas. The neighbouring
experiment cannot do this: the corrected mean reads the samples again.

Assignment is batched over the centroids rather than looped, and re-estimation
is a product against a one-hot membership matrix, which averages the K groups
at once. The sordid detail: no `scatter` primitive is common to numpy, torch,
cupy and jax — a matrix product is.

## float64, mandatory

All three metrics end up on the eigenvalues of a 5×5 covariance estimated from
25 samples, and two of them take the logarithm. In float32 the sign of the
smallest eigenvalue is not reliable, and its logarithm is either a large
negative number or a NaN — silently. The check is made after the transfer, not
before: it is the transfer that degrades.

Consequences: no `torch-mps` (Metal has no float64, and `get_data_on_device`
downgrades without saying so), and **no `jax-*` either** — nothing in this
repository calls `jax.config.update("jax_enable_x64", True)`, so JAX would
compute the whole thing in single precision without even a warning. The backend
refuses them explicitly.

## Running it

```sh
uv run qanat experiment run learning_hyperspectral_metrics --scene salinas --backend torch-cuda
uv run qanat experiment run learning_hyperspectral_metrics --scene indianpines --backend torch-cuda --n_init 20
```

## Reading the scores

`scores.json` holds the timings **and** the description of the card. The two
are read together: float64 runs at half the float32 rate on a datacentre card
and at a sixty-fourth of it on a workstation card. `riemann` is bound by
eigendecompositions, the two flat metrics by matrix products — so the *ranking
by time* is as much a property of the card as of the method.

Inertia compares the restarts of **one** metric and nothing else: the three
measure lengths in different geometries. What ranks the metrics is the accuracy
and the mIoU, which are against the ground truth.
