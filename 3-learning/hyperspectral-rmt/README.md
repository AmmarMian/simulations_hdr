# Clustering a hyperspectral scene

Serves the segmentation-map figure — the one that judges the correction
**downstream** rather than on itself.

## Why this experiment and not the MSE alone

The MSE figure measures an *internal* criterion: the distance between the
estimated mean and the true one. But the claim being made is that the criterion
has left the model, so the estimator has to be judged on the task, and a
segmentation map shows that at a glance.

## The pipeline

```
scene → remove the global mean → PCA to n_features bands
      → sliding window → one covariance per pixel
      → Riemannian K-means → comparison against the ground truth
```

This is where the dimensional regime becomes concrete: a 5×5 window on 5
principal components gives **25 samples for 5 variables**, so $c = 0.2$. Many
matrices, each of them poorly estimated — exactly the regime of the second
panel of the MSE figure.

## What separates the four methods

One thing only: the distance the centroids minimise. The alternation loop, the
restarts and the choice of the best one by inertia are shared and written once.
A comparison therefore measures the correction, not the optimiser.

This differs from the published protocol, where the baselines went through a
different optimiser than the corrected method. The baselines here are markedly
better than the published ones and the gap in favour of the correction is
narrower — but it is a gap in the only thing that ought to vary.

## Running it

```sh
uv run qanat experiment run learning_hyperspectral --scene indianpines --n_init 10
uv run qanat experiment run learning_hyperspectral --scene salinas --stride 2 --n_init 5
```

Indian Pines (145×145) takes about an hour at `stride 1`. Salinas (512×217) is
twenty times larger: `--stride 2` brings it back to a comparable cost, at the
price of map resolution — which the caption has to say.

## Data

`download_scene` fetches the `.mat` files on first run and checks that what
came back really is one: some mirrors answer 200 with a challenge page, and the
error would otherwise only surface three steps later. If the download fails,
put the files in `data/hyperspectral` by hand.

## Constraints

float64, so no `torch-mps` — see `hdrlib.learning.rmt.require_double`. No
scikit-learn either: it is numpy-only, and the pipeline has to be able to run
on a GPU backend.
