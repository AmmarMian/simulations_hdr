# Chapter 3 · Learning under a corrected model

Code for the figures of `ch:learning`. Like chapters 1 and 2, and unlike
chapter 4, these experiments go through
[`hdrlib.core.backend`](../api/core/backend.md): one script runs on numpy,
torch, cupy or jax.

The correction and the corrected mean live in
[`hdrlib.core.rmt`](../api/core/rmt.md), the two K-means in
[`hdrlib.core.clustering`](../api/core/clustering.md). Both carry over from
[`AmmarMian/icml-rmt-2024`](https://github.com/AmmarMian/icml-rmt-2024).

## Scope

Both pieces of work in this chapter are conference papers, reproduced in full at
the end of the chapter in the dissertation. Most of their numerical campaigns
are **not** replayed here — the learned graphs (`animals`, GNSS) stay in the
paper, with their context. What is replayed is what the dissertation argues from
and must therefore be able to show under its own provenance chain.

| Experiment | Data | Status |
|---|---|---|
| `learning_marchenko_pastur` | simulated | done |
| `learning_frechet_mse` | simulated | done |
| `learning_hyperspectral_metrics` | Salinas | done |
| `learning_hyperspectral_rmt` | Indian Pines | optional |

What each one shows, and what it is evidence for, belongs to the dissertation
and is not restated here.

## Requirements

Everything in this chapter needs float64. `torch-mps` is therefore out: Metal
has no double precision. `riemannian_kmeans` and `spd_kmeans` refuse the jax
backends outright, because nothing in this repository enables `x64` and jax
would otherwise compute the eigenvalue logarithms in single precision without
warning.

The two hyperspectral scenes are downloaded on first use into
`data/hyperspectral/` (about 30 MB); no manual step is needed.

`learning_hyperspectral_rmt` is the expensive one. On an RTX 4000 Ada, the four
methods over the full Salinas scene — 108 204 covariances, 16 classes, 5
restarts, 30 rounds — take about an hour, of which the corrected method is
two thirds. The other three experiments are seconds to minutes.

## Shared protocol

The two hyperspectral experiments write their alternation loop, their restarts
and their selection of the best restart by inertia **once**, in
`hdrlib.core.clustering`, and every method starts from the same initial
partition at equal seed.

This differs from the published protocol, where the baselines went through a
different optimiser than the corrected method, so the measured gap combined the
effect of the estimator with the effect of the optimiser and separated neither.
The baselines are noticeably stronger here and the margin in favour of the
correction is narrower, but it is a margin over the one thing that ought to
vary.

One caveat when reading the outputs: inertia compares restarts of a **single**
metric and nothing else, since the three measure lengths in different
geometries. Only accuracy and mIoU, which are computed against the ground truth,
rank the methods against each other.

## Experiments

<!-- experiments-start -->
<div class="exp-chapter">
<div class="exp-group">
<h3 class="exp-group-heading">Learning · Random matrix theory</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">learning_frechet_mse</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">MSE of the Fréchet mean of a set of covariances, against the number of samples and against the number of matrices — SCM, Ledoit-Wolf, OAS, non-linear shrinkage and the RMT correction</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">random-matrix-theory</span><span class="exp-tag">frechet-mean</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 3-learning/frechet_mse/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_frechet_mse/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">learning_hyperspectral_metrics</div>

</div>
<div class="exp-desc">Euclidean, log-Euclidean and affine-invariant K-means on a hyperspectral scene — what the geometry alone buys, before any correction, entirely on the device</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">clustering</span><span class="exp-tag">hyperspectral</span><span class="exp-tag">gpu</span></div>
<div class="exp-run"><code>uv run python 3-learning/hyperspectral-metrics/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_hyperspectral_metrics/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">learning_hyperspectral_rmt</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Riemannian K-means segmentation of a hyperspectral scene — SCM, Ledoit-Wolf, non-linear shrinkage and the RMT correction, judged on the ground truth</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">random-matrix-theory</span><span class="exp-tag">clustering</span><span class="exp-tag">hyperspectral</span></div>
<div class="exp-run"><code>uv run python 3-learning/hyperspectral-rmt/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_hyperspectral_rmt/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">learning_marchenko_pastur</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Marchenko-Pastur law — histogram of the SCM eigenvalues against the theoretical density, for three concentration ratios</div>
<div class="exp-tags"><span class="exp-tag">learning</span><span class="exp-tag">random-matrix-theory</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 3-learning/marchenko_pastur/main.py</code></div>
<a class="exp-details-link" href="../../experiments/learning_marchenko_pastur/">Parameters &amp; details →</a>
</div>
</div>
</div>
</div>
<!-- experiments-end -->
