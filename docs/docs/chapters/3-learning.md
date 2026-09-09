# Chapter 3 · Learning under a corrected model

Clustering covariance matrices on the SPD cone, comparing four ways of
estimating the centre of a cluster: the sample covariance, two shrinkage
estimators, and a random-matrix correction.

The estimators and the corrected mean are in
[`hdrlib.learning.rmt`](../api/learning/rmt.md); the two K-means in
[`hdrlib.learning.clustering`](../api/learning/clustering.md).

## Data

Two hyperspectral scenes, Indian Pines and Salinas, are downloaded on first run
into `data/hyperspectral/` (about 30 MB). Nothing to prepare by hand.

Both are public benchmarks with a per-pixel ground truth of crop types, which is
what the segmentations are scored against.

## Caveats

**float64 is required.** These methods take logarithms of the eigenvalues of
small covariance matrices, and single precision is not enough. `torch-mps` is
therefore unavailable — Metal has no double precision — and the jax backends are
refused, since this repository does not enable jax's `x64` mode. Use `numpy`,
`torch-cpu`, `torch-cuda` or `cupy`.

**One experiment is slow.** `learning_hyperspectral_rmt` on the full Salinas
scene takes about an hour on an RTX 4000 Ada; the corrected method alone is two
thirds of that. Use `--stride 3` for a quicker look at reduced resolution. The
other three experiments run in seconds to minutes.

**Inertia is not comparable across methods.** Each geometry measures lengths
differently, so inertia only ranks restarts within one method. Use accuracy and
mIoU, which are computed against the ground truth.

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
