# Chapter 4 · Deep learning on the SPD manifold

Neural network layers that operate directly on covariance matrices — SPDNet —
and what their building blocks cost: the ReEig threshold, the batch-norm layer,
and the aggregation used when training is split across sites.

The layers come from
[`yetanotherspdnet`](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet),
so this chapter is plain PyTorch and does not use the backend layer the other
chapters share.

## Data

All four experiments here are simulated — nothing to download.

The published versions of this work also used real datasets (HDM05, HyperLeaf,
Rices90, and EEG recordings). Those are not redistributable, so the experiments
that used them are not part of this repository.

## Caveats

**Apple Silicon is not supported.** The scripts refuse `--device mps` rather
than failing obscurely later: Metal has no `float64`, and `torch.linalg.eigh` is
not implemented there — that single operation is behind ReEig, LogEig, `sqrtm`
and every mean used here. Forcing a CPU fallback with
`PYTORCH_ENABLE_MPS_FALLBACK=1` is also pointless: it measured 0.388 s against
0.364 s for running on the CPU directly.

Use `--device cpu` or `--device cuda`.

## Experiments

<!-- experiments-start -->
<div class="exp-chapter">
<div class="exp-group">
<h3 class="exp-group-heading">Deep learning · SPDnet</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">spdnet_batchnorm_cost</div>

</div>
<div class="exp-desc">Time and retained autograd memory of the SPD batch-norm layer, hand-written backward against automatic differentiation</div>
<div class="exp-tags"><span class="exp-tag">deeplearning</span><span class="exp-tag">spdnet</span><span class="exp-tag">batchnorm</span><span class="exp-tag">cost</span></div>
<div class="exp-run"><code>uv run python 4-deeplearning/batchnorm_cost/main.py</code></div>
<a class="exp-details-link" href="../../experiments/spdnet_batchnorm_cost/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">spdnet_reeig_spectrum</div>

</div>
<div class="exp-desc">What the ReEig threshold does to the spectrum of a CovPool matrix, and the 1/eps bound it puts on the Loewner factor of the backward pass</div>
<div class="exp-tags"><span class="exp-tag">deeplearning</span><span class="exp-tag">spdnet</span><span class="exp-tag">reeig</span><span class="exp-tag">covariance</span></div>
<div class="exp-run"><code>uv run python 4-deeplearning/reeig_spectrum/main.py</code></div>
<a class="exp-details-link" href="../../experiments/spdnet_reeig_spectrum/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">spdnet_stiefel_aggregation</div>

</div>
<div class="exp-desc">Order at which the projavg and rlavg aggregations coincide on the Stiefel manifold</div>
<div class="exp-tags"><span class="exp-tag">deeplearning</span><span class="exp-tag">spdnet</span><span class="exp-tag">stiefel</span><span class="exp-tag">federated</span></div>
<div class="exp-run"><code>uv run python 4-deeplearning/stiefel_aggregation/main.py</code></div>
<a class="exp-details-link" href="../../experiments/spdnet_stiefel_aggregation/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">spdnet_wishart_model</div>

</div>
<div class="exp-desc">Does the batch-norm mean matched to the Wishart model win, and by how much, as the degrees of freedom grow</div>
<div class="exp-tags"><span class="exp-tag">deeplearning</span><span class="exp-tag">spdnet</span><span class="exp-tag">batchnorm</span><span class="exp-tag">wishart</span></div>
<div class="exp-run"><code>uv run python 4-deeplearning/wishart_model/df_sweep.py</code></div>
<a class="exp-details-link" href="../../experiments/spdnet_wishart_model/">Parameters &amp; details →</a>
</div>
</div>
</div>
</div>
<!-- experiments-end -->
