# Chapter 4 · Deep learning on the SPD manifold

Code for the figures of `ch:spdnet`. Unlike chapters 1 to 3, this one **does not
go through `hdrlib.core.backend`**: the layers are those of
[`yetanotherspdnet`](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet),
so plain PyTorch. Only the export harness
([`hdrlib.core.exporter`](../api/core/exporter.md)) and the plot style
([`hdrlib.core.plot_style`](../api/core/plot_style.md)) are shared.

## Scope

What this chapter replays and what it cites is settled in
[`4-deeplearning/NOTE-reproduction.md`](https://github.com/AmmarMian/simulations_hdr/blob/main/4-deeplearning/NOTE-reproduction.md).
In short: nothing is replayed from the GPR, from the three real batch-norm
datasets, or from the federated EEG — the data is not distributable or the
repository does not exist — and two figures that appear in no paper are produced
here because the dissertation argues from them and they cost seconds.

Each experiment has its own `README.md` giving the statement it serves, the
measured result and the command line.

## Hardware

**MPS (Apple Silicon) is refused explicitly**, and the scripts say so rather
than degrading silently. MPS has no `float64`, and `torch.linalg.eigh` is not
implemented there — that is the operation behind ReEig, LogEig, `sqrtm` and the
chapter's five means. With `PYTORCH_ENABLE_MPS_FALLBACK=1` everything falls back
to the CPU one operation at a time: `eigh(256×64×64)×10` measured at 0.388 s
against 0.364 s on pure CPU.

Use `--device cpu` or `--device cuda`.

## Dependency

`yetanotherspdnet` is not yet a dependency of this repository, because its
upstream packaging is broken: `packages = ["yetanotherspdnet"]` ships no
subpackage, and the import fails on a spurious circular import. The fix is ready
on the `fix/whitening-congruence-matrix-grad` branch.

Until that lands, these three experiments will not run from a plain `uv sync`.
Point `PYTHONPATH` at a checkout of the library:

```sh
git clone -b fix/whitening-congruence-matrix-grad \
    https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet
export PYTHONPATH=$PWD/yetanotherspdnet/src
```

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
<div class="exp-desc">Order at which the projavg and rlavg aggregations of prop:spdnet-federe-equivalence coincide on the Stiefel manifold</div>
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
