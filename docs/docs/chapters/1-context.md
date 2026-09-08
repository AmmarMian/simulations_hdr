# Chapter 1 · Context

Figures illustrating the statistical ideas the later chapters build on:
elliptical distributions and how they differ from the Gaussian, what a
covariance estimated from few samples looks like, and how the space of
covariance matrices behaves when it is treated as a curved surface rather than
a flat one.

## Data

All simulated — nothing to download, and every experiment runs in seconds on a
laptop CPU.

## Caveats

These are illustrations, not measurements: most draw a few hundred points to
make a picture, and their parameters are chosen for legibility rather than
statistical power. The two Monte-Carlo ones (`context_scm_grandnombres` and
`context_scm_underregime`) are the exception and take about a minute.

## Experiments

<!-- experiments-start -->
<div class="exp-chapter">
<div class="exp-group">
<h3 class="exp-group-heading">Context · Illustrations</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_complex_circularity</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Same covariance, four pseudo-covariances — what circularity buys and what it hides</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">complex</span><span class="exp-tag">circularity</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/complex_circularity/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_complex_circularity/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_elliptical_examples</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Isodensity contours and draws for elliptical distributions sharing one scatter matrix</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">elliptical</span><span class="exp-tag">distributions</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/elliptical_examples/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_elliptical_examples/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_example_covariances</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">PGFPlots matrix visualisations of covariance regimes — identity, Toeplitz, random</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">covariance</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/examples_covariances/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_example_covariances/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_gaussian_isocontours</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Isodensity contours and samples of the bivariate Gaussian for three covariance regimes</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">gaussian</span><span class="exp-tag">distributions</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/probability_densities/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_gaussian_isocontours/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_riemann_convergence</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Fixed point, Riemannian descent and projected Euclidean descent on Tyler's cost</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">riemann</span><span class="exp-tag">optimisation</span><span class="exp-tag">robust</span></div>
<div class="exp-run"><code>uv run python 1-context/riemann_convergence/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_riemann_convergence/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_riemann_frobenius_rao</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Estimation error of the SCM and of Tyler's estimator, in Frobenius norm and in Rao distance</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">riemann</span><span class="exp-tag">robust</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 1-context/riemann_frobenius_rao/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_riemann_frobenius_rao/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_riemann_gconvexite</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Tyler's cost read along a Euclidean segment and along an affine-invariant geodesic</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">riemann</span><span class="exp-tag">robust</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/riemann_gconvexite/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_riemann_gconvexite/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_riemann_interpolation</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Euclidean, affine-invariant and log-Euclidean paths between two covariance matrices</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">riemann</span><span class="exp-tag">geometry</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/riemann_interpolation/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_riemann_interpolation/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_riemann_moyennes</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Arithmetic, log-Euclidean and Fréchet means of a cloud of covariance matrices</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">riemann</span><span class="exp-tag">geometry</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/riemann_moyennes/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_riemann_moyennes/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_robust_mestimation</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Concentration ellipses of the SCM, the model MLE and Tyler's estimator on a single draw</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">robust</span><span class="exp-tag">m-estimation</span><span class="exp-tag">illustration</span></div>
<div class="exp-run"><code>uv run python 1-context/robust_mestimation/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_robust_mestimation/">Parameters &amp; details →</a>
</div>
</div>
</div>

<div class="exp-group">
<h3 class="exp-group-heading">Context · LWF</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_lwf_underregime</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">Condition number of SCM vs LWF-regularised covariance in the under-regime (d > N)</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">lwf</span><span class="exp-tag">regularisation</span><span class="exp-tag">under-regime</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 1-context/lwf_underregime/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_lwf_underregime/">Parameters &amp; details →</a>
</div>
</div>
</div>

<div class="exp-group">
<h3 class="exp-group-heading">Context · SCM</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_scm_grandnombres</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">MC convergence of SCM mean/covariance estimators as N → ∞ (grand-nombre regime)</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">scm</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">convergence</span></div>
<div class="exp-run"><code>uv run python 1-context/scm_grandnombres/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_scm_grandnombres/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_scm_underregime</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">SCM estimator behavior in the under-regime (d > N) — mean, covariance, condition number errors</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">scm</span><span class="exp-tag">under-regime</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 1-context/scm_underregime/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_scm_underregime/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">context_wishart_mse</div>

</div>
<div class="exp-desc">Monte-Carlo check of the closed-form MSE of the SCM under a Gaussian model, swept in the sample support and in the dimension</div>
<div class="exp-tags"><span class="exp-tag">context</span><span class="exp-tag">scm</span><span class="exp-tag">wishart</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 1-context/wishart_mse/main.py</code></div>
<a class="exp-details-link" href="../../experiments/context_wishart_mse/">Parameters &amp; details →</a>
</div>
</div>
</div>
</div>
<!-- experiments-end -->
