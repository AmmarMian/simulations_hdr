<div class="page-header">
  <div class="eyebrow">Reference</div>
  <h1>API</h1>
  <p class="standfirst">Modules organised by chapter, plus cross-chapter shared
  utilities. All public classes and functions are documented with type
  signatures and NumPy-style docstrings.</p>
</div>

<div class="api-section">
  <div class="api-section-head">
    <span class="api-section-num">Shared</span>
    <span class="api-section-title">Cross-chapter utilities</span>
  </div>
  <div class="api-index">

  <a class="api-card" href="plot_style/">
    <div class="api-card-head">
      <span class="api-mod">shared.plot_style</span>
      <span class="api-badge">style</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Dark serif matplotlib theme. Call <code>apply_style()</code>
      once per process to configure rcParams globally. Exports
      <code>DARK_STYLE_DICT</code> and <code>EMBEDDED_STYLE_CODE</code> for
      standalone scripts.</p>
    </div>
  </a>

  <a class="api-card" href="plotly_style/">
    <div class="api-card-head">
      <span class="api-mod">shared.plotly_style</span>
      <span class="api-badge">style</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Plotly design tokens for the docs design system.
      Provides colour constants, font stacks, and <code>hex_to_rgba()</code>
      for all <code>action_add_to_docs.py</code> scripts.</p>
    </div>
  </a>

  </div>
</div>

<div class="api-section">
  <div class="api-section-head">
    <span class="api-section-num">2 · Detection</span>
    <span class="api-section-title">Change detection under elliptical distributions</span>
  </div>
  <div class="api-index">

<a class="api-card" href="backend/">
  <div class="api-card-head">
    <span class="api-mod">src.backend</span>
    <span class="api-badge">utility</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">Hardware-agnostic dispatch layer. Detects and normalises NumPy,
    CuPy, JAX, and PyTorch backends from a string or enum, and moves data between devices.</p>
  </div>
</a>

<a class="api-card" href="estimation/">
  <div class="api-card-head">
    <span class="api-mod">src.estimation</span>
    <span class="api-badge">core</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">Fixed-point M-estimators for scatter matrix estimation under
    elliptical distributions. Includes Tyler, Huber, Student-t, and natural-gradient
    scaled-Gaussian estimators.</p>
  </div>
</a>

<a class="api-card" href="estimation_kronecker/">
  <div class="api-card-head">
    <span class="api-mod">src.estimation_kronecker</span>
    <span class="api-badge">core</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">MM algorithms for Kronecker-structured scatter estimation under
    both H₀ and H₁. Riemannian gradient and Armijo line-search for the scaled-Gaussian
    Kronecker model.</p>
  </div>
</a>

<a class="api-card" href="estimation_online/">
  <div class="api-card-head">
    <span class="api-mod">src.estimation_online</span>
    <span class="api-badge">core</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">Online (sample-by-sample) estimators for streaming data.
    Natural-gradient updates on the HPD manifold and an online Kronecker estimator
    for memory-constrained settings.</p>
  </div>
</a>

<a class="api-card" href="manifolds/">
  <div class="api-card-head">
    <span class="api-mod">src.manifolds</span>
    <span class="api-badge">geometry</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">Riemannian manifold primitives: matrix square roots, logarithms,
    geodesic distances, and tangent-space projections for HPD, SHPD, and product manifolds.</p>
  </div>
</a>

<a class="api-card" href="detection/">
  <div class="api-card-head">
    <span class="api-mod">src.detection</span>
    <span class="api-badge">core</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">Abstract base classes for batch and online detectors.
    Subclass <code>Detector</code> or <code>OnlineDetector</code> to implement
    new GLRT-based decision statistics.</p>
  </div>
</a>

<a class="api-card" href="simulation/">
  <div class="api-card-head">
    <span class="api-mod">src.simulation</span>
    <span class="api-badge">experiment</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">Data generators for Monte-Carlo experiments. Gaussian and
    DCG (doubly correlated Gaussian) distributions under H₀ and H₁, with Kronecker
    structured ground-truth covariances.</p>
  </div>
</a>

<a class="api-card" href="exporter/">
  <div class="api-card-head">
    <span class="api-mod">src.exporter</span>
    <span class="api-badge">experiment</span>
  </div>
  <div class="api-card-body">
    <p class="api-desc">Structured result export with git provenance. Saves NumPy
    arrays alongside run metadata — parameters, timestamp, and commit SHA — for
    fully reproducible experiments.</p>
  </div>
</a>

  </div>
</div>
