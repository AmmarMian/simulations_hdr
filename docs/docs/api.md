<div class="page-header">
  <div class="eyebrow">Reference</div>
  <h1>API</h1>
  <p class="standfirst">Three subpackages — <code>hdrlib.core</code> (generic infrastructure),
  <code>hdrlib.sar</code> (SAR change detection), and <code>hdrlib.sonar</code> (sonar
  target detection). All public classes and functions are documented with type
  signatures and NumPy-style docstrings.</p>
</div>

<div class="api-section">
  <div class="api-section-head">
    <span class="api-section-num">hdrlib.core</span>
    <span class="api-section-title">Generic infrastructure shared across modalities</span>
  </div>
  <div class="api-index">

  <a class="api-card" href="core/backend/">
    <div class="api-card-head">
      <span class="api-mod">core.backend</span>
      <span class="api-badge">utility</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Hardware-agnostic dispatch layer. Detects and normalises NumPy,
      CuPy, JAX, and PyTorch backends from a string or enum, and moves data between devices.</p>
    </div>
  </a>

  <a class="api-card" href="core/estimation/">
    <div class="api-card-head">
      <span class="api-mod">core.estimation</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Fixed-point M-estimators: Tyler, Huber, Student-t, SCM, and
      natural-gradient scaled-Gaussian. Backend-agnostic; supports batched inputs.</p>
    </div>
  </a>

  <a class="api-card" href="core/detection/">
    <div class="api-card-head">
      <span class="api-mod">core.detection</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Abstract base classes <code>Detector</code> and
      <code>OnlineDetector</code>. Subclass to implement new GLRT-based decision statistics.</p>
    </div>
  </a>

  <a class="api-card" href="core/manifolds/">
    <div class="api-card-head">
      <span class="api-mod">core.manifolds</span>
      <span class="api-badge">geometry</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Riemannian manifold primitives: matrix square roots, logarithms,
      geodesic distances, and tangent-space projections for HPD, SHPD, and product manifolds.</p>
    </div>
  </a>

  <a class="api-card" href="core/simulation/">
    <div class="api-card-head">
      <span class="api-mod">core.simulation</span>
      <span class="api-badge">experiment</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Generic data generators: Gaussian and DCG distributions under
      H₀ and H₁, log-spaced T-vectors, and true covariance constructors.</p>
    </div>
  </a>

  <a class="api-card" href="core/mc/">
    <div class="api-card-head">
      <span class="api-mod">core.mc</span>
      <span class="api-badge">experiment</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">MC infrastructure shared across all experiment scripts:
      <code>MCResultExporter</code>, CLI argument helpers, chunked trial ranges,
      and backend warm-up utilities.</p>
    </div>
  </a>

  <a class="api-card" href="core/exporter/">
    <div class="api-card-head">
      <span class="api-mod">core.exporter</span>
      <span class="api-badge">experiment</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Structured result export with git provenance. Saves NumPy
      arrays alongside run metadata — parameters, timestamp, and commit SHA.</p>
    </div>
  </a>

  <a class="api-card" href="core/plot_style/">
    <div class="api-card-head">
      <span class="api-mod">core.plot_style</span>
      <span class="api-badge">style</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Dark serif matplotlib theme. Call <code>apply_style()</code>
      once per process to configure rcParams globally.</p>
    </div>
  </a>

  <a class="api-card" href="core/plotly_style/">
    <div class="api-card-head">
      <span class="api-mod">core.plotly_style</span>
      <span class="api-badge">style</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Plotly design tokens for the docs design system.
      Colour constants, font stacks, and <code>hex_to_rgba()</code>.</p>
    </div>
  </a>

  </div>
</div>

<div class="api-section">
  <div class="api-section-head">
    <span class="api-section-num">hdrlib.sar</span>
    <span class="api-section-title">SAR change detection under elliptical distributions</span>
  </div>
  <div class="api-index">

  <a class="api-card" href="sar/detectors/">
    <div class="api-card-head">
      <span class="api-mod">sar.detectors</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Offline and online GLRT detectors: Gaussian, DCG, and
      Kronecker-structured models. Implements <code>GaussianGLRT</code>,
      <code>DeterministicCompoundGaussianGLRT</code>, <code>ScaleAndShapeKroneckerGLRT</code>,
      and their online counterparts.</p>
    </div>
  </a>

  <a class="api-card" href="sar/estimation_kronecker/">
    <div class="api-card-head">
      <span class="api-mod">sar.estimation_kronecker</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">MM algorithms for Kronecker-structured scatter estimation under
      H₀ and H₁. Riemannian gradient descent with Armijo line-search.</p>
    </div>
  </a>

  <a class="api-card" href="sar/estimation_online/">
    <div class="api-card-head">
      <span class="api-mod">sar.estimation_online</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Online (sample-by-sample) estimators for streaming SAR data.
      Natural-gradient updates on the HPD manifold and an online Kronecker estimator.</p>
    </div>
  </a>

  <a class="api-card" href="sar/detection_online/">
    <div class="api-card-head">
      <span class="api-mod">sar.detection_online</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Online detector implementations: <code>OnlineGaussianGLRT</code>,
      <code>OnlineDCGDetector</code>, and <code>OnlineKroneckerDetector</code>
      with recursive state updates.</p>
    </div>
  </a>

  <a class="api-card" href="sar/simulation/">
    <div class="api-card-head">
      <span class="api-mod">sar.simulation</span>
      <span class="api-badge">experiment</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">SAR-specific data generators: H₁ change-point data for Gaussian,
      DCG, and Kronecker models; Kronecker ground-truth covariance construction.</p>
    </div>
  </a>

  <a class="api-card" href="sar/mc/">
    <div class="api-card-head">
      <span class="api-mod">sar.mc</span>
      <span class="api-badge">experiment</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">SAR MC aggregation: online/offline convergence statistics,
      H₁ power curves, plot templates, and the <code>finish_h0</code> / <code>finish_h1</code>
      export helpers.</p>
    </div>
  </a>

  </div>
</div>

<div class="api-section">
  <div class="api-section-head">
    <span class="api-section-num">hdrlib.sonar</span>
    <span class="api-section-title">Sonar target detection under MSG noise</span>
  </div>
  <div class="api-index">

  <a class="api-card" href="sonar/detectors/">
    <div class="api-card-head">
      <span class="api-mod">sonar.detectors</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Known-M and adaptive detectors for the two-array sonar model:
      M-NMF-G/R/I, NMF single-array, MIMO matched filter, and adaptive variants
      using SCM or 2TYL covariance estimates.</p>
    </div>
  </a>

  <a class="api-card" href="sonar/estimation/">
    <div class="api-card-head">
      <span class="api-mod">sonar.estimation</span>
      <span class="api-badge">core</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Two-array Tyler MLE (2TYL): fixed-point estimator for MSG
      covariance with per-array texture parameters τ₁, τ₂. Trace-normalised and
      convergence-checked at each iteration.</p>
    </div>
  </a>

  <a class="api-card" href="sonar/simulation/">
    <div class="api-card-head">
      <span class="api-mod">sonar.simulation</span>
      <span class="api-badge">experiment</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Two-array sonar data model: Toeplitz cross-covariance,
      far-field steering matrix, H₀/H₁ primary data generation, secondary data,
      and SNR sweep helpers.</p>
    </div>
  </a>

  <a class="api-card" href="sonar/mc/">
    <div class="api-card-head">
      <span class="api-mod">sonar.mc</span>
      <span class="api-badge">experiment</span>
    </div>
    <div class="api-card-body">
      <p class="api-desc">Sonar MC aggregation: PD-vs-SNR curves, PFA-vs-threshold,
      PD angle maps, and 2TYL convergence diagnostics. Standalone plot-script
      templates for all four experiment types.</p>
    </div>
  </a>

  </div>
</div>
