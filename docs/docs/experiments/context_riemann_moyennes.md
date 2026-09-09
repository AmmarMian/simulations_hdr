<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_riemann_moyennes</span>
</nav>

# context_riemann_moyennes

Arithmetic, log-Euclidean and Fréchet means of a cloud of covariance matrices

**Tags:** `context`  `riemann`  `geometry`  `illustration`

## Run

```sh
uv run python 1-context/riemann_moyennes/main.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--n_matrices 15</code><br>
  <code>--dispersion 0.9</code><br>
  <code>--condition 4.0</code><br>
  <code>--radius 1.0</code><br>
  <code>--iter_max 100</code><br>
  <code>--tol 1e-10</code><br>
  <code>--axis_width 0.45\textwidth</code><br>
  <code>--axis_height 4.6cm</code><br>
  <code>--backend numpy</code><br>
  <code>--seed 42</code><br>
  <span class="mn-date">0264bed · 2026-08-27</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/riemann_moyennes/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">314 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/riemann_moyennes/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Three means of the same set of covariance matrices</span>
<span class="c1">#</span>
<span class="c1"># A cloud of 2x2 matrices is drawn around a common centre, then averaged in</span>
<span class="c1"># three ways: arithmetically, log-Euclidean-wise, and by the Fréchet mean of</span>
<span class="c1"># the affine-invariant metric. The left panel shows the ellipses, the right</span>
<span class="c1"># one the determinants, which is where the three differ most clearly.</span>
<span class="c1">#</span>
<span class="c1"># The cloud is generated *on the manifold* rather than by perturbing the</span>
<span class="c1"># entries: each matrix is the Riemannian exponential of a random tangent</span>
<span class="c1"># vector at the centre, so the set is symmetric around it in the geometry the</span>
<span class="c1"># Fréchet mean uses, and the centre is by construction what that mean should</span>
<span class="c1"># recover.</span>
<span class="c1">#</span>
<span class="c1"># The Fréchet mean has no closed form for more than two matrices and is</span>
<span class="c1"># computed by the hand-written Riemannian descent of</span>
<span class="c1"># hdrlib.core.estimation.frechet_mean_affine_invariant.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="n">frechet_mean_affine_invariant</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.manifolds</span><span class="w"> </span><span class="kn">import</span> <span class="n">HermitianPositiveDefinite</span><span class="p">,</span> <span class="n">logm_psd</span><span class="p">,</span> <span class="n">multiherm</span>


<span class="k">def</span><span class="w"> </span><span class="nf">sample_cloud</span><span class="p">(</span><span class="n">center</span><span class="p">,</span> <span class="n">dispersion</span><span class="p">,</span> <span class="n">n_matrices</span><span class="p">,</span> <span class="n">manifold</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">seed</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Matrices spread around a centre along random geodesics.</span>

<span class="sd">    Draws a symmetric tangent vector of unit Riemannian norm at the centre,</span>
<span class="sd">    scales it by a random length of standard deviation ``dispersion``, and</span>
<span class="sd">    follows the geodesic. The result is a cloud whose spread is measured in</span>
<span class="sd">    the affine-invariant metric and not in the entries of the matrices.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span><span class="n">seed</span><span class="p">)</span>
    <span class="n">center_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">center</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">n_features</span> <span class="o">=</span> <span class="n">center</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>

    <span class="n">cloud</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n_matrices</span><span class="p">):</span>
        <span class="n">noise</span> <span class="o">=</span> <span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">n_features</span><span class="p">,</span> <span class="n">n_features</span><span class="p">))</span>
        <span class="n">tangent</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span>
            <span class="n">multiherm</span><span class="p">(</span><span class="n">get_data_on_device</span><span class="p">(</span><span class="n">noise</span><span class="p">,</span> <span class="n">backend</span><span class="p">),</span> <span class="n">backend</span><span class="p">)</span>
        <span class="p">)</span>
        <span class="n">tangent_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">tangent</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
        <span class="n">norm</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span><span class="n">manifold</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">center_device</span><span class="p">,</span> <span class="n">tangent_device</span><span class="p">))</span>
        <span class="n">length</span> <span class="o">=</span> <span class="n">dispersion</span> <span class="o">*</span> <span class="nb">abs</span><span class="p">(</span><span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">())</span>
        <span class="n">tangent_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span>
            <span class="n">length</span> <span class="o">*</span> <span class="n">tangent</span> <span class="o">/</span> <span class="n">norm</span><span class="p">,</span> <span class="n">backend</span>
        <span class="p">)</span>
        <span class="n">cloud</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">to_numpy</span><span class="p">(</span><span class="n">manifold</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">center_device</span><span class="p">,</span> <span class="n">tangent_device</span><span class="p">)))</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">cloud</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">log_euclidean_mean</span><span class="p">(</span><span class="n">covariances</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;expm of the arithmetic mean of the logarithms — the closed form.&quot;&quot;&quot;</span>
    <span class="n">logarithms</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span>
        <span class="n">logm_psd</span><span class="p">(</span><span class="n">get_data_on_device</span><span class="p">(</span><span class="n">covariances</span><span class="p">,</span> <span class="n">backend</span><span class="p">),</span> <span class="n">backend</span><span class="p">)</span>
    <span class="p">)</span>
    <span class="n">values</span><span class="p">,</span> <span class="n">vectors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">(</span><span class="n">logarithms</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">vectors</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">values</span><span class="p">))</span> <span class="o">@</span> <span class="n">vectors</span><span class="o">.</span><span class="n">T</span>


<span class="k">def</span><span class="w"> </span><span class="nf">harmonic_mean</span><span class="p">(</span><span class="n">covariances</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Inverse of the arithmetic mean of the inverses — the closed form.</span>

<span class="sd">    The Fréchet mean of the right Kullback-Leibler divergence, i.e. the maximum</span>
<span class="sd">    likelihood estimate under an inverse-Wishart model</span>
<span class="sd">    (prop:spdnet-moyennes-frechet).</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">inv</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">inv</span><span class="p">(</span><span class="n">covariances</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>


<span class="k">def</span><span class="w"> </span><span class="nf">gah_mean</span><span class="p">(</span><span class="n">covariances</span><span class="p">,</span> <span class="n">manifold</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Midpoint of the geodesic between the arithmetic and harmonic means.</span>

<span class="sd">    The Fréchet mean of the *symmetrised* Kullback-Leibler divergence. It is the</span>
<span class="sd">    only closed-form mean of prop:spdnet-moyennes-frechet that keeps both the</span>
<span class="sd">    congruence and the inversion invariance, which is why the batch-norm work of</span>
<span class="sd">    ch:spdnet ends up preferring it to the geometric mean.</span>

<span class="sd">    Computed as the midpoint of the affine-invariant geodesic rather than by the</span>
<span class="sd">    usual closed formula, so that what is drawn is the definition itself.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">arithmetic</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">covariances</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">),</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">harmonic</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">harmonic_mean</span><span class="p">(</span><span class="n">covariances</span><span class="p">),</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">tangent</span> <span class="o">=</span> <span class="n">manifold</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">arithmetic</span><span class="p">,</span> <span class="n">harmonic</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">manifold</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">arithmetic</span><span class="p">,</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="n">tangent</span><span class="p">))</span>


<span class="k">def</span><span class="w"> </span><span class="nf">concentration_ellipse</span><span class="p">(</span><span class="n">shape</span><span class="p">,</span> <span class="n">radius</span><span class="p">,</span> <span class="n">n_points</span><span class="o">=</span><span class="mi">200</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Curve {x : x^T shape^{-1} x = radius^2}, as a (2, n_points) array.&quot;&quot;&quot;</span>
    <span class="n">angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="p">,</span> <span class="n">n_points</span><span class="p">)</span>
    <span class="n">circle</span> <span class="o">=</span> <span class="n">radius</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">angles</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">angles</span><span class="p">)])</span>
    <span class="n">values</span><span class="p">,</span> <span class="n">vectors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">(</span><span class="n">shape</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">vectors</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">values</span><span class="p">))</span> <span class="o">@</span> <span class="n">circle</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Arithmetic, log-Euclidean and Fréchet means of a cloud of covariances.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_matrices&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of matrices averaged. Kept small enough for the cloud to &quot;</span>
             <span class="s2">&quot;remain readable as a set of ellipses.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dispersion&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Standard deviation of the geodesic distance between a matrix of &quot;</span>
             <span class="s2">&quot;the cloud and its centre, in the affine-invariant metric.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--condition&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">4.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Ratio of the eigenvalues of the centre of the cloud.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--radius&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Radius of the drawn ellipses, in units of the Mahalanobis distance.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--iter_max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Maximum number of Riemannian gradient steps for the Fréchet mean.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-10</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Stopping tolerance on the gradient norm of the Fréchet mean.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/riemann_moyennes&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Output directory for LaTeX exports (injected by qanat, or set manually).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--show-interactive&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Show plots interactively with matplotlib.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--export&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="n">argparse</span><span class="o">.</span><span class="n">BooleanOptionalAction</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Save TikZ/PGFPlots figure (.tex) (default: True).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_width&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.45</span><span class="se">\\</span><span class="s2">textwidth&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of a single panel in the exported PGFPlots figure. Set &quot;</span>
             <span class="s2">&quot;here rather than patched into the .tex afterwards, so that a &quot;</span>
             <span class="s2">&quot;re-sync into the dissertation does not undo it.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_height&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;4.6cm&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Height of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--backend&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;numpy&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Compute backend (numpy, torch-cpu, torch-mps, ...).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;random seed generation base seed&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="c1"># Constant(s)</span>
    <span class="n">d</span> <span class="o">=</span> <span class="mi">2</span>

    <span class="n">manifold</span> <span class="o">=</span> <span class="n">HermitianPositiveDefinite</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">spread</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">)</span>
    <span class="n">center</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">([</span><span class="n">spread</span><span class="p">,</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="n">spread</span><span class="p">])</span>
    <span class="n">cloud</span> <span class="o">=</span> <span class="n">sample_cloud</span><span class="p">(</span>
        <span class="n">center</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">dispersion</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_matrices</span><span class="p">,</span> <span class="n">manifold</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">seed</span>
    <span class="p">)</span>

    <span class="n">frechet</span><span class="p">,</span> <span class="n">history</span> <span class="o">=</span> <span class="n">frechet_mean_affine_invariant</span><span class="p">(</span>
        <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">cloud</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
        <span class="n">iter_max</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">tol</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="c1"># Insertion order is drawing order, and the dashed log-Euclidean mean is</span>
    <span class="c1"># kept last so that it stays visible where it lands on the Fréchet one.</span>
    <span class="c1">#</span>
    <span class="c1"># The harmonic and GAH means are here for ch:spdnet</span>
    <span class="c1"># (prop:spdnet-moyennes-frechet), which needs the five of them side by side:</span>
    <span class="c1"># arithmetic and harmonic bracket the cloud from either side — left and</span>
    <span class="c1"># right Kullback-Leibler, Wishart and inverse-Wishart — and GAH, their</span>
    <span class="c1"># symmetrised compromise, lands next to the Fréchet mean without any</span>
    <span class="c1"># iteration. That is the whole argument of sec:spdnet-batchnorm-moyennes,</span>
    <span class="c1"># in one picture.</span>
    <span class="n">means</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;arithmétique&quot;</span><span class="p">:</span> <span class="n">cloud</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">),</span>
        <span class="s2">&quot;harmonique&quot;</span><span class="p">:</span> <span class="n">harmonic_mean</span><span class="p">(</span><span class="n">cloud</span><span class="p">),</span>
        <span class="s2">&quot;de Fréchet&quot;</span><span class="p">:</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">frechet</span><span class="p">),</span>
        <span class="s2">&quot;</span><span class="se">\\</span><span class="s2">textsc</span><span class="si">{gah}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">gah_mean</span><span class="p">(</span><span class="n">cloud</span><span class="p">,</span> <span class="n">manifold</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
        <span class="s2">&quot;log-euclidienne&quot;</span><span class="p">:</span> <span class="n">log_euclidean_mean</span><span class="p">(</span><span class="n">cloud</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
    <span class="p">}</span>

    <span class="n">colors</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;arithmétique&quot;</span><span class="p">:</span> <span class="s2">&quot;C1&quot;</span><span class="p">,</span>
        <span class="s2">&quot;harmonique&quot;</span><span class="p">:</span> <span class="s2">&quot;C4&quot;</span><span class="p">,</span>
        <span class="s2">&quot;log-euclidienne&quot;</span><span class="p">:</span> <span class="s2">&quot;C3&quot;</span><span class="p">,</span>
        <span class="s2">&quot;de Fréchet&quot;</span><span class="p">:</span> <span class="s2">&quot;C2&quot;</span><span class="p">,</span>
        <span class="s2">&quot;</span><span class="se">\\</span><span class="s2">textsc</span><span class="si">{gah}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="s2">&quot;C5&quot;</span><span class="p">,</span>
    <span class="p">}</span>
    <span class="c1"># The log-Euclidean, GAH and Fréchet means are close enough to overlap on</span>
    <span class="c1"># both panels, which is itself worth seeing: the ones drawn on top are</span>
    <span class="c1"># dashed so that a superposition reads as a superposition and not as a</span>
    <span class="c1"># missing curve.</span>
    <span class="n">linestyles</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;arithmétique&quot;</span><span class="p">:</span> <span class="s2">&quot;-&quot;</span><span class="p">,</span>
        <span class="s2">&quot;harmonique&quot;</span><span class="p">:</span> <span class="s2">&quot;-&quot;</span><span class="p">,</span>
        <span class="s2">&quot;log-euclidienne&quot;</span><span class="p">:</span> <span class="s2">&quot;--&quot;</span><span class="p">,</span>
        <span class="s2">&quot;de Fréchet&quot;</span><span class="p">:</span> <span class="s2">&quot;-&quot;</span><span class="p">,</span>
        <span class="s2">&quot;</span><span class="se">\\</span><span class="s2">textsc</span><span class="si">{gah}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="s2">&quot;--&quot;</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">))</span>

    <span class="n">ax</span> <span class="o">=</span> <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">cloud</span><span class="p">):</span>
        <span class="n">curve</span> <span class="o">=</span> <span class="n">concentration_ellipse</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">radius</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">curve</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">curve</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C0&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="s2">&quot;échantillon&quot;</span> <span class="k">if</span> <span class="n">index</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">means</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">curve</span> <span class="o">=</span> <span class="n">concentration_ellipse</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">radius</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">curve</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">curve</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.6</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">linestyle</span><span class="o">=</span><span class="n">linestyles</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="n">name</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="s2">&quot;equal&quot;</span><span class="p">)</span>
    <span class="c1"># Framed on a high quantile rather than the maximum: one very elongated</span>
    <span class="c1"># draw would otherwise shrink everything else to a dot. It is still drawn,</span>
    <span class="c1"># simply not entirely inside the frame.</span>
    <span class="n">limit</span> <span class="o">=</span> <span class="mf">1.1</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">quantile</span><span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span>
            <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span>
                <span class="p">[</span><span class="n">concentration_ellipse</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">radius</span><span class="p">)</span> <span class="k">for</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">cloud</span><span class="p">]</span>
            <span class="p">)</span>
        <span class="p">),</span>
        <span class="mf">0.99</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_1$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_2$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s2">&quot;ellipses de concentration&quot;</span><span class="p">)</span>

    <span class="c1"># Determinants, sorted, with the three means as horizontal lines: the</span>
    <span class="c1"># arithmetic one sits above the cloud, the two others inside it.</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">determinants</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">matrix</span><span class="p">)</span> <span class="k">for</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">cloud</span><span class="p">])</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_matrices</span> <span class="o">+</span> <span class="mi">1</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">determinants</span><span class="p">),</span>
        <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C0&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">means</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">axhline</span><span class="p">(</span>
            <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">matrix</span><span class="p">),</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.4</span><span class="p">,</span>
            <span class="n">linestyle</span><span class="o">=</span><span class="n">linestyles</span><span class="p">[</span><span class="n">name</span><span class="p">],</span>
        <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;matrices triées&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\det$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s2">&quot;déterminants&quot;</span><span class="p">)</span>
    <span class="c1"># Legend below the panels rather than inside one of them: exported at this</span>
    <span class="c1"># size, an inner legend either covers the data or spills over the frame.</span>
    <span class="c1"># It is attached to an axis and not to the figure, since matplot2tikz</span>
    <span class="c1"># exports axis legends and silently drops figure ones.</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;upper left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.45</span><span class="p">),</span> <span class="n">ncol</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
        <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;N = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_matrices</span><span class="si">}</span><span class="s2"> matrices, dispersion </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">dispersion</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;Fréchet mean in </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">history</span><span class="p">[</span><span class="s1">&#39;variance&#39;</span><span class="p">])</span><span class="w"> </span><span class="o">-</span><span class="w"> </span><span class="mi">1</span><span class="si">}</span><span class="s2"> iterations &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;(gradient norm </span><span class="si">{</span><span class="n">history</span><span class="p">[</span><span class="s1">&#39;gradient_norm&#39;</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">:</span><span class="s2">.2e</span><span class="si">}</span><span class="s2">)&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  geometric mean of the determinants: &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">determinants</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">())</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">means</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">distance</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span>
            <span class="n">manifold</span><span class="o">.</span><span class="n">dist</span><span class="p">(</span>
                <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
                <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">center</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
            <span class="p">)</span>
        <span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  </span><span class="si">{</span><span class="n">name</span><span class="si">:</span><span class="s2">16</span><span class="si">}</span><span class="s2"> det = </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">matrix</span><span class="p">)</span><span class="si">:</span><span class="s2">7.3f</span><span class="si">}</span><span class="s2">   &quot;</span>
              <span class="sa">f</span><span class="s2">&quot;distance to the centre = </span><span class="si">{</span><span class="n">distance</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_matrices</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_matrices</span><span class="p">,</span> <span class="n">dispersion</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dispersion</span><span class="p">,</span>
        <span class="n">condition</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">radius</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">radius</span><span class="p">,</span>
        <span class="n">center</span><span class="o">=</span><span class="n">center</span><span class="p">,</span> <span class="n">cloud</span><span class="o">=</span><span class="n">cloud</span><span class="p">,</span>
        <span class="n">variance</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">history</span><span class="p">[</span><span class="s2">&quot;variance&quot;</span><span class="p">]),</span>
        <span class="n">gradient_norm</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">history</span><span class="p">[</span><span class="s2">&quot;gradient_norm&quot;</span><span class="p">]),</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;mean_</span><span class="si">{</span><span class="n">name</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;-&#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;é&#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;e&#39;</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">matrix</span>
           <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">means</span><span class="o">.</span><span class="n">items</span><span class="p">()},</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;moyennes.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved means in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_matrices</span><span class="param-type">int</span><span class="param-default">default <b>15</b></span>
</div>
<p class="param-help">Number of matrices averaged. Kept small enough for the cloud to remain readable as a set of ellipses.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dispersion</span><span class="param-type">float</span><span class="param-default">default <b>0.9</b></span>
</div>
<p class="param-help">Standard deviation of the geodesic distance between a matrix of the cloud and its centre, in the affine-invariant metric.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--condition</span><span class="param-type">float</span><span class="param-default">default <b>4.0</b></span>
</div>
<p class="param-help">Ratio of the eigenvalues of the centre of the cloud.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--radius</span><span class="param-type">float</span><span class="param-default">default <b>1.0</b></span>
</div>
<p class="param-help">Radius of the drawn ellipses, in units of the Mahalanobis distance.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--iter_max</span><span class="param-type">int</span><span class="param-default">default <b>100</b></span>
</div>
<p class="param-help">Maximum number of Riemannian gradient steps for the Fréchet mean.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-10</b></span>
</div>
<p class="param-help">Stopping tolerance on the gradient norm of the Fréchet mean.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/riemann_moyennes</b></span>
</div>
<p class="param-help">Output directory for LaTeX exports (injected by qanat, or set manually).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--show-interactive</span><span class="param-type">flag</span>
</div>
<p class="param-help">Show plots interactively with matplotlib.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--export</span><span class="param-default">default <b>True</b></span>
</div>
<p class="param-help">Save TikZ/PGFPlots figure (.tex) (default: True).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_width</span><span class="param-type">str</span><span class="param-default">default <b>0.45\textwidth</b></span>
</div>
<p class="param-help">Width of a single panel in the exported PGFPlots figure. Set here rather than patched into the .tex afterwards, so that a re-sync into the dissertation does not undo it.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_height</span><span class="param-type">str</span><span class="param-default">default <b>4.6cm</b></span>
</div>
<p class="param-help">Height of a single panel in the exported PGFPlots figure.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--backend</span><span class="param-type">str</span><span class="param-default">default <b>numpy</b></span>
</div>
<p class="param-help">Compute backend (numpy, torch-cpu, torch-mps, ...).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">random seed generation base seed</p>
</div>
</div>

## Results

<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_moyennes.json" data-title="context_riemann_moyennes"></div>
</div>

## Config

`1-context/experiments/context_riemann_moyennes.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
