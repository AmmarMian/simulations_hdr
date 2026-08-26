<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_riemann_interpolation</span>
</nav>

# context_riemann_interpolation

Euclidean, affine-invariant and log-Euclidean paths between two covariance matrices

**Tags:** `context`  `riemann`  `geometry`  `illustration`

## Run

```sh
uv run python 1-context/riemann_interpolation/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/riemann_interpolation/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">260 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/riemann_interpolation/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Three ways of going from one covariance matrix to another</span>
<span class="c1">#</span>
<span class="c1"># The same two 2x2 matrices are joined by the Euclidean segment, by the</span>
<span class="c1"># affine-invariant geodesic and by the log-Euclidean geodesic. Each path is</span>
<span class="c1"># drawn as a family of concentration ellipses, and the last panel follows the</span>
<span class="c1"># determinant along the three of them.</span>
<span class="c1">#</span>
<span class="c1"># The two endpoints are deliberately taken with the same determinant and</span>
<span class="c1"># orthogonal principal directions, which is the configuration where the</span>
<span class="c1"># Euclidean average of two ellipses is visibly larger than either of them:</span>
<span class="c1"># the determinant bulges in the middle. The two Riemannian paths interpolate</span>
<span class="c1"># the determinant geometrically instead, so nothing is created along the way.</span>
<span class="c1">#</span>
<span class="c1"># The geodesic and the exponential come from hdrlib.core.manifolds unchanged;</span>
<span class="c1"># the log-Euclidean path is the one closed form the chapter gives explicitly,</span>
<span class="c1"># so it is written out here rather than hidden behind a manifold object.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.manifolds</span><span class="w"> </span><span class="kn">import</span> <span class="n">HermitianPositiveDefinite</span><span class="p">,</span> <span class="n">logm_psd</span>


<span class="k">def</span><span class="w"> </span><span class="nf">rotation</span><span class="p">(</span><span class="n">angle</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Plane rotation of the given angle, in radians.&quot;&quot;&quot;</span>
    <span class="n">cosine</span><span class="p">,</span> <span class="n">sine</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">angle</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">angle</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="n">cosine</span><span class="p">,</span> <span class="o">-</span><span class="n">sine</span><span class="p">],</span> <span class="p">[</span><span class="n">sine</span><span class="p">,</span> <span class="n">cosine</span><span class="p">]])</span>


<span class="k">def</span><span class="w"> </span><span class="nf">endpoints</span><span class="p">(</span><span class="n">condition</span><span class="p">,</span> <span class="n">angle</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Two matrices of unit determinant, of the same shape but rotated.</span>

<span class="sd">    Both have eigenvalues sqrt(condition) and 1/sqrt(condition), so the</span>
<span class="sd">    Euclidean path between them cannot change the eigenvalues without</span>
<span class="sd">    changing the determinant — which is exactly what it does.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">spread</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">condition</span><span class="p">)</span>
    <span class="n">diagonal</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">([</span><span class="n">spread</span><span class="p">,</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="n">spread</span><span class="p">])</span>
    <span class="n">rotated</span> <span class="o">=</span> <span class="n">rotation</span><span class="p">(</span><span class="n">angle</span><span class="p">)</span> <span class="o">@</span> <span class="n">diagonal</span> <span class="o">@</span> <span class="n">rotation</span><span class="p">(</span><span class="n">angle</span><span class="p">)</span><span class="o">.</span><span class="n">T</span>
    <span class="k">return</span> <span class="n">diagonal</span><span class="p">,</span> <span class="n">rotated</span>


<span class="k">def</span><span class="w"> </span><span class="nf">euclidean_path</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">times</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;The straight segment (1-t) A + t B, which stays in the cone.&quot;&quot;&quot;</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([(</span><span class="mf">1.0</span> <span class="o">-</span> <span class="n">t</span><span class="p">)</span> <span class="o">*</span> <span class="n">start</span> <span class="o">+</span> <span class="n">t</span> <span class="o">*</span> <span class="n">end</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">times</span><span class="p">])</span>


<span class="k">def</span><span class="w"> </span><span class="nf">affine_invariant_path</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">times</span><span class="p">,</span> <span class="n">manifold</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;The geodesic A #_t B, obtained as exp_A(t log_A(B)).&quot;&quot;&quot;</span>
    <span class="n">start_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">end_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">end</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">direction</span> <span class="o">=</span> <span class="n">manifold</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">start_device</span><span class="p">,</span> <span class="n">end_device</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span>
        <span class="p">[</span><span class="n">to_numpy</span><span class="p">(</span><span class="n">manifold</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">start_device</span><span class="p">,</span> <span class="n">t</span> <span class="o">*</span> <span class="n">direction</span><span class="p">))</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">times</span><span class="p">]</span>
    <span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">log_euclidean_path</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">times</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;expm((1-t) logm A + t logm B): the segment read in logarithm coordinates.&quot;&quot;&quot;</span>
    <span class="n">log_start</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">logm_psd</span><span class="p">(</span><span class="n">get_data_on_device</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">backend</span><span class="p">),</span> <span class="n">backend</span><span class="p">))</span>
    <span class="n">log_end</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">logm_psd</span><span class="p">(</span><span class="n">get_data_on_device</span><span class="p">(</span><span class="n">end</span><span class="p">,</span> <span class="n">backend</span><span class="p">),</span> <span class="n">backend</span><span class="p">))</span>
    <span class="n">path</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">times</span><span class="p">:</span>
        <span class="n">values</span><span class="p">,</span> <span class="n">vectors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">((</span><span class="mf">1.0</span> <span class="o">-</span> <span class="n">t</span><span class="p">)</span> <span class="o">*</span> <span class="n">log_start</span> <span class="o">+</span> <span class="n">t</span> <span class="o">*</span> <span class="n">log_end</span><span class="p">)</span>
        <span class="n">path</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">vectors</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">values</span><span class="p">))</span> <span class="o">@</span> <span class="n">vectors</span><span class="o">.</span><span class="n">T</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">path</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">concentration_ellipse</span><span class="p">(</span><span class="n">shape</span><span class="p">,</span> <span class="n">radius</span><span class="p">,</span> <span class="n">n_points</span><span class="o">=</span><span class="mi">200</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Curve {x : x^T shape^{-1} x = radius^2}, as a (2, n_points) array.&quot;&quot;&quot;</span>
    <span class="n">angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="p">,</span> <span class="n">n_points</span><span class="p">)</span>
    <span class="n">circle</span> <span class="o">=</span> <span class="n">radius</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">angles</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">angles</span><span class="p">)])</span>
    <span class="n">values</span><span class="p">,</span> <span class="n">vectors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">(</span><span class="n">shape</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">vectors</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">values</span><span class="p">))</span> <span class="o">@</span> <span class="n">circle</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Euclidean, affine-invariant and log-Euclidean paths between two covariances.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--condition&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">16.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Ratio of the two eigenvalues of each endpoint. The larger it is, &quot;</span>
             <span class="s2">&quot;the more elongated the ellipses and the more visible the &quot;</span>
             <span class="s2">&quot;swelling of the Euclidean path.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--angle&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.35</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Angle between the principal directions of the two endpoints, in &quot;</span>
             <span class="s2">&quot;units of pi. A quarter turn, 0.5, is the worst case for the &quot;</span>
             <span class="s2">&quot;Euclidean path, but it makes the two endpoints commute, and the &quot;</span>
             <span class="s2">&quot;two Riemannian paths then coincide exactly; a value away from it &quot;</span>
             <span class="s2">&quot;keeps the swelling and separates them.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_steps&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">7</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of ellipses drawn along each path, endpoints included.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--radius&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Radius of the drawn ellipses, in units of the Mahalanobis distance.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/riemann_interpolation&quot;</span><span class="p">,</span>
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
    <span class="n">start</span><span class="p">,</span> <span class="n">end</span> <span class="o">=</span> <span class="n">endpoints</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">angle</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="p">)</span>
    <span class="n">times</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_steps</span><span class="p">)</span>

    <span class="n">paths</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;euclidienne&quot;</span><span class="p">:</span> <span class="n">euclidean_path</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">times</span><span class="p">),</span>
        <span class="s2">&quot;affine invariante&quot;</span><span class="p">:</span> <span class="n">affine_invariant_path</span><span class="p">(</span>
            <span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">times</span><span class="p">,</span> <span class="n">manifold</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span>
        <span class="p">),</span>
        <span class="s2">&quot;log-euclidienne&quot;</span><span class="p">:</span> <span class="n">log_euclidean_path</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">times</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
    <span class="p">}</span>
    <span class="n">determinants</span> <span class="o">=</span> <span class="p">{</span>
        <span class="n">name</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">matrix</span><span class="p">)</span> <span class="k">for</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">path</span><span class="p">])</span>
        <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">path</span> <span class="ow">in</span> <span class="n">paths</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
    <span class="p">}</span>

    <span class="n">colors</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;euclidienne&quot;</span><span class="p">:</span> <span class="s2">&quot;C1&quot;</span><span class="p">,</span>
        <span class="s2">&quot;affine invariante&quot;</span><span class="p">:</span> <span class="s2">&quot;C2&quot;</span><span class="p">,</span>
        <span class="s2">&quot;log-euclidienne&quot;</span><span class="p">:</span> <span class="s2">&quot;C3&quot;</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">))</span>
    <span class="n">axes</span> <span class="o">=</span> <span class="n">axes</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>

    <span class="n">limit</span> <span class="o">=</span> <span class="mf">1.15</span> <span class="o">*</span> <span class="nb">max</span><span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">concentration_ellipse</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">radius</span><span class="p">))</span><span class="o">.</span><span class="n">max</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">path</span> <span class="ow">in</span> <span class="n">paths</span><span class="o">.</span><span class="n">values</span><span class="p">()</span> <span class="k">for</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">path</span>
    <span class="p">)</span>

    <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">path</span><span class="p">))</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">paths</span><span class="o">.</span><span class="n">items</span><span class="p">())):</span>
        <span class="k">for</span> <span class="n">step</span><span class="p">,</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">path</span><span class="p">):</span>
            <span class="n">curve</span> <span class="o">=</span> <span class="n">concentration_ellipse</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">radius</span><span class="p">)</span>
            <span class="c1"># The endpoints are shared by the three panels and drawn alike, so</span>
            <span class="c1"># that only what happens between them distinguishes the metrics.</span>
            <span class="n">endpoint</span> <span class="o">=</span> <span class="n">step</span> <span class="ow">in</span> <span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">path</span><span class="p">)</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
                <span class="n">curve</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">curve</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span>
                <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C7&quot;</span> <span class="k">if</span> <span class="n">endpoint</span> <span class="k">else</span> <span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span>
                <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span> <span class="k">if</span> <span class="n">endpoint</span> <span class="k">else</span> <span class="s2">&quot;-&quot;</span><span class="p">,</span>
                <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.3</span> <span class="k">if</span> <span class="n">endpoint</span> <span class="k">else</span> <span class="mf">1.0</span><span class="p">,</span>
                <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span> <span class="k">if</span> <span class="n">endpoint</span> <span class="k">else</span> <span class="mi">2</span><span class="p">,</span>
            <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="s2">&quot;equal&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">name</span><span class="p">)</span>
        <span class="c1"># Outer labels only: with four panels stacked in a text block this</span>
        <span class="c1"># narrow, an inner label runs into the title of the panel below it.</span>
        <span class="k">if</span> <span class="n">index</span> <span class="o">&gt;=</span> <span class="mi">2</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_1$&quot;</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">index</span> <span class="o">%</span> <span class="mi">2</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_2$&quot;</span><span class="p">)</span>

    <span class="n">ax</span> <span class="o">=</span> <span class="n">axes</span><span class="p">[</span><span class="mi">3</span><span class="p">]</span>
    <span class="n">fine_times</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">,</span> <span class="mi">101</span><span class="p">)</span>
    <span class="n">fine_paths</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;euclidienne&quot;</span><span class="p">:</span> <span class="n">euclidean_path</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">fine_times</span><span class="p">),</span>
        <span class="s2">&quot;affine invariante&quot;</span><span class="p">:</span> <span class="n">affine_invariant_path</span><span class="p">(</span>
            <span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">fine_times</span><span class="p">,</span> <span class="n">manifold</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span>
        <span class="p">),</span>
        <span class="s2">&quot;log-euclidienne&quot;</span><span class="p">:</span> <span class="n">log_euclidean_path</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="p">,</span> <span class="n">fine_times</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
    <span class="p">}</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">path</span> <span class="ow">in</span> <span class="n">fine_paths</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">fine_times</span><span class="p">,</span>
            <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">matrix</span><span class="p">)</span> <span class="k">for</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">path</span><span class="p">],</span>
            <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.4</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">name</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$t$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\det$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s2">&quot;déterminant le long du chemin&quot;</span><span class="p">)</span>

    <span class="c1"># Legend below the grid rather than inside a panel: the panels are barely</span>
    <span class="c1"># five centimetres wide once exported, and an inner legend either covers</span>
    <span class="c1"># the curves or spills over the frame. It is attached to the panel whose</span>
    <span class="c1"># curves it names, and pushed left of it so that it spans the whole width:</span>
    <span class="c1"># matplot2tikz only exports entries for the labelled curves of the very</span>
    <span class="c1"># axis the legend belongs to.</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;upper left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="o">-</span><span class="mf">1.3</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.45</span><span class="p">),</span> <span class="n">ncol</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
        <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Endpoints of determinant </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">start</span><span class="p">)</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2"> and &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">end</span><span class="p">)</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">, condition number </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">values</span> <span class="ow">in</span> <span class="n">determinants</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  </span><span class="si">{</span><span class="n">name</span><span class="si">:</span><span class="s2">18</span><span class="si">}</span><span class="s2"> det at t=1/2: </span><span class="si">{</span><span class="n">values</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">values</span><span class="p">)</span><span class="w"> </span><span class="o">//</span><span class="w"> </span><span class="mi">2</span><span class="p">]</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span>
              <span class="sa">f</span><span class="s2">&quot;   max: </span><span class="si">{</span><span class="n">values</span><span class="o">.</span><span class="n">max</span><span class="p">()</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">condition</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">angle</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">angle</span><span class="p">,</span>
        <span class="n">n_steps</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_steps</span><span class="p">,</span> <span class="n">radius</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">radius</span><span class="p">,</span>
        <span class="n">times</span><span class="o">=</span><span class="n">times</span><span class="p">,</span> <span class="n">start</span><span class="o">=</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="o">=</span><span class="n">end</span><span class="p">,</span>
        <span class="n">fine_times</span><span class="o">=</span><span class="n">fine_times</span><span class="p">,</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;path_</span><span class="si">{</span><span class="n">name</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;-&#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">path</span>
           <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">path</span> <span class="ow">in</span> <span class="n">paths</span><span class="o">.</span><span class="n">items</span><span class="p">()},</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;det_</span><span class="si">{</span><span class="n">name</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;-&#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span>
           <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">matrix</span><span class="p">)</span> <span class="k">for</span> <span class="n">matrix</span> <span class="ow">in</span> <span class="n">path</span><span class="p">])</span>
           <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">path</span> <span class="ow">in</span> <span class="n">fine_paths</span><span class="o">.</span><span class="n">items</span><span class="p">()},</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;interpolation.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved interpolation paths in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--condition</span><span class="param-type">float</span><span class="param-default">default <b>16.0</b></span>
</div>
<p class="param-help">Ratio of the two eigenvalues of each endpoint. The larger it is, the more elongated the ellipses and the more visible the swelling of the Euclidean path.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--angle</span><span class="param-type">float</span><span class="param-default">default <b>0.35</b></span>
</div>
<p class="param-help">Angle between the principal directions of the two endpoints, in units of pi. A quarter turn, 0.5, is the worst case for the Euclidean path, but it makes the two endpoints commute, and the two Riemannian paths then coincide exactly; a value away from it keeps the swelling and separates them.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_steps</span><span class="param-type">int</span><span class="param-default">default <b>7</b></span>
</div>
<p class="param-help">Number of ellipses drawn along each path, endpoints included.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--radius</span><span class="param-type">float</span><span class="param-default">default <b>1.0</b></span>
</div>
<p class="param-help">Radius of the drawn ellipses, in units of the Mahalanobis distance.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/riemann_interpolation</b></span>
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

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-08-19</span><br>
  <code>--condition</code> <span class='mn-default'>16.0</span><br>
  <code>--angle</code> <span class='mn-default'>0.35</span><br>
  <code>--n_steps</code> <span class='mn-default'>7</span><br>
  <code>--radius</code> <span class='mn-default'>1.0</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--axis_width</code> <span class='mn-default'>0.45\textwidth</span><br>
  <code>--axis_height</code> <span class='mn-default'>4.6cm</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_interpolation.json" data-title="context_riemann_interpolation"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Endpoints of determinant 1.000 and 1.000, condition number 16
  euclidienne        det at t=1/2: 3.791   max: 3.791
  affine invariante  det at t=1/2: 1.000   max: 1.000
  log-euclidienne    det at t=1/2: 1.000   max: 1.000
Saved interpolation paths in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_riemann_interpolation/run_29/interpolation.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_riemann_interpolation.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
