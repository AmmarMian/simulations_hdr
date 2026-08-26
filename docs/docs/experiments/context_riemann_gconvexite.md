<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_riemann_gconvexite</span>
</nav>

# context_riemann_gconvexite

Tyler's cost read along a Euclidean segment and along an affine-invariant geodesic

**Tags:** `context`  `riemann`  `robust`  `illustration`

## Run

```sh
uv run python 1-context/riemann_gconvexite/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/riemann_gconvexite/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">216 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/riemann_gconvexite/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Tyler&#39;s cost along a segment and along a geodesic</span>
<span class="c1">#</span>
<span class="c1"># The same cost is read along the two paths joining the same two points of the</span>
<span class="c1"># cone: the Euclidean segment (1-t) A + t B, which does stay inside the cone</span>
<span class="c1"># since it is convex, and the affine-invariant geodesic A #_t B.</span>
<span class="c1">#</span>
<span class="c1"># The two endpoints are a strongly ill-conditioned matrix and its inverse,</span>
<span class="c1"># which are at equal distance from the identity — the true scatter matrix of</span>
<span class="c1"># the data — so the middle of both paths is where the minimiser should be.</span>
<span class="c1"># It is where the geodesic finds it, and it is a local *maximum* of the</span>
<span class="c1"># Euclidean reading, which therefore shows two spurious minima at its ends.</span>
<span class="c1"># The criterion is the same in both panels: what changes is the notion of</span>
<span class="c1"># straight line along which it is read, and with it the convexity.</span>
<span class="c1">#</span>
<span class="c1"># The cost is hdrlib.core.estimation.tyler_cost, the very function the</span>
<span class="c1"># convergence experiment minimises.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="n">tyler_cost</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.elliptical</span><span class="w"> </span><span class="kn">import</span> <span class="n">StudentTDistribution</span><span class="p">,</span> <span class="n">sample_elliptical</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.manifolds</span><span class="w"> </span><span class="kn">import</span> <span class="n">HermitianPositiveDefinite</span>


<span class="k">def</span><span class="w"> </span><span class="nf">endpoints</span><span class="p">(</span><span class="n">condition</span><span class="p">,</span> <span class="n">rotation</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;A matrix of unit determinant and its inverse, in a common basis.</span>

<span class="sd">    Their eigenvalues span the given condition number symmetrically in</span>
<span class="sd">    logarithm, so the two are exchanged by inversion and their geometric mean</span>
<span class="sd">    — the middle of the geodesic — is the identity.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">n_features</span> <span class="o">=</span> <span class="n">rotation</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">eigenvalues</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">logspace</span><span class="p">(</span>
        <span class="o">-</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">condition</span><span class="p">),</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">condition</span><span class="p">),</span> <span class="n">n_features</span>
    <span class="p">)</span>
    <span class="n">start</span> <span class="o">=</span> <span class="n">rotation</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">eigenvalues</span><span class="p">)</span> <span class="o">@</span> <span class="n">rotation</span><span class="o">.</span><span class="n">T</span>
    <span class="n">end</span> <span class="o">=</span> <span class="n">rotation</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="mf">1.0</span> <span class="o">/</span> <span class="n">eigenvalues</span><span class="p">)</span> <span class="o">@</span> <span class="n">rotation</span><span class="o">.</span><span class="n">T</span>
    <span class="k">return</span> <span class="n">start</span><span class="p">,</span> <span class="n">end</span>


<span class="k">def</span><span class="w"> </span><span class="nf">local_minima</span><span class="p">(</span><span class="n">times</span><span class="p">,</span> <span class="n">values</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Indices of the strict local minima of a sampled curve, ends included.&quot;&quot;&quot;</span>
    <span class="n">interior</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">index</span>
        <span class="k">for</span> <span class="n">index</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">values</span><span class="p">)</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">values</span><span class="p">[</span><span class="n">index</span><span class="p">]</span> <span class="o">&lt;</span> <span class="n">values</span><span class="p">[</span><span class="n">index</span> <span class="o">-</span> <span class="mi">1</span><span class="p">]</span> <span class="ow">and</span> <span class="n">values</span><span class="p">[</span><span class="n">index</span><span class="p">]</span> <span class="o">&lt;</span> <span class="n">values</span><span class="p">[</span><span class="n">index</span> <span class="o">+</span> <span class="mi">1</span><span class="p">]</span>
    <span class="p">]</span>
    <span class="k">if</span> <span class="n">values</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">&lt;</span> <span class="n">values</span><span class="p">[</span><span class="mi">1</span><span class="p">]:</span>
        <span class="n">interior</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">values</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="o">&lt;</span> <span class="n">values</span><span class="p">[</span><span class="o">-</span><span class="mi">2</span><span class="p">]:</span>
        <span class="n">interior</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">values</span><span class="p">)</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">interior</span>


<span class="k">def</span><span class="w"> </span><span class="nf">second_difference</span><span class="p">(</span><span class="n">times</span><span class="p">,</span> <span class="n">values</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Discrete second derivative, negative wherever the curve is concave.&quot;&quot;&quot;</span>
    <span class="n">step</span> <span class="o">=</span> <span class="n">times</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="o">-</span> <span class="n">times</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">values</span><span class="p">[</span><span class="mi">2</span><span class="p">:]</span> <span class="o">-</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">values</span><span class="p">[</span><span class="mi">1</span><span class="p">:</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="o">+</span> <span class="n">values</span><span class="p">[:</span><span class="o">-</span><span class="mi">2</span><span class="p">])</span> <span class="o">/</span> <span class="n">step</span><span class="o">**</span><span class="mi">2</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Tyler&#39;s cost read along a Euclidean segment and along a geodesic.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Dimension of the observations.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of observations. A short sample makes the cost surface &quot;</span>
             <span class="s2">&quot;sharper, hence the effect easier to see; the phenomenon itself &quot;</span>
             <span class="s2">&quot;does not depend on it.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dof&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">3.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Degrees of freedom of the Student data.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--condition&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e4</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Condition number of the two endpoints. The larger it is, the &quot;</span>
             <span class="s2">&quot;more pronounced the interior maximum of the Euclidean reading.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_points&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">201</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of points at which the cost is evaluated on each path.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/riemann_gconvexite&quot;</span><span class="p">,</span>
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

    <span class="n">d</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span>
    <span class="n">manifold</span> <span class="o">=</span> <span class="n">HermitianPositiveDefinite</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>

    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>
    <span class="n">rotation</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">qr</span><span class="p">(</span><span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">d</span><span class="p">,</span> <span class="n">d</span><span class="p">)))[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">start</span><span class="p">,</span> <span class="n">end</span> <span class="o">=</span> <span class="n">endpoints</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">rotation</span><span class="p">)</span>

    <span class="c1"># Spherical data: the true scatter matrix is the identity, which is both</span>
    <span class="c1"># the middle of the geodesic and the point the two endpoints surround.</span>
    <span class="n">distribution</span> <span class="o">=</span> <span class="n">StudentTDistribution</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span>
        <span class="n">sample_elliptical</span><span class="p">(</span>
            <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span>
            <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">d</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
            <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">eye</span><span class="p">(</span><span class="n">d</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
            <span class="n">distribution</span><span class="p">,</span>
            <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="p">)</span>
    <span class="n">data_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">start_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">start</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">end_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">end</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">direction</span> <span class="o">=</span> <span class="n">manifold</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">start_device</span><span class="p">,</span> <span class="n">end_device</span><span class="p">)</span>

    <span class="n">times</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_points</span><span class="p">)</span>
    <span class="n">costs</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;segment euclidien&quot;</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span>
            <span class="n">tyler_cost</span><span class="p">(</span>
                <span class="n">data_device</span><span class="p">,</span>
                <span class="n">get_data_on_device</span><span class="p">((</span><span class="mf">1.0</span> <span class="o">-</span> <span class="n">t</span><span class="p">)</span> <span class="o">*</span> <span class="n">start</span> <span class="o">+</span> <span class="n">t</span> <span class="o">*</span> <span class="n">end</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
                <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
            <span class="p">)</span>
            <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">times</span>
        <span class="p">]),</span>
        <span class="s2">&quot;géodésique&quot;</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span>
            <span class="n">tyler_cost</span><span class="p">(</span>
                <span class="n">data_device</span><span class="p">,</span> <span class="n">manifold</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">start_device</span><span class="p">,</span> <span class="n">t</span> <span class="o">*</span> <span class="n">direction</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span>
            <span class="p">)</span>
            <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">times</span>
        <span class="p">]),</span>
    <span class="p">}</span>

    <span class="n">colors</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;segment euclidien&quot;</span><span class="p">:</span> <span class="s2">&quot;C1&quot;</span><span class="p">,</span> <span class="s2">&quot;géodésique&quot;</span><span class="p">:</span> <span class="s2">&quot;C2&quot;</span><span class="p">}</span>

    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">),</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">ax</span><span class="p">,</span> <span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">values</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">costs</span><span class="o">.</span><span class="n">items</span><span class="p">()):</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">times</span><span class="p">,</span> <span class="n">values</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.5</span><span class="p">)</span>
        <span class="n">minima</span> <span class="o">=</span> <span class="n">local_minima</span><span class="p">(</span><span class="n">times</span><span class="p">,</span> <span class="n">values</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">times</span><span class="p">[</span><span class="n">minima</span><span class="p">],</span> <span class="n">values</span><span class="p">[</span><span class="n">minima</span><span class="p">],</span>
            <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C7&quot;</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$t$&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">name</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$L$&quot;</span><span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;d = </span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">, N = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">, Student nu = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;condition </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, geodesic distance between the &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;endpoints </span><span class="si">{</span><span class="nb">float</span><span class="p">(</span><span class="n">manifold</span><span class="o">.</span><span class="n">dist</span><span class="p">(</span><span class="n">start_device</span><span class="p">,</span><span class="w"> </span><span class="n">end_device</span><span class="p">))</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">values</span> <span class="ow">in</span> <span class="n">costs</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">curvature</span> <span class="o">=</span> <span class="n">second_difference</span><span class="p">(</span><span class="n">times</span><span class="p">,</span> <span class="n">values</span><span class="p">)</span>
        <span class="n">minima</span> <span class="o">=</span> <span class="n">local_minima</span><span class="p">(</span><span class="n">times</span><span class="p">,</span> <span class="n">values</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;  </span><span class="si">{</span><span class="n">name</span><span class="si">:</span><span class="s2">18</span><span class="si">}</span><span class="s2"> min curvature </span><span class="si">{</span><span class="n">curvature</span><span class="o">.</span><span class="n">min</span><span class="p">()</span><span class="si">:</span><span class="s2">+9.2f</span><span class="si">}</span><span class="s2">   &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;local minima at t = &quot;</span>
            <span class="o">+</span> <span class="s2">&quot;, &quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">times</span><span class="p">[</span><span class="n">index</span><span class="p">]</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">&quot;</span> <span class="k">for</span> <span class="n">index</span> <span class="ow">in</span> <span class="n">minima</span><span class="p">)</span>
        <span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_features</span><span class="o">=</span><span class="n">d</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="p">,</span>
        <span class="n">condition</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">n_points</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_points</span><span class="p">,</span>
        <span class="n">start</span><span class="o">=</span><span class="n">start</span><span class="p">,</span> <span class="n">end</span><span class="o">=</span><span class="n">end</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">data</span><span class="p">,</span> <span class="n">times</span><span class="o">=</span><span class="n">times</span><span class="p">,</span>
        <span class="n">cost_euclidean</span><span class="o">=</span><span class="n">costs</span><span class="p">[</span><span class="s2">&quot;segment euclidien&quot;</span><span class="p">],</span>
        <span class="n">cost_geodesic</span><span class="o">=</span><span class="n">costs</span><span class="p">[</span><span class="s2">&quot;géodésique&quot;</span><span class="p">],</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;gconvexite.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved cost profiles in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_features</span><span class="param-type">int</span><span class="param-default">default <b>3</b></span>
</div>
<p class="param-help">Dimension of the observations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>10</b></span>
</div>
<p class="param-help">Number of observations. A short sample makes the cost surface sharper, hence the effect easier to see; the phenomenon itself does not depend on it.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dof</span><span class="param-type">float</span><span class="param-default">default <b>3.0</b></span>
</div>
<p class="param-help">Degrees of freedom of the Student data.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--condition</span><span class="param-type">float</span><span class="param-default">default <b>10000.0</b></span>
</div>
<p class="param-help">Condition number of the two endpoints. The larger it is, the more pronounced the interior maximum of the Euclidean reading.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_points</span><span class="param-type">int</span><span class="param-default">default <b>201</b></span>
</div>
<p class="param-help">Number of points at which the cost is evaluated on each path.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/riemann_gconvexite</b></span>
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
  <code>--n_features</code> <span class='mn-default'>3</span><br>
  <code>--n_samples</code> <span class='mn-default'>10</span><br>
  <code>--dof</code> <span class='mn-default'>3.0</span><br>
  <code>--condition</code> <span class='mn-default'>10000.0</span><br>
  <code>--n_points</code> <span class='mn-default'>201</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--axis_width</code> <span class='mn-default'>0.45\textwidth</span><br>
  <code>--axis_height</code> <span class='mn-default'>4.6cm</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_gconvexite.json" data-title="context_riemann_gconvexite"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">d = 3, N = 10, Student nu = 3, condition 10000, geodesic distance between the endpoints 13.025
  segment euclidien  min curvature    -43.54   local minima at t = 0.01, 0.96
  géodésique         min curvature     +8.15   local minima at t = 0.45
Saved cost profiles in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_riemann_gconvexite/run_31/gconvexite.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_riemann_gconvexite.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
