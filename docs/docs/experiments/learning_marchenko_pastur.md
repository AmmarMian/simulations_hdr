<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/3-learning/">3 · Learning</a>
<span class="sep">/</span>
<span class="here">learning_marchenko_pastur</span>
</nav>

# learning_marchenko_pastur

Marchenko-Pastur law — histogram of the SCM eigenvalues against the theoretical density, for three concentration ratios

**Tags:** `learning`  `random-matrix-theory`  `monte-carlo`

## Run

```sh
uv run python 3-learning/marchenko_pastur/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/3-learning/marchenko_pastur/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">213 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">3-learning/marchenko_pastur/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Marchenko-Pastur: what the dimensional regime does to a spectrum</span>
<span class="c1">#</span>
<span class="c1"># The true covariance is the identity, so every one of its eigenvalues is 1.</span>
<span class="c1"># The eigenvalues of the sample covariance matrix are not: they spread over</span>
<span class="c1"># [(1-sqrt(c))^2, (1+sqrt(c))^2] with c = d/N, and that spread does not shrink</span>
<span class="c1"># when more data is added — it depends on c alone. Adding data at fixed c is</span>
<span class="c1"># not the same thing as adding data at fixed d.</span>
<span class="c1">#</span>
<span class="c1"># The figure is the illustration behind the remark &quot;Ce que cette loi dit</span>
<span class="c1"># vraiment&quot; of the learning chapter (ch:learning, subsec:learning-rmt), which</span>
<span class="c1"># is the claim the whole RMT correction rests on: the bias is deterministic and</span>
<span class="c1"># perfectly described, hence correctable — unlike the shrinkage of</span>
<span class="c1"># sec:context-covariance-regularized-estimation, which contracts the spectrum</span>
<span class="c1"># without knowing by how much.</span>
<span class="c1">#</span>
<span class="c1"># Three panels, one per concentration ratio. Each superposes the histogram of</span>
<span class="c1"># the pooled eigenvalues of n_trials sample covariance matrices on the</span>
<span class="c1"># Marchenko-Pastur density (eq:learning-marchenko-pastur). The dimension d is</span>
<span class="c1"># held fixed and N is derived from c, so the three panels differ by the sample</span>
<span class="c1"># support alone.</span>
<span class="c1">#</span>
<span class="c1"># Backend-free: the sampling and the eigenvalue decomposition go through</span>
<span class="c1"># hdrlib.core.backend, so the same script runs on numpy, torch, cupy or jax.</span>
<span class="c1"># The density itself is evaluated in numpy — it is a scalar formula on a</span>
<span class="c1"># plotting grid, not a computation on the data.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">batched_eigh</span><span class="p">,</span>
    <span class="n">get_backend_module</span><span class="p">,</span>
    <span class="n">sample_standard_normal</span><span class="p">,</span>
    <span class="n">to_numpy</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="n">add_mc_base_args</span><span class="p">,</span> <span class="n">init_logging</span><span class="p">,</span> <span class="n">make_mc_parser</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span>


<span class="k">def</span><span class="w"> </span><span class="nf">marchenko_pastur_density</span><span class="p">(</span><span class="n">grid</span><span class="p">,</span> <span class="n">ratio</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Density of eq:learning-marchenko-pastur, zero outside its support.</span>

<span class="sd">    Valid for ``ratio &lt; 1``. At ``ratio == 1`` the lower edge reaches the</span>
<span class="sd">    origin and the density diverges there like 1/sqrt(x); the value is finite</span>
<span class="sd">    everywhere the grid actually samples, so the formula is used as is and the</span>
<span class="sd">    plotting range is what keeps the picture readable.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">lower</span> <span class="o">=</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">-</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">ratio</span><span class="p">))</span> <span class="o">**</span> <span class="mi">2</span>
    <span class="n">upper</span> <span class="o">=</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">+</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">ratio</span><span class="p">))</span> <span class="o">**</span> <span class="mi">2</span>
    <span class="n">density</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros_like</span><span class="p">(</span><span class="n">grid</span><span class="p">)</span>
    <span class="n">inside</span> <span class="o">=</span> <span class="p">(</span><span class="n">grid</span> <span class="o">&gt;</span> <span class="n">lower</span><span class="p">)</span> <span class="o">&amp;</span> <span class="p">(</span><span class="n">grid</span> <span class="o">&lt;</span> <span class="n">upper</span><span class="p">)</span>
    <span class="n">density</span><span class="p">[</span><span class="n">inside</span><span class="p">]</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span>
        <span class="p">(</span><span class="n">upper</span> <span class="o">-</span> <span class="n">grid</span><span class="p">[</span><span class="n">inside</span><span class="p">])</span> <span class="o">*</span> <span class="p">(</span><span class="n">grid</span><span class="p">[</span><span class="n">inside</span><span class="p">]</span> <span class="o">-</span> <span class="n">lower</span><span class="p">)</span>
    <span class="p">)</span> <span class="o">/</span> <span class="p">(</span><span class="mf">2.0</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span> <span class="o">*</span> <span class="n">ratio</span> <span class="o">*</span> <span class="n">grid</span><span class="p">[</span><span class="n">inside</span><span class="p">])</span>
    <span class="k">return</span> <span class="n">density</span><span class="p">,</span> <span class="n">lower</span><span class="p">,</span> <span class="n">upper</span>


<span class="k">def</span><span class="w"> </span><span class="nf">sample_eigenvalues</span><span class="p">(</span><span class="n">n_features</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">n_trials</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">seed</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Pooled eigenvalues of ``n_trials`` sample covariance matrices.</span>

<span class="sd">    The data of every trial is drawn in one call and the trials are stacked in</span>
<span class="sd">    the leading dimension, so a single batched eigendecomposition covers the</span>
<span class="sd">    whole Monte-Carlo. This is the shape ``batched_eigh`` is written for, and</span>
<span class="sd">    it is what makes the non-numpy backends worth anything here.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">bm</span> <span class="o">=</span> <span class="n">get_backend_module</span><span class="p">(</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">sample_standard_normal</span><span class="p">(</span>
        <span class="n">n_trials</span><span class="p">,</span> <span class="p">[</span><span class="n">n_features</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">],</span> <span class="n">backend</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">seed</span>
    <span class="p">)</span>
    <span class="c1"># One SCM per trial: (n_trials, d, N) -&gt; (n_trials, d, d). The true</span>
    <span class="c1"># covariance is the identity, so no whitening is needed.</span>
    <span class="n">scm</span> <span class="o">=</span> <span class="n">bm</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">_transpose</span><span class="p">(</span><span class="n">bm</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">backend</span><span class="p">))</span> <span class="o">/</span> <span class="n">n_samples</span>
    <span class="n">eigenvalues</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">batched_eigh</span><span class="p">(</span><span class="n">backend</span><span class="p">,</span> <span class="n">scm</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">eigenvalues</span><span class="p">)</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_transpose</span><span class="p">(</span><span class="n">bm</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Swap the last two axes, whichever backend ``x`` lives on.&quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="nb">hasattr</span><span class="p">(</span><span class="n">bm</span><span class="p">,</span> <span class="s2">&quot;swapaxes&quot;</span><span class="p">):</span>
        <span class="k">return</span> <span class="n">bm</span><span class="o">.</span><span class="n">swapaxes</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">bm</span><span class="o">.</span><span class="n">transpose</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">)</span>  <span class="c1"># torch</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span>
        <span class="s2">&quot;Marchenko-Pastur law: histogram of the sample covariance eigenvalues &quot;</span>
        <span class="s2">&quot;against the theoretical density, for several concentration ratios.&quot;</span>
    <span class="p">)</span>
    <span class="n">add_mc_base_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Dimension d, held fixed across the panels. Large enough for the &quot;</span>
             <span class="s2">&quot;asymptotic density to be visible, small enough for the batched &quot;</span>
             <span class="s2">&quot;eigendecomposition to stay cheap.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--ratios&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Concentration ratios c = d/N, one panel each. The number of &quot;</span>
             <span class="s2">&quot;samples of a panel is N = round(d / c), so only the sample &quot;</span>
             <span class="s2">&quot;support changes from one panel to the next.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_bins&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">80</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Histogram bins per panel.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_width&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.31</span><span class="se">\\</span><span class="s2">textwidth&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of a single panel in the exported PGFPlots figure. Set &quot;</span>
             <span class="s2">&quot;here rather than patched into the .tex afterwards, so that a &quot;</span>
             <span class="s2">&quot;re-sync into the dissertation does not undo it.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_height&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;4.4cm&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Height of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>
    <span class="c1"># The base MC parser defaults to 10 000 trials, which is far more than this</span>
    <span class="c1"># figure needs: every trial already contributes d eigenvalues to the</span>
    <span class="c1"># histogram, so a few hundred trials give a smooth curve.</span>
    <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">export_path</span>

    <span class="n">init_logging</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">d</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span>
        <span class="mi">1</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">ratios</span><span class="p">),</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.0</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">ratios</span><span class="p">),</span> <span class="mf">3.0</span><span class="p">),</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">False</span>
    <span class="p">)</span>
    <span class="n">axes</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">atleast_1d</span><span class="p">(</span><span class="n">axes</span><span class="p">)</span>

    <span class="n">saved</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">ratio</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">ratios</span><span class="p">)):</span>
        <span class="n">n_samples</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">round</span><span class="p">(</span><span class="n">d</span> <span class="o">/</span> <span class="n">ratio</span><span class="p">))</span>
        <span class="c1"># Each panel gets its own seed, otherwise the three would share the</span>
        <span class="c1"># same underlying draw and the comparison would be between three views</span>
        <span class="c1"># of one realisation rather than three independent ones.</span>
        <span class="n">eigenvalues</span> <span class="o">=</span> <span class="n">sample_eigenvalues</span><span class="p">(</span>
            <span class="n">d</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="n">index</span>
        <span class="p">)</span>

        <span class="n">upper_plot</span> <span class="o">=</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">+</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">ratio</span><span class="p">))</span> <span class="o">**</span> <span class="mi">2</span> <span class="o">*</span> <span class="mf">1.15</span>
        <span class="n">grid</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mf">1e-4</span><span class="p">,</span> <span class="n">upper_plot</span><span class="p">,</span> <span class="mi">600</span><span class="p">)</span>
        <span class="n">density</span><span class="p">,</span> <span class="n">lower</span><span class="p">,</span> <span class="n">upper</span> <span class="o">=</span> <span class="n">marchenko_pastur_density</span><span class="p">(</span><span class="n">grid</span><span class="p">,</span> <span class="n">ratio</span><span class="p">)</span>

        <span class="c1"># Only the first panel carries labels: matplot2tikz emits one legend</span>
        <span class="c1"># entry per labelled artist per axis, so labelling all three panels</span>
        <span class="c1"># would print the same legend three times under the exported figure.</span>
        <span class="n">first</span> <span class="o">=</span> <span class="n">index</span> <span class="o">==</span> <span class="mi">0</span>
        <span class="n">counts</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">ax</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span>
            <span class="n">eigenvalues</span><span class="p">,</span> <span class="n">bins</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_bins</span><span class="p">,</span> <span class="nb">range</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="n">upper_plot</span><span class="p">),</span>
            <span class="n">density</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C0&quot;</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.55</span><span class="p">,</span> <span class="n">edgecolor</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="s2">&quot;valeurs propres de la scm&quot;</span> <span class="k">if</span> <span class="n">first</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">grid</span><span class="p">,</span> <span class="n">density</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C3&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.6</span><span class="p">,</span>
                <span class="n">label</span><span class="o">=</span><span class="s2">&quot;loi de MP&quot;</span> <span class="k">if</span> <span class="n">first</span> <span class="k">else</span> <span class="kc">None</span><span class="p">)</span>
        <span class="c1"># The true spectrum is a single point; it is worth drawing, because the</span>
        <span class="c1"># whole reading of the figure is the gap between it and the histogram.</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;k&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
                   <span class="n">label</span><span class="o">=</span><span class="s2">&quot;spectre vrai&quot;</span> <span class="k">if</span> <span class="n">first</span> <span class="k">else</span> <span class="kc">None</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="n">upper_plot</span><span class="p">)</span>
        <span class="c1"># At c = 1 the density diverges at the origin like 1/sqrt(x): letting</span>
        <span class="c1"># the axis follow the theoretical curve would flatten the histogram of</span>
        <span class="c1"># that panel into the baseline. The frame is set on the histogram</span>
        <span class="c1"># instead, so the three panels stay comparable and the divergence</span>
        <span class="c1"># simply leaves the top of the frame.</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">1.25</span> <span class="o">*</span> <span class="nb">float</span><span class="p">(</span><span class="n">counts</span><span class="o">.</span><span class="n">max</span><span class="p">()))</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\lambda$&quot;</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">index</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;densité&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="sa">rf</span><span class="s2">&quot;$c = </span><span class="si">{</span><span class="n">ratio</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">)</span>

        <span class="n">saved</span><span class="p">[</span><span class="sa">f</span><span class="s2">&quot;eigenvalues_c</span><span class="si">{</span><span class="n">index</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">eigenvalues</span>
        <span class="n">saved</span><span class="p">[</span><span class="sa">f</span><span class="s2">&quot;ratio_c</span><span class="si">{</span><span class="n">index</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">ratio</span>
        <span class="n">saved</span><span class="p">[</span><span class="sa">f</span><span class="s2">&quot;n_samples_c</span><span class="si">{</span><span class="n">index</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">n_samples</span>
        <span class="nb">print</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;c = </span><span class="si">{</span><span class="n">ratio</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">: d = </span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">, N = </span><span class="si">{</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">, &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;support [</span><span class="si">{</span><span class="n">lower</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">upper</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">], &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;observed [</span><span class="si">{</span><span class="n">eigenvalues</span><span class="o">.</span><span class="n">min</span><span class="p">()</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">eigenvalues</span><span class="o">.</span><span class="n">max</span><span class="p">()</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">]&quot;</span>
        <span class="p">)</span>

    <span class="c1"># Legend below the panels rather than inside one of them: at export size an</span>
    <span class="c1"># inner legend either covers the histogram or spills out of the frame. It</span>
    <span class="c1"># is attached to an axis and not to the figure, since matplot2tikz exports</span>
    <span class="c1"># axis legends and silently drops figure ones.</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;upper left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.32</span><span class="p">),</span> <span class="n">ncol</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
        <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
            <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
            <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_features</span><span class="o">=</span><span class="n">d</span><span class="p">,</span> <span class="n">n_trials</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span>
            <span class="n">ratios</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">ratios</span><span class="p">),</span> <span class="n">n_bins</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_bins</span><span class="p">,</span>
            <span class="o">**</span><span class="n">saved</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;marchenko.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved Marchenko-Pastur figure in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_features</span><span class="param-type">int</span><span class="param-default">default <b>200</b></span>
</div>
<p class="param-help">Dimension d, held fixed across the panels. Large enough for the asymptotic density to be visible, small enough for the batched eigendecomposition to stay cheap.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--ratios</span><span class="param-type">float</span><span class="param-default">default <b>[0.1, 0.5, 1.0]</b></span>
</div>
<p class="param-help">Concentration ratios c = d/N, one panel each. The number of samples of a panel is N = round(d / c), so only the sample support changes from one panel to the next.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_bins</span><span class="param-type">int</span><span class="param-default">default <b>80</b></span>
</div>
<p class="param-help">Histogram bins per panel.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_width</span><span class="param-type">str</span><span class="param-default">default <b>0.31\textwidth</b></span>
</div>
<p class="param-help">Width of a single panel in the exported PGFPlots figure. Set here rather than patched into the .tex afterwards, so that a re-sync into the dissertation does not undo it.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_height</span><span class="param-type">str</span><span class="param-default">default <b>4.4cm</b></span>
</div>
<p class="param-help">Height of a single panel in the exported PGFPlots figure.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-trials</span><span class="param-type">int</span><span class="param-default">default <b>10000</b></span>
</div>
<p class="param-help">Number of Monte-Carlo trials (default 10000).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">RNG seed for data generation (default 42).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--backend</span><span class="param-type">str</span><span class="param-default">default <b>numpy</b></span>
</div>
<p class="param-help">Compute backend. numpy → multiprocessing.Pool (one worker per trial); all others → trials stacked in leading batch dimension (default numpy).</p><p class="param-choices">choices: numpy, torch-cpu, torch-cuda, torch-mps, jax-cpu, jax-cuda, jax-metal, cupy</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-workers</span><span class="param-type">int</span>
</div>
<p class="param-help">Pool workers for numpy backend (default: os.cpu_count()).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--export</span><span class="param-default">default <b>True</b></span>
</div>
<p class="param-help">Save .npz results + provenance sidecar + plot script (default: True).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage-path</span><span class="param-alias">--storage_path</span><span class="param-alias">--export-path</span><span class="param-type">str</span><span class="param-default">default <b>./exports</b></span>
</div>
<p class="param-help">Directory for exported results; --storage-path is the qanat alias (default: ./exports).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--show-interactive</span><span class="param-type">flag</span>
</div>
<p class="param-help">Display figures interactively at the end of the simulation.</p>
</div>
</div>

## Results

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--n-trials 400</code><br>
  <code>--seed 42</code><br>
  <code>--backend numpy</code><br>
  <code>--n_features 200</code><br>
  <code>--ratios [0.1, 0.5, 1.0]</code><br>
  <code>--n_bins 80</code><br>
  <code>--axis_width 0.31\textwidth</code><br>
  <code>--axis_height 4.4cm</code><br>
  <span class="mn-date">3acd057 · 2026-08-27</span>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/learning_marchenko_pastur.json" data-title="learning_marchenko_pastur"></div>
</div>

## Config

`3-learning/experiments/learning_marchenko_pastur.yaml`

<a class="back-link" href="../../chapters/3-learning/">← All experiments in 3 · Learning</a>
