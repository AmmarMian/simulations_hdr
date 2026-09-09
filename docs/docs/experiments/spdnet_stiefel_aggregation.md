<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/4-deeplearning/">4 · Deep Learning</a>
<span class="sep">/</span>
<span class="here">spdnet_stiefel_aggregation</span>
</nav>

# spdnet_stiefel_aggregation

Order at which the projavg and rlavg aggregations coincide on the Stiefel manifold

**Tags:** `deeplearning`  `spdnet`  `stiefel`  `federated`

## Run

```sh
uv run python 4-deeplearning/stiefel_aggregation/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/4-deeplearning/stiefel_aggregation/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">285 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">4-deeplearning/stiefel_aggregation/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># How far apart are the two aggregations of sec:spdnet-federe-agregation?</span>
<span class="c1">#</span>
<span class="c1"># prop:spdnet-federe-equivalence states that projavg (eq:spdnet-projavg) and</span>
<span class="c1"># rlavg (eq:spdnet-rlavg) coincide up to O(eps^2) when the local weights stay</span>
<span class="c1"># within O(eps) of the global iterate. The chapter currently supports that with</span>
<span class="c1"># EEG validation curves that lie on top of each other, which shows the two agree</span>
<span class="c1"># but says nothing about the *order* at which they do.</span>
<span class="c1">#</span>
<span class="c1"># This measures the order directly, with no data and no training: draw K local</span>
<span class="c1"># weights at geodesic distance eps from a base point of the Stiefel manifold,</span>
<span class="c1"># aggregate both ways, and look at ||projavg - rlavg||_F as eps shrinks.</span>
<span class="c1">#</span>
<span class="c1"># The measured order is three, not two — the proposition is true but</span>
<span class="c1"># conservative. Swept over (d0, d1, K) here rather than asserted from one</span>
<span class="c1"># geometry, since a single configuration cannot tell an exponent from a</span>
<span class="c1"># coincidence.</span>
<span class="c1">#</span>
<span class="c1"># Reuses stiefel_projection_polar and stiefel_projection_tangent_orthogonal of</span>
<span class="c1"># yetanotherspdnet, which are exactly the polarf and Lift of def:spdnet-lift.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="n">Progress</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">yetanotherspdnet.functions.stiefel</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">stiefel_projection_polar</span><span class="p">,</span>
    <span class="n">stiefel_projection_tangent_orthogonal</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">yetanotherspdnet.random.stiefel</span><span class="w"> </span><span class="kn">import</span> <span class="n">random_stiefel</span>


<span class="k">def</span><span class="w"> </span><span class="nf">aggregate_both_ways</span><span class="p">(</span><span class="n">base</span><span class="p">,</span> <span class="n">locals_</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;projavg and rlavg of the same local weights, at the same base point.&quot;&quot;&quot;</span>
    <span class="n">projavg</span> <span class="o">=</span> <span class="n">stiefel_projection_polar</span><span class="p">(</span><span class="n">locals_</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
    <span class="n">lifted</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span>
        <span class="p">[</span><span class="n">stiefel_projection_tangent_orthogonal</span><span class="p">(</span><span class="n">local</span> <span class="o">-</span> <span class="n">base</span><span class="p">,</span> <span class="n">base</span><span class="p">)</span> <span class="k">for</span> <span class="n">local</span> <span class="ow">in</span> <span class="n">locals_</span><span class="p">]</span>
    <span class="p">)</span>
    <span class="n">rlavg</span> <span class="o">=</span> <span class="n">stiefel_projection_polar</span><span class="p">(</span><span class="n">base</span> <span class="o">+</span> <span class="n">lifted</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">dim</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">projavg</span><span class="p">,</span> <span class="n">rlavg</span>


<span class="k">def</span><span class="w"> </span><span class="nf">local_weights</span><span class="p">(</span><span class="n">base</span><span class="p">,</span> <span class="n">dispersion</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">,</span> <span class="n">generator</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;K points of the manifold at distance ``dispersion`` from ``base``.</span>

<span class="sd">    Each is obtained by retracting a tangent vector of norm exactly</span>
<span class="sd">    ``dispersion``, so the dispersion of the clients is a controlled quantity</span>
<span class="sd">    and not the by-product of a random draw — which is what lets an exponent be</span>
<span class="sd">    read off the result.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">weights</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n_clients</span><span class="p">):</span>
        <span class="n">ambient</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">randn</span><span class="p">(</span>
            <span class="n">base</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="n">generator</span><span class="o">=</span><span class="n">generator</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">base</span><span class="o">.</span><span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">base</span><span class="o">.</span><span class="n">dtype</span>
        <span class="p">)</span>
        <span class="n">tangent</span> <span class="o">=</span> <span class="n">stiefel_projection_tangent_orthogonal</span><span class="p">(</span><span class="n">ambient</span><span class="p">,</span> <span class="n">base</span><span class="p">)</span>
        <span class="n">tangent</span> <span class="o">=</span> <span class="n">tangent</span> <span class="o">/</span> <span class="n">tangent</span><span class="o">.</span><span class="n">norm</span><span class="p">()</span> <span class="o">*</span> <span class="n">dispersion</span>
        <span class="n">weights</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">stiefel_projection_polar</span><span class="p">(</span><span class="n">base</span> <span class="o">+</span> <span class="n">tangent</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">weights</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">measure</span><span class="p">(</span><span class="n">dimensions</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">,</span> <span class="n">dispersions</span><span class="p">,</span> <span class="n">n_repeats</span><span class="p">,</span> <span class="n">generator</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Median gap between the two aggregations, at every dispersion.&quot;&quot;&quot;</span>
    <span class="n">n_in</span><span class="p">,</span> <span class="n">n_out</span> <span class="o">=</span> <span class="n">dimensions</span>
    <span class="n">gaps</span><span class="p">,</span> <span class="n">displacements</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">dispersion</span> <span class="ow">in</span> <span class="n">dispersions</span><span class="p">:</span>
        <span class="n">trial_gaps</span><span class="p">,</span> <span class="n">trial_displacements</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n_repeats</span><span class="p">):</span>
            <span class="n">base</span> <span class="o">=</span> <span class="n">random_stiefel</span><span class="p">(</span>
                <span class="n">n_in</span><span class="p">,</span> <span class="n">n_out</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">generator</span><span class="o">=</span><span class="n">generator</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">dtype</span>
            <span class="p">)</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
            <span class="n">locals_</span> <span class="o">=</span> <span class="n">local_weights</span><span class="p">(</span><span class="n">base</span><span class="p">,</span> <span class="n">dispersion</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">,</span> <span class="n">generator</span><span class="p">)</span>
            <span class="n">projavg</span><span class="p">,</span> <span class="n">rlavg</span> <span class="o">=</span> <span class="n">aggregate_both_ways</span><span class="p">(</span><span class="n">base</span><span class="p">,</span> <span class="n">locals_</span><span class="p">)</span>
            <span class="n">trial_gaps</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="nb">float</span><span class="p">((</span><span class="n">projavg</span> <span class="o">-</span> <span class="n">rlavg</span><span class="p">)</span><span class="o">.</span><span class="n">norm</span><span class="p">()))</span>
            <span class="c1"># How far the aggregate itself moved. The gap is only meaningful</span>
            <span class="c1"># against this: two schemes that agree to 1e-8 while both barely</span>
            <span class="c1"># moving would not be saying much.</span>
            <span class="n">trial_displacements</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="nb">float</span><span class="p">((</span><span class="n">projavg</span> <span class="o">-</span> <span class="n">base</span><span class="p">)</span><span class="o">.</span><span class="n">norm</span><span class="p">()))</span>
        <span class="n">gaps</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">trial_gaps</span><span class="p">)))</span>
        <span class="n">displacements</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">trial_displacements</span><span class="p">)))</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">gaps</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">displacements</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">fitted_order</span><span class="p">(</span><span class="n">dispersions</span><span class="p">,</span> <span class="n">gaps</span><span class="p">,</span> <span class="n">floor</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Slope of log(gap) against log(dispersion), above the rounding floor.</span>

<span class="sd">    Points at or below ``floor`` are dropped: once the gap reaches machine</span>
<span class="sd">    precision it stops following the exponent and flattens, and including that</span>
<span class="sd">    tail would drag any fit towards zero.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">dispersions</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">dispersions</span><span class="p">)</span>
    <span class="n">usable</span> <span class="o">=</span> <span class="n">gaps</span> <span class="o">&gt;</span> <span class="n">floor</span>
    <span class="k">if</span> <span class="n">usable</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">&lt;</span> <span class="mi">2</span><span class="p">:</span>
        <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="s2">&quot;nan&quot;</span><span class="p">)</span>
    <span class="n">slope</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">polyfit</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">dispersions</span><span class="p">[</span><span class="n">usable</span><span class="p">]),</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">gaps</span><span class="p">[</span><span class="n">usable</span><span class="p">]),</span> <span class="mi">1</span><span class="p">)</span>
    <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">slope</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Order at which the projavg and rlavg aggregations of &quot;</span>
        <span class="s2">&quot;prop:spdnet-federe-equivalence coincide.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dimensions&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;append&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;A Stiefel geometry as &#39;n_in n_out&#39;. Repeat the flag for several. &quot;</span>
             <span class="s2">&quot;Defaults to (40, 20), (128, 32) and (64, 60).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_clients&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">2</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">32</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Numbers of clients aggregated per round.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dispersions&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mf">1e0</span><span class="p">,</span> <span class="mf">1e-1</span><span class="p">,</span> <span class="mf">1e-2</span><span class="p">,</span> <span class="mf">1e-3</span><span class="p">,</span> <span class="mf">1e-4</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Distances between a local weight and the global iterate. Stops &quot;</span>
             <span class="s2">&quot;at 1e-4: below that the gap is at the float64 rounding floor and &quot;</span>
             <span class="s2">&quot;carries no exponent.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_repeats&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">20</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Draws of the base point and the clients at each dispersion.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--floor&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-13</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Gaps at or below this are treated as rounding, not signal.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/stiefel_aggregation&quot;</span><span class="p">,</span>
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
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_height&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;4.6cm&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Height of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--device&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;cpu&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Compute device: cpu or cuda.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Base seed.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">geometries</span> <span class="o">=</span> <span class="p">(</span>
        <span class="p">[</span><span class="nb">tuple</span><span class="p">(</span><span class="n">pair</span><span class="p">)</span> <span class="k">for</span> <span class="n">pair</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">dimensions</span><span class="p">]</span>
        <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">dimensions</span>
        <span class="k">else</span> <span class="p">[(</span><span class="mi">40</span><span class="p">,</span> <span class="mi">20</span><span class="p">),</span> <span class="p">(</span><span class="mi">128</span><span class="p">,</span> <span class="mi">32</span><span class="p">),</span> <span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="mi">60</span><span class="p">)]</span>
    <span class="p">)</span>
    <span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">device</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">device</span><span class="p">)</span>
    <span class="n">dtype</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">float64</span>

    <span class="n">generator</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">Generator</span><span class="p">(</span><span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
    <span class="n">generator</span><span class="o">.</span><span class="n">manual_seed</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>

    <span class="n">results</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="c1"># One step per (geometry, client count): the unit the sweep is written in,</span>
    <span class="c1"># and the one whose cost the user controls through --dimensions and</span>
    <span class="c1"># --n_clients.</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">geometries</span><span class="p">)</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">),</span>
        <span class="n">description</span><span class="o">=</span><span class="s2">&quot;Geometries x clients&quot;</span><span class="p">,</span> <span class="n">unit</span><span class="o">=</span><span class="s2">&quot;fits&quot;</span><span class="p">,</span>
    <span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">dimensions</span> <span class="ow">in</span> <span class="n">geometries</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">n_clients</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">:</span>
                <span class="n">gaps</span><span class="p">,</span> <span class="n">displacements</span> <span class="o">=</span> <span class="n">measure</span><span class="p">(</span>
                    <span class="n">dimensions</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">dispersions</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_repeats</span><span class="p">,</span>
                    <span class="n">generator</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="p">,</span>
                <span class="p">)</span>
                <span class="n">results</span><span class="p">[(</span><span class="n">dimensions</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">)]</span> <span class="o">=</span> <span class="p">{</span>
                    <span class="s2">&quot;gaps&quot;</span><span class="p">:</span> <span class="n">gaps</span><span class="p">,</span>
                    <span class="s2">&quot;displacements&quot;</span><span class="p">:</span> <span class="n">displacements</span><span class="p">,</span>
                    <span class="s2">&quot;order&quot;</span><span class="p">:</span> <span class="n">fitted_order</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">dispersions</span><span class="p">,</span> <span class="n">gaps</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">floor</span><span class="p">),</span>
                <span class="p">}</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>

    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">))</span>

    <span class="c1"># ---- Panel (a): the gap, with the two candidate orders for reference ----</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">dispersions</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">dispersions</span><span class="p">)</span>
    <span class="n">reference</span> <span class="o">=</span> <span class="n">results</span><span class="p">[(</span><span class="n">geometries</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">[</span><span class="mi">0</span><span class="p">])][</span><span class="s2">&quot;gaps&quot;</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="k">for</span> <span class="n">exponent</span><span class="p">,</span> <span class="n">style</span> <span class="ow">in</span> <span class="p">((</span><span class="mi">2</span><span class="p">,</span> <span class="s2">&quot;--&quot;</span><span class="p">),</span> <span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="s2">&quot;:&quot;</span><span class="p">)):</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">dispersions</span><span class="p">,</span> <span class="n">reference</span> <span class="o">*</span> <span class="p">(</span><span class="n">dispersions</span> <span class="o">/</span> <span class="n">dispersions</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">**</span> <span class="n">exponent</span><span class="p">,</span>
            <span class="n">color</span><span class="o">=</span><span class="s2">&quot;0.5&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="n">style</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="sa">rf</span><span class="s2">&quot;$\varepsilon^</span><span class="si">{</span><span class="n">exponent</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="k">for</span> <span class="n">position</span><span class="p">,</span> <span class="n">dimensions</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">geometries</span><span class="p">):</span>
        <span class="k">for</span> <span class="n">n_clients</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">:</span>
            <span class="n">entry</span> <span class="o">=</span> <span class="n">results</span><span class="p">[(</span><span class="n">dimensions</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">)]</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
                <span class="n">dispersions</span><span class="p">,</span> <span class="n">entry</span><span class="p">[</span><span class="s2">&quot;gaps&quot;</span><span class="p">],</span> <span class="n">color</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">position</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span>
                <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mf">2.5</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.85</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
                <span class="n">label</span><span class="o">=</span><span class="p">(</span>
                    <span class="sa">rf</span><span class="s2">&quot;$\mathrm</span><span class="se">{{</span><span class="s2">St</span><span class="se">}}</span><span class="s2">(</span><span class="si">{</span><span class="n">dimensions</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s2">,</span><span class="si">{</span><span class="n">dimensions</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">)$&quot;</span>
                    <span class="k">if</span> <span class="n">n_clients</span> <span class="o">==</span> <span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="k">else</span> <span class="kc">None</span>
                <span class="p">),</span>
            <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;dispersion des clients $\varepsilon$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\|\mathrm</span><span class="si">{projavg}</span><span class="s2">-\mathrm</span><span class="si">{rlavg}</span><span class="s2">\|_F$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>

    <span class="c1"># ---- Panel (b): the fitted order, per configuration --------------------</span>
    <span class="c1"># A bar per (geometry, number of clients), against the O(eps^2) the</span>
    <span class="c1"># proposition claims: the panel exists to show the exponent is the same</span>
    <span class="c1"># everywhere, and that it is three.</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">labels</span><span class="p">,</span> <span class="n">orders</span><span class="p">,</span> <span class="n">colors</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[],</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">position</span><span class="p">,</span> <span class="n">dimensions</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">geometries</span><span class="p">):</span>
        <span class="k">for</span> <span class="n">n_clients</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">:</span>
            <span class="n">labels</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;(</span><span class="si">{</span><span class="n">dimensions</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s2">,</span><span class="si">{</span><span class="n">dimensions</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">)</span><span class="se">\n</span><span class="s2">$K=</span><span class="si">{</span><span class="n">n_clients</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">)</span>
            <span class="n">orders</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">results</span><span class="p">[(</span><span class="n">dimensions</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">)][</span><span class="s2">&quot;order&quot;</span><span class="p">])</span>
            <span class="n">colors</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">position</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">orders</span><span class="p">)),</span> <span class="n">orders</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">,</span> <span class="n">width</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">axhline</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C3&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
               <span class="n">label</span><span class="o">=</span><span class="sa">r</span><span class="s2">&quot;$O(\varepsilon^2)$ annoncé&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xticks</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">orders</span><span class="p">)))</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xticklabels</span><span class="p">(</span><span class="n">labels</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">4</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;ordre mesuré&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="c1"># ---- Digest ------------------------------------------------------------</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;n_repeats = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_repeats</span><span class="si">}</span><span class="s2">, seed = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;geometrie&#39;</span><span class="si">:</span><span class="s2">&gt;16s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;K&#39;</span><span class="si">:</span><span class="s2">&gt;4s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;ordre&#39;</span><span class="si">:</span><span class="s2">&gt;7s</span><span class="si">}</span><span class="s2"> &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;ecart a eps=1e-2&#39;</span><span class="si">:</span><span class="s2">&gt;17s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;deplacement&#39;</span><span class="si">:</span><span class="s2">&gt;13s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;rapport&#39;</span><span class="si">:</span><span class="s2">&gt;9s</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">index</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">dispersions</span><span class="o">.</span><span class="n">index</span><span class="p">(</span><span class="mf">1e-2</span><span class="p">)</span> <span class="k">if</span> <span class="mf">1e-2</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">dispersions</span> <span class="k">else</span> <span class="mi">0</span>
    <span class="k">for</span> <span class="n">dimensions</span> <span class="ow">in</span> <span class="n">geometries</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">n_clients</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">:</span>
            <span class="n">entry</span> <span class="o">=</span> <span class="n">results</span><span class="p">[(</span><span class="n">dimensions</span><span class="p">,</span> <span class="n">n_clients</span><span class="p">)]</span>
            <span class="n">gap</span> <span class="o">=</span> <span class="n">entry</span><span class="p">[</span><span class="s2">&quot;gaps&quot;</span><span class="p">][</span><span class="n">index</span><span class="p">]</span>
            <span class="n">displacement</span> <span class="o">=</span> <span class="n">entry</span><span class="p">[</span><span class="s2">&quot;displacements&quot;</span><span class="p">][</span><span class="n">index</span><span class="p">]</span>
            <span class="nb">print</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="sa">f</span><span class="s1">&#39;St(</span><span class="si">{</span><span class="n">dimensions</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s1">,</span><span class="si">{</span><span class="n">dimensions</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s1">)&#39;</span><span class="si">:</span><span class="s2">&gt;16s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">n_clients</span><span class="si">:</span><span class="s2">4d</span><span class="si">}</span><span class="s2"> &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">entry</span><span class="p">[</span><span class="s1">&#39;order&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">7.2f</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">gap</span><span class="si">:</span><span class="s2">17.3e</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">displacement</span><span class="si">:</span><span class="s2">13.3e</span><span class="si">}</span><span class="s2"> &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">gap</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">displacement</span><span class="si">:</span><span class="s2">9.2e</span><span class="si">}</span><span class="s2">&quot;</span>
            <span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">dispersions</span><span class="o">=</span><span class="n">dispersions</span><span class="p">,</span>
        <span class="n">geometries</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">geometries</span><span class="p">),</span>
        <span class="n">n_clients</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_clients</span><span class="p">),</span>
        <span class="n">gaps</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s2">&quot;gaps&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">key</span> <span class="ow">in</span> <span class="n">results</span><span class="p">]),</span>
        <span class="n">displacements</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s2">&quot;displacements&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">key</span> <span class="ow">in</span> <span class="n">results</span><span class="p">]),</span>
        <span class="n">orders</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">key</span><span class="p">][</span><span class="s2">&quot;order&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">key</span> <span class="ow">in</span> <span class="n">results</span><span class="p">]),</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">figure_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;stiefel_aggregation.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">figure_path</span><span class="p">,</span>
            <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span>
            <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">figure_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--dimensions</span><span class="param-type">int</span>
</div>
<p class="param-help">A Stiefel geometry as &#x27;n_in n_out&#x27;. Repeat the flag for several. Defaults to (40, 20), (128, 32) and (64, 60).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_clients</span><span class="param-type">int</span><span class="param-default">default <b>[2, 8, 32]</b></span>
</div>
<p class="param-help">Numbers of clients aggregated per round.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dispersions</span><span class="param-type">float</span><span class="param-default">default <b>[1.0, 0.1, 0.01, 0.001, 0.0001]</b></span>
</div>
<p class="param-help">Distances between a local weight and the global iterate. Stops at 1e-4: below that the gap is at the float64 rounding floor and carries no exponent.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_repeats</span><span class="param-type">int</span><span class="param-default">default <b>20</b></span>
</div>
<p class="param-help">Draws of the base point and the clients at each dispersion.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--floor</span><span class="param-type">float</span><span class="param-default">default <b>1e-13</b></span>
</div>
<p class="param-help">Gaps at or below this are treated as rounding, not signal.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/stiefel_aggregation</b></span>
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
<p class="param-help">Width of a single panel in the exported PGFPlots figure.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_height</span><span class="param-type">str</span><span class="param-default">default <b>4.6cm</b></span>
</div>
<p class="param-help">Height of a single panel in the exported PGFPlots figure.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--device</span><span class="param-type">str</span><span class="param-default">default <b>cpu</b></span>
</div>
<p class="param-help">Compute device: cpu or cuda.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">Base seed.</p>
</div>
</div>

## Config

`4-deeplearning/experiments/spdnet_stiefel_aggregation.yaml`

<a class="back-link" href="../../chapters/4-deeplearning/">← All experiments in 4 · Deep Learning</a>
