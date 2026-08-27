<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/4-deeplearning/">4 · Deep Learning</a>
<span class="sep">/</span>
<span class="here">spdnet_batchnorm_cost</span>
</nav>

# spdnet_batchnorm_cost

Time and retained autograd memory of the SPD batch-norm layer, hand-written backward against automatic differentiation

**Tags:** `deeplearning`  `spdnet`  `batchnorm`  `cost`

## Run

```sh
uv run python 4-deeplearning/batchnorm_cost/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/4-deeplearning/batchnorm_cost/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">357 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">4-deeplearning/batchnorm_cost/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># What the hand-written backward of the batch-norm layer costs, and what it saves.</span>
<span class="c1">#</span>
<span class="c1"># Volet 1 of sec:spdnet-batchnorm-resultats. The figures of the article that</span>
<span class="c1"># section reports were produced by code that was never committed — none of the</span>
<span class="c1"># five SPDnet repositories contains any timing or memory instrumentation — so</span>
<span class="c1"># this is a rewrite rather than a replay. It needs no data.</span>
<span class="c1">#</span>
<span class="c1"># On memory. The article measures torch.cuda.max_memory_allocated, which needs</span>
<span class="c1"># a GPU and mixes the quantity of interest with the allocator&#39;s behaviour. What</span>
<span class="c1"># prop:spdnet-grad-geo actually predicts is narrower and exactly measurable:</span>
<span class="c1"># automatic differentiation has to *retain the n_iterations iterates* of the</span>
<span class="c1"># fixed point eq:spdnet-geometrique-iteration, where the hand-written backward</span>
<span class="c1"># recomputes what it needs. torch.autograd.graph.saved_tensors_hooks intercepts</span>
<span class="c1"># every tensor the graph keeps alive, so summing their sizes measures precisely</span>
<span class="c1"># that, on any device. Peak allocator memory is reported too when running on</span>
<span class="c1"># CUDA, so the two can be compared against the article.</span>
<span class="c1">#</span>
<span class="c1"># On time. Measured, and reported without being oversold: on CPU the manual</span>
<span class="c1"># backward of the geometric mean is *slower* than autograd. The case for</span>
<span class="c1"># deriving by hand is memory and numerical robustness, not speed, and the</span>
<span class="c1"># chapter should say so plainly.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">time</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">yetanotherspdnet.nn.batchnorm</span><span class="w"> </span><span class="kn">import</span> <span class="n">BatchNormSPDMean</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">yetanotherspdnet.random.spd</span><span class="w"> </span><span class="kn">import</span> <span class="n">random_SPD</span>

<span class="c1"># The two means the article compares: the one with no closed form, whose</span>
<span class="c1"># backward has to unroll a fixed point, and the closed-form alternative that</span>
<span class="c1"># replaces it.</span>
<span class="n">MEANS</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">&quot;affine_invariant&quot;</span><span class="p">:</span> <span class="s2">&quot;géométrique&quot;</span><span class="p">,</span>
    <span class="s2">&quot;geometric_arithmetic_harmonic&quot;</span><span class="p">:</span> <span class="sa">r</span><span class="s2">&quot;\textsc</span><span class="si">{gah}</span><span class="s2">&quot;</span><span class="p">,</span>
<span class="p">}</span>


<span class="k">class</span><span class="w"> </span><span class="nc">GraphFootprint</span><span class="p">:</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Bytes retained by the autograd graph inside the ``with`` block.</span>

<span class="sd">    Counts every tensor saved for backward, once per storage: a tensor saved by</span>
<span class="sd">    several nodes is one allocation, and counting it twice would inflate the</span>
<span class="sd">    manual path and the automatic one differently.</span>
<span class="sd">    &quot;&quot;&quot;</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">total</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_seen</span> <span class="o">=</span> <span class="nb">set</span><span class="p">()</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__enter__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="k">def</span><span class="w"> </span><span class="nf">pack</span><span class="p">(</span><span class="n">tensor</span><span class="p">):</span>
            <span class="n">key</span> <span class="o">=</span> <span class="n">tensor</span><span class="o">.</span><span class="n">untyped_storage</span><span class="p">()</span><span class="o">.</span><span class="n">data_ptr</span><span class="p">()</span>
            <span class="k">if</span> <span class="n">key</span> <span class="ow">not</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">_seen</span><span class="p">:</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">_seen</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">total</span> <span class="o">+=</span> <span class="n">tensor</span><span class="o">.</span><span class="n">untyped_storage</span><span class="p">()</span><span class="o">.</span><span class="n">nbytes</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">tensor</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">_hooks</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">autograd</span><span class="o">.</span><span class="n">graph</span><span class="o">.</span><span class="n">saved_tensors_hooks</span><span class="p">(</span><span class="n">pack</span><span class="p">,</span> <span class="k">lambda</span> <span class="n">t</span><span class="p">:</span> <span class="n">t</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_hooks</span><span class="o">.</span><span class="fm">__enter__</span><span class="p">()</span>
        <span class="k">return</span> <span class="bp">self</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__exit__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="o">*</span><span class="n">exc</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_hooks</span><span class="o">.</span><span class="fm">__exit__</span><span class="p">(</span><span class="o">*</span><span class="n">exc</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">build</span><span class="p">(</span><span class="n">n_features</span><span class="p">,</span> <span class="n">mean_type</span><span class="p">,</span> <span class="n">use_autograd</span><span class="p">,</span> <span class="n">depth</span><span class="p">,</span> <span class="n">n_iterations</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;A stack of ``depth`` batch-norm layers, which is what the depth sweep varies.&quot;&quot;&quot;</span>
    <span class="n">options</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;n_iterations&quot;</span><span class="p">:</span> <span class="n">n_iterations</span><span class="p">}</span> <span class="k">if</span> <span class="n">mean_type</span> <span class="o">==</span> <span class="s2">&quot;affine_invariant&quot;</span> <span class="k">else</span> <span class="kc">None</span>
    <span class="k">return</span> <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">Sequential</span><span class="p">(</span>
        <span class="o">*</span><span class="p">[</span>
            <span class="n">BatchNormSPDMean</span><span class="p">(</span>
                <span class="n">n_features</span><span class="p">,</span>
                <span class="n">mean_type</span><span class="o">=</span><span class="n">mean_type</span><span class="p">,</span>
                <span class="n">mean_options</span><span class="o">=</span><span class="n">options</span><span class="p">,</span>
                <span class="n">use_autograd</span><span class="o">=</span><span class="n">use_autograd</span><span class="p">,</span>
                <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">,</span>
                <span class="n">dtype</span><span class="o">=</span><span class="n">dtype</span><span class="p">,</span>
            <span class="p">)</span>
            <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">depth</span><span class="p">)</span>
        <span class="p">]</span>
    <span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">one_measurement</span><span class="p">(</span>
    <span class="n">n_features</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">,</span> <span class="n">depth</span><span class="p">,</span> <span class="n">mean_type</span><span class="p">,</span> <span class="n">use_autograd</span><span class="p">,</span> <span class="n">args</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span>
<span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Time and retained memory of one forward and backward pass.&quot;&quot;&quot;</span>
    <span class="n">generator</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">Generator</span><span class="p">(</span><span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
    <span class="n">generator</span><span class="o">.</span><span class="n">manual_seed</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">random_SPD</span><span class="p">(</span>
        <span class="n">n_features</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">,</span> <span class="n">cond</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">cond</span><span class="p">,</span>
        <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">dtype</span><span class="p">,</span> <span class="n">generator</span><span class="o">=</span><span class="n">generator</span><span class="p">,</span>
    <span class="p">)</span><span class="o">.</span><span class="n">clone</span><span class="p">()</span><span class="o">.</span><span class="n">requires_grad_</span><span class="p">(</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">model</span> <span class="o">=</span> <span class="n">build</span><span class="p">(</span>
        <span class="n">n_features</span><span class="p">,</span> <span class="n">mean_type</span><span class="p">,</span> <span class="n">use_autograd</span><span class="p">,</span> <span class="n">depth</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_iterations</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span>
    <span class="p">)</span>

    <span class="c1"># One untimed pass first: the first call through a spectral layer pays for</span>
    <span class="c1"># lazily initialised buffers and, on CUDA, for the kernels themselves.</span>
    <span class="p">(</span><span class="n">model</span><span class="p">(</span><span class="n">data</span><span class="p">)</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
    <span class="n">data</span><span class="o">.</span><span class="n">grad</span> <span class="o">=</span> <span class="kc">None</span>

    <span class="k">if</span> <span class="n">device</span><span class="o">.</span><span class="n">type</span> <span class="o">==</span> <span class="s2">&quot;cuda&quot;</span><span class="p">:</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">reset_peak_memory_stats</span><span class="p">()</span>

    <span class="n">footprint</span> <span class="o">=</span> <span class="n">GraphFootprint</span><span class="p">()</span>
    <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_repeats</span><span class="p">):</span>
        <span class="n">data</span><span class="o">.</span><span class="n">grad</span> <span class="o">=</span> <span class="kc">None</span>
        <span class="k">with</span> <span class="n">footprint</span><span class="p">:</span>
            <span class="n">loss</span> <span class="o">=</span> <span class="p">(</span><span class="n">model</span><span class="p">(</span><span class="n">data</span><span class="p">)</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
        <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
    <span class="k">if</span> <span class="n">device</span><span class="o">.</span><span class="n">type</span> <span class="o">==</span> <span class="s2">&quot;cuda&quot;</span><span class="p">:</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">synchronize</span><span class="p">()</span>
    <span class="n">elapsed</span> <span class="o">=</span> <span class="p">(</span><span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">start</span><span class="p">)</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">n_repeats</span>

    <span class="k">return</span> <span class="p">{</span>
        <span class="s2">&quot;time&quot;</span><span class="p">:</span> <span class="n">elapsed</span><span class="p">,</span>
        <span class="c1"># The hooks fire on every repeat, so divide back out.</span>
        <span class="s2">&quot;graph_bytes&quot;</span><span class="p">:</span> <span class="n">footprint</span><span class="o">.</span><span class="n">total</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">n_repeats</span><span class="p">,</span>
        <span class="s2">&quot;peak_bytes&quot;</span><span class="p">:</span> <span class="p">(</span>
            <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">max_memory_allocated</span><span class="p">()</span> <span class="k">if</span> <span class="n">device</span><span class="o">.</span><span class="n">type</span> <span class="o">==</span> <span class="s2">&quot;cuda&quot;</span> <span class="k">else</span> <span class="nb">float</span><span class="p">(</span><span class="s2">&quot;nan&quot;</span><span class="p">)</span>
        <span class="p">),</span>
    <span class="p">}</span>


<span class="n">SWEEPS</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;taille de matrice&quot;</span><span class="p">,</span> <span class="k">lambda</span> <span class="n">v</span><span class="p">,</span> <span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_features</span><span class="o">=</span><span class="n">v</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">a</span><span class="o">.</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">depth</span><span class="o">=</span><span class="mi">1</span><span class="p">)),</span>
    <span class="s2">&quot;batch&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;taille de batch&quot;</span><span class="p">,</span> <span class="k">lambda</span> <span class="n">v</span><span class="p">,</span> <span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_features</span><span class="o">=</span><span class="n">a</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">v</span><span class="p">,</span> <span class="n">depth</span><span class="o">=</span><span class="mi">1</span><span class="p">)),</span>
    <span class="s2">&quot;depth&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;profondeur&quot;</span><span class="p">,</span> <span class="k">lambda</span> <span class="n">v</span><span class="p">,</span> <span class="n">a</span><span class="p">:</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_features</span><span class="o">=</span><span class="n">a</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span> <span class="n">batch_size</span><span class="o">=</span><span class="n">a</span><span class="o">.</span><span class="n">batch_size</span><span class="p">,</span> <span class="n">depth</span><span class="o">=</span><span class="n">v</span><span class="p">)),</span>
    <span class="s2">&quot;iterations&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;itérations du point fixe&quot;</span><span class="p">,</span> <span class="kc">None</span><span class="p">),</span>
<span class="p">}</span>


<span class="k">def</span><span class="w"> </span><span class="nf">run_sweep</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">values</span><span class="p">,</span> <span class="n">args</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Every (mean, differentiation) combination along one axis.&quot;&quot;&quot;</span>
    <span class="n">records</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">values</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">mean_type</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">use_autograd</span> <span class="ow">in</span> <span class="p">(</span><span class="kc">False</span><span class="p">,</span> <span class="kc">True</span><span class="p">):</span>
                <span class="k">if</span> <span class="n">name</span> <span class="o">==</span> <span class="s2">&quot;iterations&quot;</span><span class="p">:</span>
                    <span class="c1"># Only the geometric mean has a fixed point to unroll; the</span>
                    <span class="c1"># closed-form means do not depend on this axis at all.</span>
                    <span class="k">if</span> <span class="n">mean_type</span> <span class="o">!=</span> <span class="s2">&quot;affine_invariant&quot;</span><span class="p">:</span>
                        <span class="k">continue</span>
                    <span class="n">shape</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span>
                        <span class="n">n_features</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span>
                        <span class="n">batch_size</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">batch_size</span><span class="p">,</span>
                        <span class="n">depth</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
                    <span class="p">)</span>
                    <span class="n">args</span><span class="o">.</span><span class="n">n_iterations</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">value</span><span class="p">)</span>
                <span class="k">else</span><span class="p">:</span>
                    <span class="n">shape</span> <span class="o">=</span> <span class="n">SWEEPS</span><span class="p">[</span><span class="n">name</span><span class="p">][</span><span class="mi">1</span><span class="p">](</span><span class="n">value</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
                <span class="n">measurement</span> <span class="o">=</span> <span class="n">one_measurement</span><span class="p">(</span>
                    <span class="n">mean_type</span><span class="o">=</span><span class="n">mean_type</span><span class="p">,</span> <span class="n">use_autograd</span><span class="o">=</span><span class="n">use_autograd</span><span class="p">,</span>
                    <span class="n">args</span><span class="o">=</span><span class="n">args</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">dtype</span><span class="p">,</span> <span class="o">**</span><span class="n">shape</span><span class="p">,</span>
                <span class="p">)</span>
                <span class="n">measurement</span><span class="o">.</span><span class="n">update</span><span class="p">(</span>
                    <span class="n">value</span><span class="o">=</span><span class="n">value</span><span class="p">,</span> <span class="n">mean_type</span><span class="o">=</span><span class="n">mean_type</span><span class="p">,</span> <span class="n">use_autograd</span><span class="o">=</span><span class="n">use_autograd</span>
                <span class="p">)</span>
                <span class="n">records</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">measurement</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">records</span>


<span class="k">def</span><span class="w"> </span><span class="nf">draw</span><span class="p">(</span><span class="n">records</span><span class="p">,</span> <span class="n">axis_label</span><span class="p">,</span> <span class="n">args</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Two panels: retained memory on the left, wall time on the right.&quot;&quot;&quot;</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">))</span>
    <span class="n">values</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">({</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;value&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">})</span>

    <span class="k">for</span> <span class="n">position</span><span class="p">,</span> <span class="n">mean_type</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">):</span>
        <span class="k">for</span> <span class="n">use_autograd</span><span class="p">,</span> <span class="n">style</span><span class="p">,</span> <span class="n">marker</span> <span class="ow">in</span> <span class="p">((</span><span class="kc">False</span><span class="p">,</span> <span class="s2">&quot;-&quot;</span><span class="p">,</span> <span class="s2">&quot;o&quot;</span><span class="p">),</span> <span class="p">(</span><span class="kc">True</span><span class="p">,</span> <span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="s2">&quot;s&quot;</span><span class="p">)):</span>
            <span class="n">selected</span> <span class="o">=</span> <span class="p">[</span>
                <span class="n">record</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span>
                <span class="k">if</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;mean_type&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">mean_type</span>
                <span class="ow">and</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;use_autograd&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">use_autograd</span>
            <span class="p">]</span>
            <span class="k">if</span> <span class="ow">not</span> <span class="n">selected</span><span class="p">:</span>
                <span class="k">continue</span>
            <span class="n">selected</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">record</span><span class="p">:</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;value&quot;</span><span class="p">])</span>
            <span class="n">label</span> <span class="o">=</span> <span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">MEANS</span><span class="p">[</span><span class="n">mean_type</span><span class="p">]</span><span class="si">}</span><span class="s2">, &quot;</span>
                <span class="o">+</span> <span class="p">(</span><span class="s2">&quot;autograd&quot;</span> <span class="k">if</span> <span class="n">use_autograd</span> <span class="k">else</span> <span class="s2">&quot;manuel&quot;</span><span class="p">)</span>
            <span class="p">)</span>
            <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
                <span class="p">[</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;value&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">selected</span><span class="p">],</span>
                <span class="p">[</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;graph_bytes&quot;</span><span class="p">]</span> <span class="o">/</span> <span class="mi">2</span><span class="o">**</span><span class="mi">20</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">selected</span><span class="p">],</span>
                <span class="n">color</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">position</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="n">style</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="n">marker</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
                <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.3</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">label</span><span class="p">,</span>
            <span class="p">)</span>
            <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
                <span class="p">[</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;value&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">selected</span><span class="p">],</span>
                <span class="p">[</span><span class="mf">1e3</span> <span class="o">*</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;time&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">selected</span><span class="p">],</span>
                <span class="n">color</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">position</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="n">style</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="n">marker</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
                <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.3</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">label</span><span class="p">,</span>
            <span class="p">)</span>

    <span class="k">for</span> <span class="n">axis</span><span class="p">,</span> <span class="n">ylabel</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="p">(</span><span class="s2">&quot;mémoire retenue (Mio)&quot;</span><span class="p">,</span> <span class="s2">&quot;temps (ms)&quot;</span><span class="p">)):</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="n">axis_label</span><span class="p">)</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="n">ylabel</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
    <span class="k">return</span> <span class="n">fig</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Time and memory of the batch-norm layer, hand-written backward against &quot;</span>
        <span class="s2">&quot;automatic differentiation.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--sweep&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;size&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="nb">sorted</span><span class="p">(</span><span class="n">SWEEPS</span><span class="p">),</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Which axis to vary. &#39;iterations&#39; is the mechanism behind the &quot;</span>
             <span class="s2">&quot;other three and is not in the article.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--values&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Points along the swept axis. Defaults per sweep: 8..512 for &quot;</span>
             <span class="s2">&quot;size and batch, 1..32 for depth, 1..20 for iterations.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--means&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="nb">sorted</span><span class="p">(</span><span class="n">MEANS</span><span class="p">),</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Means to compare. The article compares the geometric one with GAH.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Matrix size, when it is not the swept axis.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--batch_size&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Batch size, when it is not the swept axis.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_iterations&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Fixed-point iterations of the geometric mean, when not swept.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--cond&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Condition number of the drawn matrices, as in the article.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_repeats&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Passes averaged at each point.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/batchnorm_cost&quot;</span><span class="p">,</span>
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
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Compute device: cpu or cuda. On cuda the allocator peak is &quot;</span>
             <span class="s2">&quot;reported alongside the retained-graph measurement.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Base seed.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">device</span> <span class="o">==</span> <span class="s2">&quot;mps&quot;</span><span class="p">:</span>
        <span class="k">raise</span> <span class="ne">SystemExit</span><span class="p">(</span>
            <span class="s2">&quot;MPS cannot run this study: no float64, and linalg.eigh is not &quot;</span>
            <span class="s2">&quot;implemented for it. Use --device cpu or --device cuda.&quot;</span>
        <span class="p">)</span>
    <span class="n">device</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">device</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">device</span><span class="p">)</span>
    <span class="n">dtype</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">float64</span>

    <span class="n">defaults</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;size&quot;</span><span class="p">:</span> <span class="p">[</span><span class="mi">8</span><span class="p">,</span> <span class="mi">16</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">64</span><span class="p">,</span> <span class="mi">128</span><span class="p">,</span> <span class="mi">256</span><span class="p">,</span> <span class="mi">512</span><span class="p">],</span>
        <span class="s2">&quot;batch&quot;</span><span class="p">:</span> <span class="p">[</span><span class="mi">8</span><span class="p">,</span> <span class="mi">16</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">64</span><span class="p">,</span> <span class="mi">128</span><span class="p">,</span> <span class="mi">256</span><span class="p">,</span> <span class="mi">512</span><span class="p">],</span>
        <span class="s2">&quot;depth&quot;</span><span class="p">:</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">4</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">16</span><span class="p">,</span> <span class="mi">32</span><span class="p">],</span>
        <span class="s2">&quot;iterations&quot;</span><span class="p">:</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">20</span><span class="p">],</span>
    <span class="p">}</span>
    <span class="n">values</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">values</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">values</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span> <span class="k">else</span> <span class="n">defaults</span><span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">sweep</span><span class="p">]</span>
    <span class="n">values</span> <span class="o">=</span> <span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">value</span><span class="p">)</span> <span class="k">for</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">values</span><span class="p">]</span>

    <span class="n">records</span> <span class="o">=</span> <span class="n">run_sweep</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">sweep</span><span class="p">,</span> <span class="n">values</span><span class="p">,</span> <span class="n">args</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="p">)</span>

    <span class="n">axis_label</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">SWEEPS</span><span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">sweep</span><span class="p">]</span>
    <span class="n">figure</span> <span class="o">=</span> <span class="n">draw</span><span class="p">(</span><span class="n">records</span><span class="p">,</span> <span class="n">axis_label</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>

    <span class="c1"># ---- Digest ------------------------------------------------------------</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;sweep = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">sweep</span><span class="si">}</span><span class="s2">, device = </span><span class="si">{</span><span class="n">device</span><span class="si">}</span><span class="s2">, cond = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">cond</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;n_repeats = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_repeats</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;valeur&#39;</span><span class="si">:</span><span class="s2">&gt;8s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;moyenne&#39;</span><span class="si">:</span><span class="s2">&gt;32s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;mem manuel&#39;</span><span class="si">:</span><span class="s2">&gt;12s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;mem auto&#39;</span><span class="si">:</span><span class="s2">&gt;11s</span><span class="si">}</span><span class="s2"> &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;rapport&#39;</span><span class="si">:</span><span class="s2">&gt;8s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;t manuel&#39;</span><span class="si">:</span><span class="s2">&gt;10s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;t auto&#39;</span><span class="si">:</span><span class="s2">&gt;9s</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">values</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">mean_type</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">:</span>
            <span class="n">pair</span> <span class="o">=</span> <span class="p">{</span>
                <span class="n">record</span><span class="p">[</span><span class="s2">&quot;use_autograd&quot;</span><span class="p">]:</span> <span class="n">record</span>
                <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span>
                <span class="k">if</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;value&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">value</span> <span class="ow">and</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;mean_type&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">mean_type</span>
            <span class="p">}</span>
            <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">pair</span><span class="p">)</span> <span class="o">!=</span> <span class="mi">2</span><span class="p">:</span>
                <span class="k">continue</span>
            <span class="n">manual</span><span class="p">,</span> <span class="n">auto</span> <span class="o">=</span> <span class="n">pair</span><span class="p">[</span><span class="kc">False</span><span class="p">],</span> <span class="n">pair</span><span class="p">[</span><span class="kc">True</span><span class="p">]</span>
            <span class="nb">print</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">value</span><span class="si">:</span><span class="s2">8d</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">mean_type</span><span class="si">:</span><span class="s2">&gt;32s</span><span class="si">}</span><span class="s2"> &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">manual</span><span class="p">[</span><span class="s1">&#39;graph_bytes&#39;</span><span class="p">]</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="mi">2</span><span class="o">**</span><span class="mi">20</span><span class="si">:</span><span class="s2">11.2f</span><span class="si">}</span><span class="s2">M &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">auto</span><span class="p">[</span><span class="s1">&#39;graph_bytes&#39;</span><span class="p">]</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="mi">2</span><span class="o">**</span><span class="mi">20</span><span class="si">:</span><span class="s2">10.2f</span><span class="si">}</span><span class="s2">M &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">auto</span><span class="p">[</span><span class="s1">&#39;graph_bytes&#39;</span><span class="p">]</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">manual</span><span class="p">[</span><span class="s1">&#39;graph_bytes&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">7.2f</span><span class="si">}</span><span class="s2">x &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="mf">1e3</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">manual</span><span class="p">[</span><span class="s1">&#39;time&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">9.1f</span><span class="si">}</span><span class="s2">ms </span><span class="si">{</span><span class="mf">1e3</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">auto</span><span class="p">[</span><span class="s1">&#39;time&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">8.1f</span><span class="si">}</span><span class="s2">ms&quot;</span>
            <span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">sweep</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">sweep</span><span class="p">,</span>
        <span class="n">values</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;value&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">mean_types</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;mean_type&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">use_autograd</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;use_autograd&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">graph_bytes</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;graph_bytes&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">peak_bytes</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;peak_bytes&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">times</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;time&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">figure_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
            <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="sa">f</span><span class="s2">&quot;batchnorm_cost_</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">sweep</span><span class="si">}</span><span class="s2">.tex&quot;</span>
        <span class="p">)</span>
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
<span class="param-flag">--sweep</span><span class="param-type">str</span><span class="param-default">default <b>size</b></span>
</div>
<p class="param-help">Which axis to vary. &#x27;iterations&#x27; is the mechanism behind the other three and is not in the article.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--values</span><span class="param-type">float</span>
</div>
<p class="param-help">Points along the swept axis. Defaults per sweep: 8..512 for size and batch, 1..32 for depth, 1..20 for iterations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--means</span><span class="param-type">str</span>
</div>
<p class="param-help">Means to compare. The article compares the geometric one with GAH.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_features</span><span class="param-type">int</span><span class="param-default">default <b>64</b></span>
</div>
<p class="param-help">Matrix size, when it is not the swept axis.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--batch_size</span><span class="param-type">int</span><span class="param-default">default <b>64</b></span>
</div>
<p class="param-help">Batch size, when it is not the swept axis.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_iterations</span><span class="param-type">int</span><span class="param-default">default <b>5</b></span>
</div>
<p class="param-help">Fixed-point iterations of the geometric mean, when not swept.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--cond</span><span class="param-type">float</span><span class="param-default">default <b>100000.0</b></span>
</div>
<p class="param-help">Condition number of the drawn matrices, as in the article.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_repeats</span><span class="param-type">int</span><span class="param-default">default <b>5</b></span>
</div>
<p class="param-help">Passes averaged at each point.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/batchnorm_cost</b></span>
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
<p class="param-help">Compute device: cpu or cuda. On cuda the allocator peak is reported alongside the retained-graph measurement.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">Base seed.</p>
</div>
</div>

## Config

`4-deeplearning/experiments/spdnet_batchnorm_cost.yaml`

<a class="back-link" href="../../chapters/4-deeplearning/">← All experiments in 4 · Deep Learning</a>
