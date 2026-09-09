<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sonar_tyler_conv</span>
</nav>

# sonar_tyler_conv

2TYL fixed-point convergence — relative Frobenius deviation vs iteration

**Tags:** `sonar`  `estimation`  `monte-carlo`  `convergence`

## Run

```sh
uv run python 2-detection/sonar_experiments/mc_simulations/tyler_convergence.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--n-trials 1000</code><br>
  <code>--seed 42</code><br>
  <code>--backend numpy</code><br>
  <code>--m 64</code><br>
  <code>--beta 0.0003</code><br>
  <code>--rho1 0.4</code><br>
  <code>--rho2 0.9</code><br>
  <code>--theta1 45.0</code><br>
  <code>--theta2 45.0</code><br>
  <code>--k-dist gaussian</code><br>
  <code>--nu 0.5</code><br>
  <code>--snr-min -25.0</code><br>
  <code>--snr-max 5.0</code><br>
  <code>--n-snr 150</code><br>
  <code>--pfa 0.01</code><br>
  <code>--iter-max 500</code><br>
  <span class="mn-date">9512517 · 2026-08-25</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sonar_experiments/mc_simulations/tyler_convergence.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">112 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sonar_experiments/mc_simulations/tyler_convergence.py</p>
<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/env python</span>
<span class="sd">&quot;&quot;&quot;2TYL fixed-point convergence: relative Frobenius deviation vs iteration.</span>

<span class="sd">Tracks ||M̂^(k) - M̂^(k-1)||_F / ||M̂^(k-1)||_F for each iteration of the</span>
<span class="sd">two-array Tyler MLE, averaged over n_trials secondary datasets.  Compares</span>
<span class="sd">behaviour under Gaussian and K-distributed clutter and for different K values.</span>

<span class="sd">Backend selection:</span>
<span class="sd">  numpy     → single batched numpy pass (already parallelised across trials)</span>
<span class="sd">  all other → move secondary data to device, run Tyler iteration on device</span>
<span class="sd">&quot;&quot;&quot;</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="n">MCResultExporter</span><span class="p">,</span> <span class="n">make_mc_parser</span><span class="p">,</span> <span class="n">timed_run</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sonar</span><span class="w"> </span><span class="kn">import</span> <span class="n">mc</span> <span class="k">as</span> <span class="n">smc</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sonar</span><span class="w"> </span><span class="kn">import</span> <span class="n">simulation</span> <span class="k">as</span> <span class="n">sim</span>

<span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Numpy path (already batched across trials — no Pool needed)</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_run_numpy</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">K_conv</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">iter_max</span><span class="p">,</span> <span class="n">seed</span><span class="p">):</span>
    <span class="n">X_sec</span> <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">generate_secondary_data</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">K_conv</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">seed</span><span class="p">)</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Secondary data generated: </span><span class="si">{</span><span class="n">X_sec</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">smc</span><span class="o">.</span><span class="n">tyler_relative_deviations</span><span class="p">(</span><span class="n">X_sec</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">iter_max</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="s2">&quot;numpy&quot;</span><span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Batched path (non-numpy backends)</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_run_batched</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">K_conv</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">iter_max</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">seed</span><span class="p">):</span>
    <span class="n">X_sec</span> <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">generate_secondary_data</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">K_conv</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">seed</span><span class="p">)</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Secondary data generated: </span><span class="si">{</span><span class="n">X_sec</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">; moving to </span><span class="si">{</span><span class="n">backend</span><span class="si">}</span><span class="s2">...&quot;</span><span class="p">)</span>
    <span class="n">X_dev</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">X_sec</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span>
        <span class="n">smc</span><span class="o">.</span><span class="n">tyler_relative_deviations</span><span class="p">(</span><span class="n">X_dev</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">iter_max</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>
    <span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Entry point</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span><span class="vm">__doc__</span><span class="p">)</span>
    <span class="n">smc</span><span class="o">.</span><span class="n">add_mc_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">smc</span><span class="o">.</span><span class="n">add_tyler_conv_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">smc</span><span class="o">.</span><span class="n">apply_debug</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">logger</span><span class="p">)</span>

    <span class="n">logging</span><span class="o">.</span><span class="n">basicConfig</span><span class="p">(</span><span class="n">level</span><span class="o">=</span><span class="n">logging</span><span class="o">.</span><span class="n">INFO</span><span class="p">,</span> <span class="nb">format</span><span class="o">=</span><span class="s2">&quot;</span><span class="si">%(levelname)s</span><span class="s2"> </span><span class="si">%(message)s</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">m</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">m</span>
    <span class="n">K</span> <span class="o">=</span> <span class="n">smc</span><span class="o">.</span><span class="n">resolve_K</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>
    <span class="n">K_conv</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">K_conv</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">K_conv</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span> <span class="k">else</span> <span class="n">K</span>
    <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span> <span class="o">=</span> <span class="n">smc</span><span class="o">.</span><span class="n">clutter_params</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>
    <span class="n">iter_max</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">iter_max</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;2TYL convergence: m=</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">, K_conv=</span><span class="si">{</span><span class="n">K_conv</span><span class="si">}</span><span class="s2">, clutter=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">clutter</span><span class="si">}</span><span class="s2">, &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;n_trials=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">, iter_max=</span><span class="si">{</span><span class="n">iter_max</span><span class="si">}</span><span class="s2">, backend=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">M</span> <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">make_sonar_covariance</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">beta</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho1</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho2</span><span class="p">)</span>

    <span class="n">deviations</span><span class="p">,</span> <span class="n">elapsed</span> <span class="o">=</span> <span class="n">timed_run</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_numpy</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">K_conv</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span>
                           <span class="n">iter_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">),</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_batched</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">K_conv</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span>
                             <span class="n">iter_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">),</span>
    <span class="p">)</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Done in </span><span class="si">{</span><span class="n">elapsed</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2">s&quot;</span><span class="p">)</span>

    <span class="n">iterations</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">iter_max</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>

    <span class="n">export_stats</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;iterations&quot;</span><span class="p">:</span>      <span class="n">iterations</span><span class="p">,</span>
        <span class="s2">&quot;deviations_mean&quot;</span><span class="p">:</span> <span class="n">deviations</span><span class="p">,</span>
        <span class="s2">&quot;K&quot;</span><span class="p">:</span>               <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">K_conv</span><span class="p">]),</span>
        <span class="s2">&quot;m&quot;</span><span class="p">:</span>               <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">m</span><span class="p">]),</span>
    <span class="p">}</span>

    <span class="n">clutter_tag</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">clutter</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">clutter</span> <span class="o">==</span> <span class="s2">&quot;gaussian&quot;</span> <span class="k">else</span> <span class="sa">f</span><span class="s2">&quot;k_nu</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">}</span><span class="s2">&quot;</span>
    <span class="n">stem_tag</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">&quot;m</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">_K</span><span class="si">{</span><span class="n">K_conv</span><span class="si">}</span><span class="s2">_</span><span class="si">{</span><span class="n">clutter_tag</span><span class="si">}</span><span class="s2">_n</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">&quot;</span>

    <span class="n">exporter</span> <span class="o">=</span> <span class="n">MCResultExporter</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">Path</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span> <span class="n">stem_tag</span><span class="p">,</span>
                                <span class="n">plot_template</span><span class="o">=</span><span class="n">smc</span><span class="o">.</span><span class="n">_MC_TYLER_CONV_TEMPLATE</span><span class="p">)</span>
    <span class="n">title</span> <span class="o">=</span> <span class="p">(</span><span class="sa">f</span><span class="s2">&quot;2TYL convergence (m=</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">, K=</span><span class="si">{</span><span class="n">K_conv</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">clutter</span><span class="si">}</span><span class="s2"> clutter)&quot;</span><span class="p">)</span>
    <span class="n">exporter</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="n">export_stats</span><span class="p">,</span> <span class="s2">&quot;sonar_tyler_conv&quot;</span><span class="p">,</span> <span class="n">elapsed</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">tol</span> <span class="ow">in</span> <span class="p">[</span><span class="mf">1e-3</span><span class="p">,</span> <span class="mf">1e-6</span><span class="p">]:</span>
        <span class="n">below</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">deviations</span> <span class="o">&lt;</span> <span class="n">tol</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
        <span class="k">if</span> <span class="n">below</span><span class="o">.</span><span class="n">size</span><span class="p">:</span>
            <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  tol=</span><span class="si">{</span><span class="n">tol</span><span class="si">:</span><span class="s2">.0e</span><span class="si">}</span><span class="s2">: converged at iteration </span><span class="si">{</span><span class="n">below</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">+</span><span class="mi">1</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  tol=</span><span class="si">{</span><span class="n">tol</span><span class="si">:</span><span class="s2">.0e</span><span class="si">}</span><span class="s2">: did not converge within </span><span class="si">{</span><span class="n">iter_max</span><span class="si">}</span><span class="s2"> iters&quot;</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--iter-max</span><span class="param-type">int</span><span class="param-default">default <b>500</b></span>
</div>
<p class="param-help">Number of fixed-point iterations to track (default 500).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--K-conv</span><span class="param-type">int</span>
</div>
<p class="param-help">Secondary samples for convergence experiment (default = K_secondary).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--m</span><span class="param-type">int</span><span class="param-default">default <b>64</b></span>
</div>
<p class="param-help">Per-array dimension (total = 2m, default 64).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--beta</span><span class="param-type">float</span><span class="param-default">default <b>0.0003</b></span>
</div>
<p class="param-help">Covariance scale factor beta (default 3e-4).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho1</span><span class="param-type">float</span><span class="param-default">default <b>0.4</b></span>
</div>
<p class="param-help">Array-1 correlation coefficient rho1 (default 0.4).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho2</span><span class="param-type">float</span><span class="param-default">default <b>0.9</b></span>
</div>
<p class="param-help">Array-2 correlation coefficient rho2 (default 0.9).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--K</span><span class="param-type">int</span>
</div>
<p class="param-help">Secondary data size K (default = 2*2m = 4m).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--theta1</span><span class="param-type">float</span><span class="param-default">default <b>45.0</b></span>
</div>
<p class="param-help">Array-1 steering angle in degrees (default 45).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--theta2</span><span class="param-type">float</span><span class="param-default">default <b>45.0</b></span>
</div>
<p class="param-help">Array-2 steering angle in degrees (default 45).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--gaussian</span><span class="param-default">default <b>gaussian</b></span>
</div>
<p class="param-help">Gaussian clutter (default).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--k-dist</span>
</div>
<p class="param-help">K-distributed clutter.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--nu</span><span class="param-type">float</span><span class="param-default">default <b>0.5</b></span>
</div>
<p class="param-help">K-distribution shape parameter nu (default 0.5).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--snr-min</span><span class="param-type">float</span><span class="param-default">default <b>-25.0</b></span>
</div>
<p class="param-help">Minimum SNR in dB (default -25).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--snr-max</span><span class="param-type">float</span><span class="param-default">default <b>5.0</b></span>
</div>
<p class="param-help">Maximum SNR in dB (default 5).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-snr</span><span class="param-type">int</span><span class="param-default">default <b>150</b></span>
</div>
<p class="param-help">Number of SNR values (default 150).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--pfa</span><span class="param-type">float</span><span class="param-default">default <b>0.01</b></span>
</div>
<p class="param-help">Nominal PFA for PD curves (default 1e-2).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--debug</span><span class="param-type">flag</span>
</div>
<p class="param-help">Tiny configuration, to validate the pipeline in seconds. Results are NOT publication grade: see apply_debug for what is reduced.</p>
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

## Config

`2-detection/experiments/sonar/sonar_tyler_conv.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
