<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sar_mc_power</span>
</nav>

# sar_mc_power

Detection power against the number of dates, for the four change detectors, offline and online

**Tags:** `detection`  `kronecker`  `puissance`  `H1`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_power_detectors.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sar_experiments/mc_simulations/mc_power_detectors.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">289 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sar_experiments/mc_simulations/mc_power_detectors.py</p>
<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/env python</span>
<span class="sd">&quot;&quot;&quot;Power against T for the four change detectors, offline and online.</span>

<span class="sd">Replaces the three ROC planes of Mian et al., Signal Processing 224 (2024)</span>
<span class="sd">(Figures 4 to 6) by the single quantity the chapter argues about: how fast the</span>
<span class="sd">online detectors catch up with their offline counterparts as the time series</span>
<span class="sd">grows, at a fixed false alarm probability.</span>

<span class="sd">Four detectors, in the notation of the paper:</span>
<span class="sd">  SG     -- offline scale-and-shape GLRT, unstructured  (Lambda_SG)</span>
<span class="sd">  K-SG   -- offline GLRT with Kronecker structure       (Lambda_K-SG)</span>
<span class="sd">  SG-O   -- recursive counterpart of SG                 (Lambda_SG-O)</span>
<span class="sd">  K-SG-O -- recursive counterpart of K-SG               (Lambda_K-SG-O)</span>
<span class="sd">and optionally G, the Gaussian covariance equality GLRT, as a baseline.</span>

<span class="sd">Data follow the same setup as the ROC configuration of the released code:</span>
<span class="sd">Toeplitz factors of unit determinant, rho changing at T/2 under H1, patches of</span>
<span class="sd">n = a*b+1 samples. Run once with --texture k (K-distributed, nu=1) and once</span>
<span class="sd">with --texture gaussian to expose the result the paper found surprising: the</span>
<span class="sd">online detectors converge more slowly in the Gaussian case than in the</span>
<span class="sd">heterogeneous one.</span>

<span class="sd">Thresholds are set per detector and per T at the (1 - PFA) quantile of the H0</span>
<span class="sd">statistics, so every curve is read at the same false alarm rate.</span>

<span class="sd">Backend selection:</span>
<span class="sd">  numpy     → multiprocessing.Pool, one trial per worker</span>
<span class="sd">  all other → trials in leading batch dim, single-pass on device</span>
<span class="sd">&quot;&quot;&quot;</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">multiprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Pool</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.simulation</span><span class="w"> </span><span class="kn">import</span> <span class="n">T_vec_logspace</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">Progress</span><span class="p">,</span>
    <span class="n">MCResultExporter</span><span class="p">,</span>
    <span class="n">init_logging</span><span class="p">,</span>
    <span class="n">make_mc_parser</span><span class="p">,</span>
    <span class="n">maybe_empty_cache</span><span class="p">,</span>
    <span class="n">timed_run</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.simulation</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">make_ab_toeplitz</span><span class="p">,</span>
    <span class="n">generate_kronecker_data</span><span class="p">,</span>
    <span class="n">generate_kronecker_data_h1</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.detectors</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">DeterministicCompoundGaussianGLRT</span><span class="p">,</span>
    <span class="n">GaussianGLRT</span><span class="p">,</span>
    <span class="n">ScaleAndShapeKroneckerGLRT</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.detection_online</span><span class="w"> </span><span class="kn">import</span> <span class="n">OnlineDCGDetector</span><span class="p">,</span> <span class="n">OnlineKroneckerDetector</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.mc</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">_MC_PLOT_TEMPLATE_POWER_MULTI</span><span class="p">,</span>
    <span class="n">add_mc_args</span><span class="p">,</span>
    <span class="n">add_mc_h1_args</span><span class="p">,</span>
    <span class="n">finish_power_multi</span><span class="p">,</span>
    <span class="n">online_single_pass</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Detector construction</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_build_detectors</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Return {name: (kind, detector)} with kind in {&quot;offline&quot;, &quot;online&quot;}.&quot;&quot;&quot;</span>
    <span class="n">dets</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;SG&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;offline&quot;</span><span class="p">,</span> <span class="n">DeterministicCompoundGaussianGLRT</span><span class="p">(</span>
            <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;tol&quot;</span><span class="p">],</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;iter_max&quot;</span><span class="p">])),</span>
        <span class="s2">&quot;K-SG&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;offline&quot;</span><span class="p">,</span> <span class="n">ScaleAndShapeKroneckerGLRT</span><span class="p">(</span>
            <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;tol&quot;</span><span class="p">],</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;iter_max&quot;</span><span class="p">])),</span>
        <span class="s2">&quot;SG-O&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;online&quot;</span><span class="p">,</span> <span class="n">OnlineDCGDetector</span><span class="p">(</span>
            <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">,</span> <span class="n">h0_step_rule</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;step_rule&quot;</span><span class="p">],</span>
            <span class="n">h0_alpha_0</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;alpha_0&quot;</span><span class="p">],</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;iter_max&quot;</span><span class="p">],</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;tol&quot;</span><span class="p">])),</span>
        <span class="s2">&quot;K-SG-O&quot;</span><span class="p">:</span> <span class="p">(</span><span class="s2">&quot;online&quot;</span><span class="p">,</span> <span class="n">OnlineKroneckerDetector</span><span class="p">(</span>
            <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">h0_step_rule</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;step_rule&quot;</span><span class="p">],</span> <span class="n">h0_init_mode</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;init_mode&quot;</span><span class="p">],</span>
            <span class="n">h0_alpha_0</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;alpha_0&quot;</span><span class="p">],</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;iter_max&quot;</span><span class="p">],</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;tol&quot;</span><span class="p">])),</span>
    <span class="p">}</span>
    <span class="k">if</span> <span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;with_gaussian&quot;</span><span class="p">]:</span>
        <span class="n">dets</span><span class="p">[</span><span class="s2">&quot;G&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="s2">&quot;offline&quot;</span><span class="p">,</span> <span class="n">GaussianGLRT</span><span class="p">(</span><span class="n">backend</span><span class="p">))</span>
    <span class="n">keep</span> <span class="o">=</span> <span class="n">cfg</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">&quot;detectors&quot;</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">keep</span><span class="p">:</span>
        <span class="n">missing</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">keep</span><span class="p">)</span> <span class="o">-</span> <span class="nb">set</span><span class="p">(</span><span class="n">dets</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">missing</span><span class="p">:</span>
            <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;unknown detector(s) </span><span class="si">{</span><span class="nb">sorted</span><span class="p">(</span><span class="n">missing</span><span class="p">)</span><span class="si">}</span><span class="s2">; have </span><span class="si">{</span><span class="nb">sorted</span><span class="p">(</span><span class="n">dets</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
        <span class="n">dets</span> <span class="o">=</span> <span class="p">{</span><span class="n">k</span><span class="p">:</span> <span class="n">v</span> <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">dets</span><span class="o">.</span><span class="n">items</span><span class="p">()</span> <span class="k">if</span> <span class="n">k</span> <span class="ow">in</span> <span class="n">keep</span><span class="p">}</span>
    <span class="k">return</span> <span class="n">dets</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_statistics</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Statistics of every detector at every T checkpoint, for one data set.</span>

<span class="sd">    data : (..., T_max, N, p). Offline detectors are evaluated on the first T</span>
<span class="sd">    dates; online ones are streamed once and read at each checkpoint.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">dets</span> <span class="o">=</span> <span class="n">_build_detectors</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">out</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="p">(</span><span class="n">kind</span><span class="p">,</span> <span class="n">det</span><span class="p">)</span> <span class="ow">in</span> <span class="n">dets</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="k">if</span> <span class="n">kind</span> <span class="o">==</span> <span class="s2">&quot;online&quot;</span><span class="p">:</span>
            <span class="n">raw</span> <span class="o">=</span> <span class="n">online_single_pass</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">det</span><span class="p">)</span>
            <span class="n">out</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span><span class="n">T</span><span class="p">:</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">v</span><span class="p">)</span> <span class="k">for</span> <span class="n">T</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">raw</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">out</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span><span class="n">T</span><span class="p">:</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">det</span><span class="o">.</span><span class="n">compute</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="o">...</span><span class="p">,</span> <span class="p">:</span><span class="n">T</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:]))</span> <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">}</span>
        <span class="n">maybe_empty_cache</span><span class="p">(</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">out</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Pool worker (numpy)</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_worker</span><span class="p">(</span><span class="n">worker_args</span><span class="p">):</span>
    <span class="n">data_h0</span><span class="p">,</span> <span class="n">h1_per_T</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span> <span class="o">=</span> <span class="n">worker_args</span>
    <span class="n">h0</span> <span class="o">=</span> <span class="n">_statistics</span><span class="p">(</span><span class="n">data_h0</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>
    <span class="c1"># Under H1 the change sits at T/2, so a fresh series is needed for each T.</span>
    <span class="n">h1</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">:</span>
        <span class="n">stats_T</span> <span class="o">=</span> <span class="n">_statistics</span><span class="p">(</span><span class="n">h1_per_T</span><span class="p">[</span><span class="n">T</span><span class="p">],</span> <span class="p">[</span><span class="n">T</span><span class="p">],</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">stats_T</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
            <span class="n">h1</span><span class="o">.</span><span class="n">setdefault</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="p">{})[</span><span class="n">T</span><span class="p">]</span> <span class="o">=</span> <span class="n">d</span><span class="p">[</span><span class="n">T</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">h0</span><span class="p">,</span> <span class="n">h1</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_run_pool</span><span class="p">(</span><span class="n">data_h0</span><span class="p">,</span> <span class="n">h1_data</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">n_workers</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">storage_path</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="n">n_trials</span> <span class="o">=</span> <span class="n">data_h0</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">worker_args</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">(</span><span class="n">data_h0</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="p">{</span><span class="n">T</span><span class="p">:</span> <span class="n">h1_data</span><span class="p">[</span><span class="n">T</span><span class="p">][</span><span class="n">i</span><span class="p">]</span> <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">},</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n_trials</span><span class="p">)</span>
    <span class="p">]</span>
    <span class="n">all_h0</span><span class="p">,</span> <span class="n">all_h1</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Starting power computation: </span><span class="si">{</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> trials via Pool, </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span><span class="si">}</span><span class="s2"> T-points...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">n_trials</span><span class="p">,</span>
                  <span class="n">description</span><span class="o">=</span><span class="s2">&quot;MC trials (Pool)&quot;</span><span class="p">,</span> <span class="n">unit</span><span class="o">=</span><span class="s2">&quot;trials&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="k">with</span> <span class="n">Pool</span><span class="p">(</span><span class="n">processes</span><span class="o">=</span><span class="n">n_workers</span><span class="p">)</span> <span class="k">as</span> <span class="n">pool</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">h0</span><span class="p">,</span> <span class="n">h1</span> <span class="ow">in</span> <span class="n">pool</span><span class="o">.</span><span class="n">imap_unordered</span><span class="p">(</span><span class="n">_worker</span><span class="p">,</span> <span class="n">worker_args</span><span class="p">):</span>
                <span class="n">all_h0</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">h0</span><span class="p">)</span>
                <span class="n">all_h1</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">h1</span><span class="p">)</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_stack</span><span class="p">(</span><span class="n">dicts</span><span class="p">):</span>
        <span class="n">names</span> <span class="o">=</span> <span class="n">dicts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">keys</span><span class="p">()</span>
        <span class="k">return</span> <span class="p">{</span><span class="n">name</span><span class="p">:</span> <span class="p">{</span><span class="n">T</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">d</span><span class="p">[</span><span class="n">name</span><span class="p">][</span><span class="n">T</span><span class="p">])</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">dicts</span><span class="p">])</span> <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">}</span>
                <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">names</span><span class="p">}</span>

    <span class="k">return</span> <span class="n">_stack</span><span class="p">(</span><span class="n">all_h0</span><span class="p">),</span> <span class="n">_stack</span><span class="p">(</span><span class="n">all_h1</span><span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Batched path</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_run_batched</span><span class="p">(</span><span class="n">data_h0</span><span class="p">,</span> <span class="n">h1_data</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">storage_path</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Starting batched power computation on </span><span class="si">{</span><span class="n">backend</span><span class="si">}</span><span class="s2">...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span><span class="n">storage_path</span><span class="p">,</span> <span class="mi">1</span> <span class="o">+</span> <span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">),</span>
                  <span class="n">description</span><span class="o">=</span><span class="s2">&quot;H0 + H1 per T&quot;</span><span class="p">,</span> <span class="n">unit</span><span class="o">=</span><span class="s2">&quot;blocks&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">h0</span> <span class="o">=</span> <span class="n">_statistics</span><span class="p">(</span><span class="n">get_data_on_device</span><span class="p">(</span><span class="n">data_h0</span><span class="p">,</span> <span class="n">backend</span><span class="p">),</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
        <span class="n">progress</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
        <span class="n">h1</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">:</span>
            <span class="n">stats_T</span> <span class="o">=</span> <span class="n">_statistics</span><span class="p">(</span><span class="n">get_data_on_device</span><span class="p">(</span><span class="n">h1_data</span><span class="p">[</span><span class="n">T</span><span class="p">],</span> <span class="n">backend</span><span class="p">),</span> <span class="p">[</span><span class="n">T</span><span class="p">],</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">stats_T</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
                <span class="n">h1</span><span class="o">.</span><span class="n">setdefault</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="p">{})[</span><span class="n">T</span><span class="p">]</span> <span class="o">=</span> <span class="n">d</span><span class="p">[</span><span class="n">T</span><span class="p">]</span>
            <span class="n">progress</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
    <span class="k">return</span> <span class="n">h0</span><span class="p">,</span> <span class="n">h1</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Entry point</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span><span class="vm">__doc__</span><span class="p">)</span>
    <span class="n">add_mc_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">add_mc_h1_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">set_defaults</span><span class="p">(</span><span class="n">T_max</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span> <span class="n">T_min</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">n_T</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span> <span class="n">n_trials</span><span class="o">=</span><span class="mi">5000</span><span class="p">,</span> <span class="n">pfa</span><span class="o">=</span><span class="mf">1e-2</span><span class="p">)</span>

    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--a&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Size of the first Kronecker factor.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--b&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Size of the second Kronecker factor.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--n-samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Samples per date (default: p+1 = a*b+1 = 13).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--texture&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;k&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;k&quot;</span><span class="p">,</span> <span class="s2">&quot;gaussian&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;&#39;k&#39; for K-distributed data with shape --nu (default), &#39;gaussian&#39; for tau = 1.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--nu&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Shape of the K-distribution texture when --texture k (default 1.0).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-a0&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.3+0.7j&quot;</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of A under H0.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-b0&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.3+0.6j&quot;</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of B under H0.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-a1&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.3+0.5j&quot;</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of A after the change.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-b1&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.4+0.5j&quot;</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of B after the change.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--step-rule&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;fixed&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;fixed&quot;</span><span class="p">,</span> <span class="s2">&quot;armijo&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Step of the recursive estimators (default &#39;fixed&#39;, the schedule of eq. 19).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--init-mode&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;mm&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;mm&quot;</span><span class="p">,</span> <span class="s2">&quot;identity&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Initialisation of the recursive Kronecker estimator (default &#39;mm&#39;).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--alpha-0&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Initial step (default 1.0).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--iter-max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">30</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Max fixed-point / MM iterations.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-4</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Convergence tolerance.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--with-gaussian&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Add the Gaussian covariance equality GLRT as a fifth baseline.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--detectors&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Comma-separated subset of detectors to run (e.g. &#39;SG-O&#39;). Default: all &quot;</span>
             <span class="s2">&quot;four. Use it to iterate on one curve without paying for the others; the &quot;</span>
             <span class="s2">&quot;thresholds are per detector anyway, so a subset gives the same numbers.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--debug&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Tiny configuration (60 trials, T up to 10, PFA 0.1) to validate the &quot;</span>
             <span class="s2">&quot;pipeline in seconds. Results are NOT publication grade.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_T</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T_min</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">pfa</span> <span class="o">=</span> <span class="mi">60</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">0.1</span>

    <span class="n">init_logging</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span><span class="s2">&quot;--debug: 60 trials, T_max=10, PFA=0.1. Pipeline check only.&quot;</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">min_trials</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="mi">10</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">pfa</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span> <span class="o">&lt;</span> <span class="n">min_trials</span><span class="p">:</span>
            <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;n_trials=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> &lt; 10/PFA=</span><span class="si">{</span><span class="n">min_trials</span><span class="si">}</span><span class="s2">: the threshold at &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;PFA=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">pfa</span><span class="si">}</span><span class="s2"> will be poorly estimated. Consider --n-trials </span><span class="si">{</span><span class="n">min_trials</span><span class="si">}</span><span class="s2">.&quot;</span><span class="p">)</span>

    <span class="n">a</span><span class="p">,</span> <span class="n">b</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">a</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">b</span>
    <span class="n">p</span> <span class="o">=</span> <span class="n">a</span> <span class="o">*</span> <span class="n">b</span>
    <span class="n">n_samples</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span> <span class="k">else</span> <span class="n">p</span> <span class="o">+</span> <span class="mi">1</span>
    <span class="n">T_vec</span> <span class="o">=</span> <span class="n">T_vec_logspace</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">T_min</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_T</span><span class="p">)</span>
    <span class="n">T_max</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span>
    <span class="n">tau_shape</span> <span class="o">=</span> <span class="kc">None</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">texture</span> <span class="o">==</span> <span class="s2">&quot;gaussian&quot;</span> <span class="k">else</span> <span class="n">args</span><span class="o">.</span><span class="n">nu</span>
    <span class="n">tau_scale</span> <span class="o">=</span> <span class="mf">1.0</span> <span class="k">if</span> <span class="n">tau_shape</span> <span class="ow">is</span> <span class="kc">None</span> <span class="k">else</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">nu</span>

    <span class="n">cfg</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;detectors&quot;</span><span class="p">:</span> <span class="p">([</span><span class="n">d</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">detectors</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">&quot;,&quot;</span><span class="p">)]</span>
                      <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">detectors</span> <span class="k">else</span> <span class="kc">None</span><span class="p">),</span>
        <span class="s2">&quot;step_rule&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">step_rule</span><span class="p">,</span>
        <span class="s2">&quot;init_mode&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">init_mode</span><span class="p">,</span>
        <span class="s2">&quot;alpha_0&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">alpha_0</span><span class="p">,</span>
        <span class="s2">&quot;iter_max&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span>
        <span class="s2">&quot;tol&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">tol</span><span class="p">,</span>
        <span class="s2">&quot;with_gaussian&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">with_gaussian</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Power vs T: a=</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">, b=</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">, p=</span><span class="si">{</span><span class="n">p</span><span class="si">}</span><span class="s2">, n_samples=</span><span class="si">{</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">, texture=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">texture</span><span class="si">}</span><span class="s2">, &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;n_trials=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">, T=[</span><span class="si">{</span><span class="n">T_vec</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s2">..</span><span class="si">{</span><span class="n">T_vec</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">] (</span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span><span class="si">}</span><span class="s2"> pts), &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;PFA=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">pfa</span><span class="si">}</span><span class="s2">, backend=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">A0</span><span class="p">,</span> <span class="n">B0</span> <span class="o">=</span> <span class="n">make_ab_toeplitz</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_a0</span><span class="p">),</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_b0</span><span class="p">))</span>
    <span class="n">A1</span><span class="p">,</span> <span class="n">B1</span> <span class="o">=</span> <span class="n">make_ab_toeplitz</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_a1</span><span class="p">),</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_b1</span><span class="p">))</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="s2">&quot;Generating H0 data...&quot;</span><span class="p">)</span>
    <span class="n">data_h0</span> <span class="o">=</span> <span class="n">generate_kronecker_data</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">T_max</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A0</span><span class="p">,</span> <span class="n">B0</span><span class="p">,</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">tau_shape</span><span class="o">=</span><span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="o">=</span><span class="n">tau_scale</span><span class="p">)</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="s2">&quot;Generating H1 data (one series per T, change at T/2)...&quot;</span><span class="p">)</span>
    <span class="n">h1_data</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">:</span>
        <span class="n">n_change</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="nb">int</span><span class="p">(</span><span class="n">T</span> <span class="o">*</span> <span class="n">args</span><span class="o">.</span><span class="n">change_fraction</span><span class="p">))</span>
        <span class="n">h1_data</span><span class="p">[</span><span class="n">T</span><span class="p">]</span> <span class="o">=</span> <span class="n">generate_kronecker_data_h1</span><span class="p">(</span>
            <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">T</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A0</span><span class="p">,</span> <span class="n">B0</span><span class="p">,</span> <span class="n">A1</span><span class="p">,</span> <span class="n">B1</span><span class="p">,</span>
            <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="mi">1000</span> <span class="o">+</span> <span class="n">T</span><span class="p">,</span> <span class="n">n_change_dates</span><span class="o">=</span><span class="n">n_change</span><span class="p">,</span>
            <span class="n">tau_shape</span><span class="o">=</span><span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="o">=</span><span class="n">tau_scale</span><span class="p">)</span>

    <span class="n">exporter</span> <span class="o">=</span> <span class="n">MCResultExporter</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span> <span class="n">Path</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span> <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">texture</span><span class="si">}</span><span class="s2">_a</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">_b</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">_T</span><span class="si">{</span><span class="n">T_max</span><span class="si">}</span><span class="s2">_n</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span>
        <span class="n">plot_template</span><span class="o">=</span><span class="n">_MC_PLOT_TEMPLATE_POWER_MULTI</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="p">(</span><span class="n">h0</span><span class="p">,</span> <span class="n">h1</span><span class="p">),</span> <span class="n">elapsed</span> <span class="o">=</span> <span class="n">timed_run</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_pool</span><span class="p">(</span><span class="n">data_h0</span><span class="p">,</span> <span class="n">h1_data</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_workers</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span>
                          <span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_batched</span><span class="p">(</span><span class="n">data_h0</span><span class="p">,</span> <span class="n">h1_data</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span>
                             <span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span>
    <span class="p">)</span>

    <span class="n">regime</span> <span class="o">=</span> <span class="s2">&quot;gaussien&quot;</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">texture</span> <span class="o">==</span> <span class="s2">&quot;gaussian&quot;</span> <span class="k">else</span> <span class="sa">f</span><span class="s2">&quot;K, nu=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">}</span><span class="s2">&quot;</span>
    <span class="n">title</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">&quot;Puissance à $P_</span><span class="se">{{</span><span class="s2">fa</span><span class="se">}}</span><span class="s2">$=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">pfa</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">  (</span><span class="si">{</span><span class="n">regime</span><span class="si">}</span><span class="s2">, a=</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">, b=</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">, n=</span><span class="si">{</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">)&quot;</span>
    <span class="n">finish_power_multi</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">exporter</span><span class="p">,</span> <span class="n">h0</span><span class="p">,</span> <span class="n">h1</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="s2">&quot;mc_power&quot;</span><span class="p">,</span> <span class="n">title</span><span class="p">,</span> <span class="n">elapsed</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--a</span><span class="param-type">int</span><span class="param-default">default <b>3</b></span>
</div>
<p class="param-help">Size of the first Kronecker factor.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--b</span><span class="param-type">int</span><span class="param-default">default <b>4</b></span>
</div>
<p class="param-help">Size of the second Kronecker factor.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-samples</span><span class="param-type">int</span>
</div>
<p class="param-help">Samples per date (default: p+1 = a*b+1 = 13).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--texture</span><span class="param-type">str</span><span class="param-default">default <b>k</b></span>
</div>
<p class="param-help">&#x27;k&#x27; for K-distributed data with shape --nu (default), &#x27;gaussian&#x27; for tau = 1.</p><p class="param-choices">choices: k, gaussian</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--nu</span><span class="param-type">float</span><span class="param-default">default <b>1.0</b></span>
</div>
<p class="param-help">Shape of the K-distribution texture when --texture k (default 1.0).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-a0</span><span class="param-type">str</span><span class="param-default">default <b>0.3+0.7j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of A under H0.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-b0</span><span class="param-type">str</span><span class="param-default">default <b>0.3+0.6j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of B under H0.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-a1</span><span class="param-type">str</span><span class="param-default">default <b>0.3+0.5j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of A after the change.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-b1</span><span class="param-type">str</span><span class="param-default">default <b>0.4+0.5j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of B after the change.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--step-rule</span><span class="param-type">str</span><span class="param-default">default <b>fixed</b></span>
</div>
<p class="param-help">Step of the recursive estimators (default &#x27;fixed&#x27;, the schedule of eq. 19).</p><p class="param-choices">choices: fixed, armijo</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--init-mode</span><span class="param-type">str</span><span class="param-default">default <b>mm</b></span>
</div>
<p class="param-help">Initialisation of the recursive Kronecker estimator (default &#x27;mm&#x27;).</p><p class="param-choices">choices: mm, identity</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--alpha-0</span><span class="param-type">float</span><span class="param-default">default <b>1.0</b></span>
</div>
<p class="param-help">Initial step (default 1.0).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--iter-max</span><span class="param-type">int</span><span class="param-default">default <b>30</b></span>
</div>
<p class="param-help">Max fixed-point / MM iterations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--tol</span><span class="param-type">float</span><span class="param-default">default <b>0.0001</b></span>
</div>
<p class="param-help">Convergence tolerance.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--with-gaussian</span><span class="param-type">flag</span>
</div>
<p class="param-help">Add the Gaussian covariance equality GLRT as a fifth baseline.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--detectors</span><span class="param-type">str</span>
</div>
<p class="param-help">Comma-separated subset of detectors to run (e.g. &#x27;SG-O&#x27;). Default: all four. Use it to iterate on one curve without paying for the others; the thresholds are per detector anyway, so a subset gives the same numbers.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--debug</span><span class="param-type">flag</span>
</div>
<p class="param-help">Tiny configuration (60 trials, T up to 10, PFA 0.1) to validate the pipeline in seconds. Results are NOT publication grade.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--sigma2-seed</span><span class="param-type">int</span><span class="param-default">default <b>1</b></span>
</div>
<p class="param-help">Seed for Sigma_2 (H1 distribution, default 1 — different from --sigma-seed).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--change-fraction</span><span class="param-type">float</span><span class="param-default">default <b>0.5</b></span>
</div>
<p class="param-help">Change point as a fraction of T, so n_change_dates = max(2, int(T * change_fraction)). Default 0.5 — change at midpoint, ensuring equal pre/post evidence at every T.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--pfa</span><span class="param-type">float</span><span class="param-default">default <b>0.001</b></span>
</div>
<p class="param-help">Target false alarm probability for power estimation (default 1e-3). Reliable threshold estimation requires at least 10/PFA H0 trials.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-features</span><span class="param-type">int</span><span class="param-default">default <b>8</b></span>
</div>
<p class="param-help">Feature dimension p; n_samples is fixed to 2*p+1 (default 8).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--T-max</span><span class="param-type">int</span><span class="param-default">default <b>1000</b></span>
</div>
<p class="param-help">Maximum number of time steps (default 1000).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--T-min</span><span class="param-type">int</span><span class="param-default">default <b>5</b></span>
</div>
<p class="param-help">Minimum number of time steps (default 5).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-T</span><span class="param-type">int</span><span class="param-default">default <b>30</b></span>
</div>
<p class="param-help">Number of T values in log scale (default 30).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--sigma-seed</span><span class="param-type">int</span><span class="param-default">default <b>0</b></span>
</div>
<p class="param-help">Seed for Sigma_true generation, independent from --seed (default 0).</p>
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

`2-detection/experiments/sar/sar_mc_power.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
