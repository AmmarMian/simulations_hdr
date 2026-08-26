<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sonar_pfa_threshold</span>
</nav>

# sonar_pfa_threshold

MC empirical PFA vs threshold for sonar two-array detectors — matrix-CFAR verification

**Tags:** `sonar`  `detection`  `monte-carlo`  `pfa-cfar`

## Run

```sh
uv run python 2-detection/sonar_experiments/mc_simulations/mc_pfa_threshold.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sonar_experiments/mc_simulations/mc_pfa_threshold.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">199 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sonar_experiments/mc_simulations/mc_pfa_threshold.py</p>
<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/env python</span>
<span class="sd">&quot;&quot;&quot;MC PFA-vs-threshold curves for sonar two-array detectors.</span>

<span class="sd">Collects H0 test statistics from all detectors under Gaussian or K-distributed</span>
<span class="sd">clutter and plots empirical P(stat &gt; eta) vs eta — verifying matrix-CFAR</span>
<span class="sd">behaviour.  Includes both known-M and adaptive (2TYL / SCM) variants.</span>

<span class="sd">Backend selection:</span>
<span class="sd">  numpy     → multiprocessing.Pool, one chunk of trials per worker</span>
<span class="sd">  all other → trials chunked on device (GPU memory budget via --chunk-size)</span>
<span class="sd">&quot;&quot;&quot;</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">math</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">multiprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Pool</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">rich.progress</span><span class="w"> </span><span class="kn">import</span> <span class="n">BarColumn</span><span class="p">,</span> <span class="n">Progress</span><span class="p">,</span> <span class="n">TextColumn</span><span class="p">,</span> <span class="n">TimeElapsedColumn</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="n">SCMEstimator</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">MCResultExporter</span><span class="p">,</span>
    <span class="n">chunk_trial_ranges</span><span class="p">,</span>
    <span class="n">make_mc_parser</span><span class="p">,</span>
    <span class="n">maybe_empty_cache</span><span class="p">,</span>
    <span class="n">timed_run</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sonar</span><span class="w"> </span><span class="kn">import</span> <span class="n">detectors</span> <span class="k">as</span> <span class="n">det</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sonar</span><span class="w"> </span><span class="kn">import</span> <span class="n">estimation</span> <span class="k">as</span> <span class="n">est</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sonar</span><span class="w"> </span><span class="kn">import</span> <span class="n">mc</span> <span class="k">as</span> <span class="n">smc</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sonar</span><span class="w"> </span><span class="kn">import</span> <span class="n">simulation</span> <span class="k">as</span> <span class="n">sim</span>

<span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>

<span class="n">_CHUNK</span> <span class="o">=</span> <span class="mi">200</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Detector registry</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_build_detectors</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="s2">&quot;numpy&quot;</span><span class="p">):</span>
    <span class="n">glrt</span>  <span class="o">=</span> <span class="n">det</span><span class="o">.</span><span class="n">MNMFGlrt</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">)</span>
    <span class="n">rao</span>   <span class="o">=</span> <span class="n">det</span><span class="o">.</span><span class="n">MNMFRao</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">)</span>
    <span class="n">indep</span> <span class="o">=</span> <span class="n">det</span><span class="o">.</span><span class="n">MNMFIndependent</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">)</span>
    <span class="n">mmf</span>   <span class="o">=</span> <span class="n">det</span><span class="o">.</span><span class="n">MimoMatchedFilter</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">)</span>

    <span class="n">tyl_est</span> <span class="o">=</span> <span class="n">est</span><span class="o">.</span><span class="n">TwoArrayTylerEstimator</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend_name</span><span class="p">)</span>
    <span class="n">scm_est</span> <span class="o">=</span> <span class="n">SCMEstimator</span><span class="p">(</span><span class="n">backend_name</span><span class="o">=</span><span class="n">backend_name</span><span class="p">)</span>

    <span class="k">return</span> <span class="p">{</span>
        <span class="s2">&quot;M-NMF-G&quot;</span><span class="p">:</span>       <span class="n">glrt</span><span class="p">,</span>
        <span class="s2">&quot;M-NMF-R&quot;</span><span class="p">:</span>       <span class="n">rao</span><span class="p">,</span>
        <span class="s2">&quot;M-NMF-I&quot;</span><span class="p">:</span>       <span class="n">indep</span><span class="p">,</span>
        <span class="s2">&quot;MIMO-MF&quot;</span><span class="p">:</span>       <span class="n">mmf</span><span class="p">,</span>
        <span class="s2">&quot;M-ANMF-G-TYL&quot;</span><span class="p">:</span>  <span class="n">det</span><span class="o">.</span><span class="n">AdaptiveSonarDetector</span><span class="p">(</span><span class="n">det</span><span class="o">.</span><span class="n">MNMFGlrt</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">),</span> <span class="n">tyl_est</span><span class="p">),</span>
        <span class="s2">&quot;M-ANMF-R-TYL&quot;</span><span class="p">:</span>  <span class="n">det</span><span class="o">.</span><span class="n">AdaptiveSonarDetector</span><span class="p">(</span><span class="n">det</span><span class="o">.</span><span class="n">MNMFRao</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">),</span>  <span class="n">tyl_est</span><span class="p">),</span>
        <span class="s2">&quot;M-ANMF-G-SCM&quot;</span><span class="p">:</span>  <span class="n">det</span><span class="o">.</span><span class="n">AdaptiveSonarDetector</span><span class="p">(</span><span class="n">det</span><span class="o">.</span><span class="n">MNMFGlrt</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">),</span> <span class="n">scm_est</span><span class="p">),</span>
        <span class="c1"># The Rao/SCM pair completes the symmetry of the two panels: each panel</span>
        <span class="c1"># shows the same three levels of knowledge on M (known, SCM, Tyler).</span>
        <span class="s2">&quot;M-ANMF-R-SCM&quot;</span><span class="p">:</span>  <span class="n">det</span><span class="o">.</span><span class="n">AdaptiveSonarDetector</span><span class="p">(</span><span class="n">det</span><span class="o">.</span><span class="n">MNMFRao</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend_name</span><span class="p">),</span>  <span class="n">scm_est</span><span class="p">),</span>
    <span class="p">}</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Pool worker (numpy only — generates own data to avoid large IPC)</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_worker</span><span class="p">(</span><span class="n">args</span><span class="p">):</span>
    <span class="n">chunk_seed</span><span class="p">,</span> <span class="n">n</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span> <span class="o">=</span> <span class="n">args</span>
    <span class="n">dets</span> <span class="o">=</span> <span class="n">_build_detectors</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>
    <span class="n">x</span>    <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">generate_sonar_data_h0</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">chunk_seed</span><span class="p">)</span>
    <span class="n">xsec</span> <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">generate_secondary_data</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">chunk_seed</span> <span class="o">+</span> <span class="mi">10000</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">smc</span><span class="o">.</span><span class="n">run_detectors</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">dets</span><span class="p">,</span> <span class="n">X_secondary</span><span class="o">=</span><span class="n">xsec</span><span class="p">)</span>  <span class="c1"># {name: (n,) array}</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_run_pool</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">chunk_size</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="p">,</span> <span class="n">n_workers</span><span class="p">):</span>
    <span class="c1"># One task per worker — chunk_size is for GPU batching only.</span>
    <span class="n">n_w</span> <span class="o">=</span> <span class="n">n_workers</span> <span class="ow">or</span> <span class="n">os</span><span class="o">.</span><span class="n">cpu_count</span><span class="p">()</span> <span class="ow">or</span> <span class="mi">1</span>
    <span class="n">worker_chunk</span> <span class="o">=</span> <span class="n">math</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">n_trials</span> <span class="o">/</span> <span class="n">n_w</span><span class="p">)</span>
    <span class="n">c_starts</span><span class="p">,</span> <span class="n">chunk</span><span class="p">,</span> <span class="n">n_chunks</span> <span class="o">=</span> <span class="n">chunk_trial_ranges</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">worker_chunk</span><span class="p">)</span>
    <span class="n">worker_args</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">(</span><span class="n">seed</span> <span class="o">+</span> <span class="n">c</span><span class="p">,</span> <span class="nb">min</span><span class="p">(</span><span class="n">chunk</span><span class="p">,</span> <span class="n">n_trials</span> <span class="o">-</span> <span class="n">c</span><span class="p">),</span> <span class="n">m</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">c_starts</span>
    <span class="p">]</span>
    <span class="n">all_stats</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">dict</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;H0: </span><span class="si">{</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> trials via Pool (</span><span class="si">{</span><span class="n">n_chunks</span><span class="si">}</span><span class="s2"> workers, ≤</span><span class="si">{</span><span class="n">chunk</span><span class="si">}</span><span class="s2"> trials each)...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;[progress.description]</span><span class="si">{task.description}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">BarColumn</span><span class="p">(),</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{task.completed}</span><span class="s2">/</span><span class="si">{task.total}</span><span class="s2"> workers&quot;</span><span class="p">),</span>
        <span class="n">TimeElapsedColumn</span><span class="p">(),</span>
    <span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="s2">&quot;[cyan]MC trials (Pool)...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="n">n_chunks</span><span class="p">)</span>
        <span class="k">with</span> <span class="n">Pool</span><span class="p">(</span><span class="n">processes</span><span class="o">=</span><span class="n">n_workers</span><span class="p">)</span> <span class="k">as</span> <span class="n">pool</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">result</span> <span class="ow">in</span> <span class="n">pool</span><span class="o">.</span><span class="n">imap_unordered</span><span class="p">(</span><span class="n">_worker</span><span class="p">,</span> <span class="n">worker_args</span><span class="p">):</span>
                <span class="n">all_stats</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">result</span><span class="p">)</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>

    <span class="k">return</span> <span class="p">{</span>
        <span class="n">name</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">([</span><span class="n">d</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">all_stats</span><span class="p">])</span>
        <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">all_stats</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="p">}</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Batched path (non-numpy backends)</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_run_batched</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">chunk_size</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="p">):</span>
    <span class="n">dets</span> <span class="o">=</span> <span class="n">_build_detectors</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">c_starts</span><span class="p">,</span> <span class="n">chunk</span><span class="p">,</span> <span class="n">n_chunks</span> <span class="o">=</span> <span class="n">chunk_trial_ranges</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">chunk_size</span><span class="p">)</span>
    <span class="n">all_stats</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">list</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span><span class="n">name</span><span class="p">:</span> <span class="p">[]</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">dets</span><span class="p">}</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;H0: </span><span class="si">{</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> trials on </span><span class="si">{</span><span class="n">backend</span><span class="si">}</span><span class="s2"> (</span><span class="si">{</span><span class="n">n_chunks</span><span class="si">}</span><span class="s2"> chunks of ≤</span><span class="si">{</span><span class="n">chunk</span><span class="si">}</span><span class="s2">)...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;[progress.description]</span><span class="si">{task.description}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">BarColumn</span><span class="p">(),</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{task.completed}</span><span class="s2">/</span><span class="si">{task.total}</span><span class="s2"> chunks&quot;</span><span class="p">),</span>
        <span class="n">TimeElapsedColumn</span><span class="p">(),</span>
    <span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;[cyan]MC batched (</span><span class="si">{</span><span class="n">backend</span><span class="si">}</span><span class="s2">)...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="n">n_chunks</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">c_starts</span><span class="p">:</span>
            <span class="n">n</span> <span class="o">=</span> <span class="nb">min</span><span class="p">(</span><span class="n">chunk</span><span class="p">,</span> <span class="n">n_trials</span> <span class="o">-</span> <span class="n">c</span><span class="p">)</span>
            <span class="n">x</span>    <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">generate_sonar_data_h0</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">seed</span> <span class="o">+</span> <span class="n">c</span><span class="p">)</span>
            <span class="n">xsec</span> <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">generate_secondary_data</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">seed</span> <span class="o">+</span> <span class="n">c</span> <span class="o">+</span> <span class="mi">10000</span><span class="p">)</span>
            <span class="n">x_dev</span>    <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">x</span><span class="p">,</span>    <span class="n">backend</span><span class="p">)</span>
            <span class="n">xsec_dev</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">xsec</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
            <span class="n">cs</span> <span class="o">=</span> <span class="n">smc</span><span class="o">.</span><span class="n">run_detectors</span><span class="p">(</span><span class="n">x_dev</span><span class="p">,</span> <span class="n">dets</span><span class="p">,</span> <span class="n">X_secondary</span><span class="o">=</span><span class="n">xsec_dev</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">s</span> <span class="ow">in</span> <span class="n">cs</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
                <span class="n">all_stats</span><span class="p">[</span><span class="n">name</span><span class="p">]</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
            <span class="n">maybe_empty_cache</span><span class="p">(</span><span class="n">backend</span><span class="p">)</span>
            <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>

    <span class="k">return</span> <span class="p">{</span><span class="n">name</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">(</span><span class="n">v</span><span class="p">)</span> <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">all_stats</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Entry point</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span><span class="vm">__doc__</span><span class="p">)</span>
    <span class="n">smc</span><span class="o">.</span><span class="n">add_mc_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--n-thresh&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">500</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of threshold grid points for PFA curve (default 500).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--chunk-size&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="n">_CHUNK</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;Trials per worker chunk (numpy/Pool) or per GPU memory batch (non-numpy). &quot;</span>
             <span class="sa">f</span><span class="s2">&quot;Default </span><span class="si">{</span><span class="n">_CHUNK</span><span class="si">}</span><span class="s2">.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">smc</span><span class="o">.</span><span class="n">apply_debug</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">logger</span><span class="p">,</span> <span class="n">n_trials</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span> <span class="n">n_thresh</span><span class="o">=</span><span class="mi">60</span><span class="p">)</span>

    <span class="n">logging</span><span class="o">.</span><span class="n">basicConfig</span><span class="p">(</span><span class="n">level</span><span class="o">=</span><span class="n">logging</span><span class="o">.</span><span class="n">INFO</span><span class="p">,</span> <span class="nb">format</span><span class="o">=</span><span class="s2">&quot;</span><span class="si">%(levelname)s</span><span class="s2"> </span><span class="si">%(message)s</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">m</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">m</span>
    <span class="n">K</span> <span class="o">=</span> <span class="n">smc</span><span class="o">.</span><span class="n">resolve_K</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>
    <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span> <span class="o">=</span> <span class="n">smc</span><span class="o">.</span><span class="n">clutter_params</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Sonar PFA-vs-threshold: m=</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">, K=</span><span class="si">{</span><span class="n">K</span><span class="si">}</span><span class="s2">, clutter=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">clutter</span><span class="si">}</span><span class="s2">, &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;n_trials=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">, backend=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">M</span> <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">make_sonar_covariance</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">beta</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho1</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho2</span><span class="p">)</span>
    <span class="n">P</span> <span class="o">=</span> <span class="n">sim</span><span class="o">.</span><span class="n">make_steering_matrix</span><span class="p">(</span><span class="n">m</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">theta1</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">theta2</span><span class="p">)</span>

    <span class="n">all_stats</span><span class="p">,</span> <span class="n">elapsed</span> <span class="o">=</span> <span class="n">timed_run</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_pool</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">chunk_size</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span>
                          <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_workers</span><span class="p">),</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_batched</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">chunk_size</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">K</span><span class="p">,</span> <span class="n">M</span><span class="p">,</span> <span class="n">P</span><span class="p">,</span>
                             <span class="n">tau_shape</span><span class="p">,</span> <span class="n">tau_scale</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">),</span>
    <span class="p">)</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Done in </span><span class="si">{</span><span class="n">elapsed</span><span class="si">:</span><span class="s2">.1f</span><span class="si">}</span><span class="s2">s&quot;</span><span class="p">)</span>

    <span class="n">detector_names</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="n">all_stats</span><span class="o">.</span><span class="n">keys</span><span class="p">())</span>
    <span class="n">export_stats</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;detector_names&quot;</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">detector_names</span><span class="p">)}</span>
    <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">detector_names</span><span class="p">:</span>
        <span class="n">thresh</span><span class="p">,</span> <span class="n">pfa</span> <span class="o">=</span> <span class="n">smc</span><span class="o">.</span><span class="n">aggregate_pfa_threshold</span><span class="p">(</span><span class="n">all_stats</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">args</span><span class="o">.</span><span class="n">n_thresh</span><span class="p">)</span>
        <span class="n">export_stats</span><span class="p">[</span><span class="sa">f</span><span class="s2">&quot;thresh_</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">thresh</span>
        <span class="n">export_stats</span><span class="p">[</span><span class="sa">f</span><span class="s2">&quot;pfa_</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">pfa</span>

    <span class="n">clutter_tag</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">clutter</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">clutter</span> <span class="o">==</span> <span class="s2">&quot;gaussian&quot;</span> <span class="k">else</span> <span class="sa">f</span><span class="s2">&quot;k_nu</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">}</span><span class="s2">&quot;</span>
    <span class="n">stem_tag</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">&quot;m</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">_K</span><span class="si">{</span><span class="n">K</span><span class="si">}</span><span class="s2">_</span><span class="si">{</span><span class="n">clutter_tag</span><span class="si">}</span><span class="s2">_n</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">&quot;</span>

    <span class="n">exporter</span> <span class="o">=</span> <span class="n">MCResultExporter</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">Path</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span> <span class="n">stem_tag</span><span class="p">,</span>
                                <span class="n">plot_template</span><span class="o">=</span><span class="n">smc</span><span class="o">.</span><span class="n">_MC_PFA_THRESHOLD_TEMPLATE</span><span class="p">)</span>
    <span class="n">title</span> <span class="o">=</span> <span class="p">(</span><span class="sa">f</span><span class="s2">&quot;PFA vs threshold — sonar (m=</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">, K=</span><span class="si">{</span><span class="n">K</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">clutter</span><span class="si">}</span><span class="s2"> clutter)&quot;</span><span class="p">)</span>
    <span class="n">exporter</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="n">export_stats</span><span class="p">,</span> <span class="s2">&quot;sonar_pfa_threshold&quot;</span><span class="p">,</span> <span class="n">elapsed</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n-thresh</span><span class="param-type">int</span><span class="param-default">default <b>500</b></span>
</div>
<p class="param-help">Number of threshold grid points for PFA curve (default 500).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--chunk-size</span><span class="param-type">int</span><span class="param-default">default <b>_CHUNK</b></span>
</div>

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

`2-detection/experiments/sonar/sonar_pfa_threshold.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
