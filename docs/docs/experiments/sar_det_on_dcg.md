<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sar_det_on_dcg</span>
</nav>

# sar_det_on_dcg

Online DCG GLRT change detection on real SAR data

**Tags:** `detection`  `dcg`  `online`  `real-data`  `SAR`

## Run

```sh
uv run python 2-detection/sar_experiments/compute_detection_real_data/online_dcg.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sar_experiments/compute_detection_real_data/online_dcg.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">117 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sar_experiments/compute_detection_real_data/online_dcg.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Online DCG (Date-Class Gaussian) change detection on CPU or GPU.</span>
<span class="c1"># Processes all dates sequentially maintaining H0 (pooled) and H1 (per-date) estimates.</span>
<span class="c1"># Memory-efficient for large images.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">sys</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>

<span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">str</span><span class="p">(</span><span class="n">Path</span><span class="p">(</span><span class="vm">__file__</span><span class="p">)</span><span class="o">.</span><span class="n">parent</span><span class="p">))</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">time</span><span class="w"> </span><span class="kn">import</span> <span class="n">perf_counter</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.detection_online</span><span class="w"> </span><span class="kn">import</span> <span class="n">OnlineDCGDetector</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">peak_memory_bytes</span><span class="p">,</span> <span class="n">reset_peak_memory</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.hardware_ressources</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">OnlineImageResourceManager</span><span class="p">,</span>
    <span class="n">OnlineImageGPURessourceManager</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.logging_config</span><span class="w"> </span><span class="kn">import</span> <span class="n">setup_logging</span><span class="p">,</span> <span class="n">log_arguments</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">utils</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">add_common_args</span><span class="p">,</span>
    <span class="n">setup_run</span><span class="p">,</span>
    <span class="n">load_sits</span><span class="p">,</span>
    <span class="n">DetectionMapExporter</span><span class="p">,</span>
    <span class="n">plot_glrt_map</span><span class="p">,</span>
<span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Online DCG change detection on CPU or GPU backend.&quot;</span>
    <span class="p">)</span>
    <span class="n">add_common_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--iter-max&quot;</span><span class="p">,</span>
        <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Maximum iterations for H0 and H1 natural gradient estimators (default 5).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--tol&quot;</span><span class="p">,</span>
        <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="mf">1e-8</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Convergence tolerance for H1 estimator (default 1e-8).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>
    <span class="n">setup_logging</span><span class="p">(</span><span class="n">quiet</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">quiet</span><span class="p">,</span> <span class="n">debug</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">log_debug</span><span class="p">)</span>
    <span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>
    <span class="n">log_arguments</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>

    <span class="n">cfg</span> <span class="o">=</span> <span class="n">setup_run</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>
    <span class="n">exporter</span> <span class="o">=</span> <span class="n">DetectionMapExporter</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">cfg</span><span class="p">)</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="s2">&quot;Loading SITS data...&quot;</span><span class="p">)</span>
    <span class="n">sits_np</span> <span class="o">=</span> <span class="n">load_sits</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>  <span class="c1"># (n_times, n_rows, n_cols, n_features)</span>
    <span class="n">n_times</span><span class="p">,</span> <span class="n">n_rows</span><span class="p">,</span> <span class="n">n_cols</span><span class="p">,</span> <span class="n">n_features</span> <span class="o">=</span> <span class="n">sits_np</span><span class="o">.</span><span class="n">shape</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span>
        <span class="sa">f</span><span class="s2">&quot;Data loaded: shape </span><span class="si">{</span><span class="n">sits_np</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">, &quot;</span>
        <span class="sa">f</span><span class="s2">&quot;backend </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">, window_size </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="si">}</span><span class="s2">, &quot;</span>
        <span class="sa">f</span><span class="s2">&quot;splitting </span><span class="si">{</span><span class="n">cfg</span><span class="o">.</span><span class="n">splitting</span><span class="si">}</span><span class="s2">&quot;</span>
    <span class="p">)</span>

    <span class="n">detector</span> <span class="o">=</span> <span class="n">OnlineDCGDetector</span><span class="p">(</span>
        <span class="n">backend_name</span><span class="o">=</span><span class="nb">str</span><span class="p">(</span><span class="n">cfg</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
        <span class="n">iter_max</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span>
        <span class="n">tol</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">tol</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">cfg</span><span class="o">.</span><span class="n">is_gpu</span><span class="p">:</span>
        <span class="n">resource_manager</span> <span class="o">=</span> <span class="n">OnlineImageGPURessourceManager</span><span class="p">(</span>
            <span class="n">sits_np</span><span class="p">,</span>
            <span class="n">window_size</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="p">,</span>
            <span class="n">stride</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
            <span class="n">detector</span><span class="o">=</span><span class="n">detector</span><span class="p">,</span>
            <span class="n">backend</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
            <span class="n">splitting</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">splitting</span><span class="p">,</span>
            <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">quiet</span> <span class="k">else</span> <span class="mi">1</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">resource_manager</span> <span class="o">=</span> <span class="n">OnlineImageResourceManager</span><span class="p">(</span>
            <span class="n">sits_np</span><span class="p">,</span>
            <span class="n">window_size</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="p">,</span>
            <span class="n">stride</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
            <span class="n">detector</span><span class="o">=</span><span class="n">detector</span><span class="p">,</span>
            <span class="n">backend</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
            <span class="n">splitting</span><span class="o">=</span><span class="n">cfg</span><span class="o">.</span><span class="n">splitting</span><span class="p">,</span>
            <span class="n">verbose</span><span class="o">=</span><span class="mi">0</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">quiet</span> <span class="k">else</span> <span class="mi">1</span><span class="p">,</span>
        <span class="p">)</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="s2">&quot;Starting online DCG detection processing...&quot;</span><span class="p">)</span>

    <span class="n">reset_peak_memory</span><span class="p">(</span><span class="n">cfg</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">t0</span> <span class="o">=</span> <span class="n">perf_counter</span><span class="p">()</span>
    <span class="n">glrt_map</span> <span class="o">=</span> <span class="n">resource_manager</span><span class="o">.</span><span class="n">process_all_data</span><span class="p">()</span>
    <span class="n">elapsed</span> <span class="o">=</span> <span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Detection completed in </span><span class="si">{</span><span class="n">elapsed</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">s.&quot;</span><span class="p">)</span>

    <span class="n">glrt_map_np</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">glrt_map</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>

    <span class="n">exporter</span><span class="o">.</span><span class="n">save</span><span class="p">(</span>
        <span class="n">glrt_map_np</span><span class="p">,</span> <span class="s2">&quot;dcg_online&quot;</span><span class="p">,</span> <span class="n">elapsed</span><span class="p">,</span> <span class="n">title</span><span class="o">=</span><span class="s2">&quot;Online DCG GLRT&quot;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;jet&quot;</span>
    <span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plot_glrt_map</span><span class="p">(</span><span class="n">glrt_map_np</span><span class="p">,</span> <span class="s2">&quot;Online DCG GLRT&quot;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;jet&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">cfg</span><span class="o">.</span><span class="n">is_gpu</span> <span class="ow">and</span> <span class="n">args</span><span class="o">.</span><span class="n">report_memory</span><span class="p">:</span>
        <span class="n">mem</span> <span class="o">=</span> <span class="n">peak_memory_bytes</span><span class="p">(</span><span class="n">cfg</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">mem</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;PEAK_GPU_MEMORY_BYTES=</span><span class="si">{</span><span class="n">mem</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="s2">&quot;Done.&quot;</span><span class="p">)</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--iter-max</span><span class="param-type">int</span><span class="param-default">default <b>5</b></span>
</div>
<p class="param-help">Maximum iterations for H0 and H1 natural gradient estimators (default 5).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-08</b></span>
</div>
<p class="param-help">Convergence tolerance for H1 estimator (default 1e-8).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--data-path</span><span class="param-type">str</span><span class="param-default">default <b>data/SAR/Scene_1.npy</b></span>
</div>
<p class="param-help">Path to the numpy data file (.npy). The loader resolves the matching *_time_first.npy produced by prepare_data.py.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--window-size</span><span class="param-type">int</span><span class="param-default">default <b>7</b></span>
</div>
<p class="param-help">Sliding window size.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--backend</span><span class="param-type">str</span><span class="param-default">default <b>numpy</b></span>
</div>
<p class="param-help">Computation backend (default: numpy).</p><p class="param-choices">choices: numpy, torch-cpu, torch-cuda, cupy, cupy-cuda, jax-cpu, jax-cuda</p>
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
<p class="param-help">Save result (.npy), provenance sidecar (.json), and plot script (_plot.py) (default: True).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--export-tikz</span><span class="param-type">flag</span>
</div>
<p class="param-help">Also save a TikZ/PGFPlots figure (.tex) alongside the exported data.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage-path</span><span class="param-alias">--storage_path</span><span class="param-alias">--export-path</span><span class="param-type">str</span><span class="param-default">default <b>./exports</b></span>
</div>
<p class="param-help">Directory for exported plots; --storage-path is the qanat alias (default: ./exports).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--debug</span><span class="param-type">flag</span>
</div>
<p class="param-help">Crop data to 100×100 for fast debugging.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--splitting</span><span class="param-type">str</span>
</div>
<p class="param-help">Grid splitting &#x27;(r,c)&#x27;. Default: (1,1) CPU, (5,5) GPU.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--wavelet</span><span class="param-type">flag</span>
</div>
<p class="param-help">Apply wavelet decomposition before detection.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--wavelet-R</span><span class="param-type">int</span><span class="param-default">default <b>2</b></span>
</div>
<p class="param-help">Range sub-bands (default 2).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--wavelet-L</span><span class="param-type">int</span><span class="param-default">default <b>2</b></span>
</div>
<p class="param-help">Azimuth sub-bands (default 2).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--wavelet-no-decimate</span><span class="param-type">flag</span>
</div>
<p class="param-help">Disable sub-band decimation. Output is R*L times larger than input (full-resolution redundant representation). Default: decimation ON.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--quiet</span><span class="param-type">flag</span>
</div>
<p class="param-help">Suppress verbose output.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--log-debug</span><span class="param-type">flag</span>
</div>
<p class="param-help">Enable debug-level logging.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--report-memory</span><span class="param-type">flag</span>
</div>
<p class="param-help">Print peak GPU memory at the end (torch-cuda only).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--repeat-times</span><span class="param-type">int</span><span class="param-default">default <b>1</b></span>
</div>
<p class="param-help">Repeat the time axis N times using a palindrome bounce (e.g. T=68, repeat=2 → 136 frames: 0..67, 66..1, 0..1, ...). Materialises the full repeated array in RAM.</p>
</div>
</div>

## Config

`2-detection/experiments/sar/sar_det_on_dcg.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
