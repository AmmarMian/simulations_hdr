<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sar_bench_time</span>
</nav>

# sar_bench_time

Time benchmark for offline Gaussian and DCG GLRT detectors (CPU + GPU)

**Tags:** `benchmark`  `time`  `SAR`

## Run

```sh
bash 2-detection/sar_experiments/benchmarks/time_benchmark.sh
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sar_experiments/benchmarks/time_benchmark.sh" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">174 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sar_experiments/benchmarks/time_benchmark.sh</p>
<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/env bash</span>
<span class="c1"># Benchmark detection scripts using hyperfine.</span>
<span class="c1"># Grid: {cpu, gpu} x {gaussian, dcg} x {no_wavelet, wavelet}</span>
<span class="c1"># Results saved as JSON per cell, aggregated to CSV, bar chart generated at end.</span>

<span class="nb">set</span> <span class="o">-</span><span class="n">euo</span> <span class="n">pipefail</span>

<span class="n">SCRIPT_DIR</span><span class="o">=</span><span class="s2">&quot;$(cd &quot;</span><span class="err">$</span><span class="p">(</span><span class="n">dirname</span> <span class="s2">&quot;$</span><span class="si">{BASH_SOURCE[0]}</span><span class="s2">&quot;</span><span class="p">)</span><span class="s2">&quot; &amp;&amp; pwd)&quot;</span>
<span class="n">DATA</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{SCRIPT_DIR}</span><span class="s2">/../../data/SAR/Scene_1.npy&quot;</span>
<span class="n">WINDOW_SIZE</span><span class="o">=</span><span class="mi">7</span>
<span class="n">RESULTS_DIR</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{SCRIPT_DIR}</span><span class="s2">/benchmark_results&quot;</span>
<span class="n">RUNS</span><span class="o">=</span><span class="mi">10</span>
<span class="n">WARMUP</span><span class="o">=</span><span class="mi">2</span>

<span class="c1"># Parse --storage_path / --storage-path from CLI args (qanat passes this).</span>
<span class="k">while</span> <span class="p">[[</span> <span class="err">$</span><span class="c1"># -gt 0 ]]; do</span>
  <span class="k">case</span> <span class="s2">&quot;$1&quot;</span> <span class="ow">in</span>
    <span class="o">--</span><span class="n">storage_path</span><span class="o">|--</span><span class="n">storage</span><span class="o">-</span><span class="n">path</span><span class="p">)</span> <span class="n">RESULTS_DIR</span><span class="o">=</span><span class="s2">&quot;$2&quot;</span><span class="p">;</span> <span class="n">shift</span> <span class="mi">2</span> <span class="p">;;</span>
    <span class="o">*</span><span class="p">)</span> <span class="n">shift</span> <span class="p">;;</span>
  <span class="n">esac</span>
<span class="n">done</span>

<span class="n">mkdir</span> <span class="o">-</span><span class="n">p</span> <span class="s2">&quot;$RESULTS_DIR&quot;</span>

<span class="n">SCRIPT_GAUSSIAN</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{SCRIPT_DIR}</span><span class="s2">/../compute_detection_real_data/offline_gaussian.py&quot;</span>
<span class="n">SCRIPT_DCG</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{SCRIPT_DIR}</span><span class="s2">/../compute_detection_real_data/offline_dcg.py&quot;</span>
<span class="n">SCRIPT_KRONECKER</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{SCRIPT_DIR}</span><span class="s2">/../compute_detection_real_data/offline_kronecker.py&quot;</span>

<span class="n">TOTAL</span><span class="o">=</span><span class="mi">10</span>
<span class="n">CURRENT</span><span class="o">=</span><span class="mi">0</span>

<span class="n">run_benchmark</span><span class="p">()</span> <span class="p">{</span>
  <span class="n">local</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;$1&quot;</span>
  <span class="n">local</span> <span class="n">cmd</span><span class="o">=</span><span class="s2">&quot;$2&quot;</span>
  <span class="n">CURRENT</span><span class="o">=</span><span class="err">$</span><span class="p">((</span><span class="n">CURRENT</span> <span class="o">+</span> <span class="mi">1</span><span class="p">))</span>
  <span class="n">echo</span> <span class="s2">&quot;&quot;</span>
  <span class="n">echo</span> <span class="s2">&quot;[$CURRENT/$TOTAL] === $label ===&quot;</span>
  <span class="n">hyperfine</span> \
    <span class="o">--</span><span class="n">runs</span> <span class="s2">&quot;$RUNS&quot;</span> \
    <span class="o">--</span><span class="n">warmup</span> <span class="s2">&quot;$WARMUP&quot;</span> \
    <span class="o">--</span><span class="n">export</span><span class="o">-</span><span class="n">json</span> <span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/$</span><span class="si">{label}</span><span class="s2">.json&quot;</span> \
    <span class="o">--</span><span class="n">show</span><span class="o">-</span><span class="n">output</span> \
    <span class="s2">&quot;$cmd&quot;</span>
<span class="p">}</span>

<span class="c1"># ---- CPU benchmarks ----------------------------------------------------------</span>
<span class="n">run_benchmark</span> <span class="s2">&quot;cpu_gaussian_no_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_GAUSSIAN}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cpu --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;cpu_gaussian_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_GAUSSIAN}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cpu --wavelet --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;cpu_dcg_no_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_DCG}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cpu --iteration-chunk 512 --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;cpu_dcg_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_DCG}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cpu --wavelet --iteration-chunk 512 --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;cpu_kronecker_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_KRONECKER}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cpu --wavelet --quiet&quot;</span>

<span class="c1"># ---- GPU benchmarks ----------------------------------------------------------</span>
<span class="n">run_benchmark</span> <span class="s2">&quot;gpu_gaussian_no_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_GAUSSIAN}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cuda --splitting (1,1) --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;gpu_gaussian_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_GAUSSIAN}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cuda --wavelet --splitting (1,1) --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;gpu_dcg_no_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_DCG}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cuda --splitting (3,3) --iteration-chunk 512 --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;gpu_dcg_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_DCG}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cuda --wavelet --splitting (6,6) --iteration-chunk 512 --quiet&quot;</span>

<span class="n">run_benchmark</span> <span class="s2">&quot;gpu_kronecker_wavelet&quot;</span> \
  <span class="s2">&quot;uv run $</span><span class="si">{SCRIPT_KRONECKER}</span><span class="s2"> --data-path $</span><span class="si">{DATA}</span><span class="s2"> --window-size $</span><span class="si">{WINDOW_SIZE}</span><span class="s2"> --backend torch-cuda --wavelet --splitting (6,6) --quiet&quot;</span>

<span class="n">echo</span> <span class="s2">&quot;&quot;</span>
<span class="n">echo</span> <span class="s2">&quot;All benchmarks done. Aggregating results and generating chart...&quot;</span>

<span class="n">uv</span> <span class="n">run</span> <span class="o">-</span> <span class="s2">&quot;$RESULTS_DIR&quot;</span> <span class="o">&lt;&lt;</span><span class="s1">&#39;EOF&#39;</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">sys</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">csv</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="n">results_dir</span> <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="n">sys</span><span class="o">.</span><span class="n">argv</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>

<span class="n">labels</span> <span class="o">=</span> <span class="p">[</span>
    <span class="s2">&quot;cpu_gaussian_no_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;cpu_gaussian_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;cpu_dcg_no_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;cpu_dcg_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;gpu_gaussian_no_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;gpu_gaussian_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;gpu_dcg_no_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;gpu_dcg_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;cpu_kronecker_wavelet&quot;</span><span class="p">,</span>
    <span class="s2">&quot;gpu_kronecker_wavelet&quot;</span><span class="p">,</span>
<span class="p">]</span>

<span class="n">rows</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">label</span> <span class="ow">in</span> <span class="n">labels</span><span class="p">:</span>
    <span class="n">json_path</span> <span class="o">=</span> <span class="n">results_dir</span> <span class="o">/</span> <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">label</span><span class="si">}</span><span class="s2">.json&quot;</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">json_path</span><span class="o">.</span><span class="n">exists</span><span class="p">()</span> <span class="ow">or</span> <span class="n">json_path</span><span class="o">.</span><span class="n">stat</span><span class="p">()</span><span class="o">.</span><span class="n">st_size</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  WARNING: missing or empty </span><span class="si">{</span><span class="n">json_path</span><span class="si">}</span><span class="s2">, skipping.&quot;</span><span class="p">)</span>
        <span class="k">continue</span>
    <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">json_path</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
        <span class="n">data</span> <span class="o">=</span> <span class="n">json</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s2">&quot;results&quot;</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">mean</span> <span class="o">=</span> <span class="n">result</span><span class="p">[</span><span class="s2">&quot;mean&quot;</span><span class="p">]</span>
    <span class="n">std</span> <span class="o">=</span> <span class="n">result</span><span class="p">[</span><span class="s2">&quot;stddev&quot;</span><span class="p">]</span>
    <span class="n">rows</span><span class="o">.</span><span class="n">append</span><span class="p">({</span><span class="s2">&quot;label&quot;</span><span class="p">:</span> <span class="n">label</span><span class="p">,</span> <span class="s2">&quot;mean_s&quot;</span><span class="p">:</span> <span class="n">mean</span><span class="p">,</span> <span class="s2">&quot;std_s&quot;</span><span class="p">:</span> <span class="n">std</span><span class="p">})</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  </span><span class="si">{</span><span class="n">label</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">mean</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2"> +/- </span><span class="si">{</span><span class="n">std</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2"> s&quot;</span><span class="p">)</span>

<span class="c1"># Write CSV</span>
<span class="n">csv_path</span> <span class="o">=</span> <span class="n">results_dir</span> <span class="o">/</span> <span class="s2">&quot;benchmark_summary.csv&quot;</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">csv_path</span><span class="p">,</span> <span class="s2">&quot;w&quot;</span><span class="p">,</span> <span class="n">newline</span><span class="o">=</span><span class="s2">&quot;&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">writer</span> <span class="o">=</span> <span class="n">csv</span><span class="o">.</span><span class="n">DictWriter</span><span class="p">(</span><span class="n">f</span><span class="p">,</span> <span class="n">fieldnames</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">,</span> <span class="s2">&quot;mean_s&quot;</span><span class="p">,</span> <span class="s2">&quot;std_s&quot;</span><span class="p">])</span>
    <span class="n">writer</span><span class="o">.</span><span class="n">writeheader</span><span class="p">()</span>
    <span class="n">writer</span><span class="o">.</span><span class="n">writerows</span><span class="p">(</span><span class="n">rows</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="se">\n</span><span class="s2">CSV saved to </span><span class="si">{</span><span class="n">csv_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">matplotlib.patches</span><span class="w"> </span><span class="kn">import</span> <span class="n">Patch</span>

<span class="k">def</span><span class="w"> </span><span class="nf">make_chart</span><span class="p">(</span><span class="n">rows</span><span class="p">,</span> <span class="n">detector_name</span><span class="p">,</span> <span class="n">results_dir</span><span class="p">):</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">rows</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  No data for </span><span class="si">{</span><span class="n">detector_name</span><span class="si">}</span><span class="s2">, skipping chart.&quot;</span><span class="p">)</span>
        <span class="k">return</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">8</span><span class="p">,</span> <span class="mi">5</span><span class="p">))</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">rows</span><span class="p">))</span>
    <span class="n">means</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span><span class="p">[</span><span class="s2">&quot;mean_s&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span><span class="p">]</span>
    <span class="n">stds</span>  <span class="o">=</span> <span class="p">[</span><span class="n">r</span><span class="p">[</span><span class="s2">&quot;std_s&quot;</span><span class="p">]</span>  <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span><span class="p">]</span>
    <span class="n">tick_labels</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span><span class="p">]</span>
    <span class="n">colors</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;steelblue&quot;</span> <span class="k">if</span> <span class="n">r</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">&quot;cpu&quot;</span><span class="p">)</span> <span class="k">else</span> <span class="s2">&quot;darkorange&quot;</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span><span class="p">]</span>

    <span class="n">ax</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">means</span><span class="p">,</span> <span class="n">yerr</span><span class="o">=</span><span class="n">stds</span><span class="p">,</span> <span class="n">capsize</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">,</span> <span class="n">edgecolor</span><span class="o">=</span><span class="s2">&quot;black&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">0.6</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xticks</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xticklabels</span><span class="p">(</span><span class="n">tick_labels</span><span class="p">,</span> <span class="n">rotation</span><span class="o">=</span><span class="mi">30</span><span class="p">,</span> <span class="n">ha</span><span class="o">=</span><span class="s2">&quot;right&quot;</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">9</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;Time (s)&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">detector_name</span><span class="si">}</span><span class="s2"> detection time (mean +/- std)&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span><span class="n">handles</span><span class="o">=</span><span class="p">[</span>
        <span class="n">Patch</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="s2">&quot;steelblue&quot;</span><span class="p">,</span>  <span class="n">label</span><span class="o">=</span><span class="s2">&quot;CPU&quot;</span><span class="p">),</span>
        <span class="n">Patch</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="s2">&quot;darkorange&quot;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;GPU&quot;</span><span class="p">),</span>
    <span class="p">])</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="n">stem</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">&quot;benchmark_</span><span class="si">{</span><span class="n">detector_name</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span><span class="si">}</span><span class="s2">&quot;</span>
    <span class="n">png_path</span> <span class="o">=</span> <span class="n">results_dir</span> <span class="o">/</span> <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">stem</span><span class="si">}</span><span class="s2">.png&quot;</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">savefig</span><span class="p">(</span><span class="n">png_path</span><span class="p">,</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">150</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Chart saved to </span><span class="si">{</span><span class="n">png_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">try</span><span class="p">:</span>
        <span class="kn">import</span><span class="w"> </span><span class="nn">matplot2tikz</span>
        <span class="n">tex_path</span> <span class="o">=</span> <span class="n">results_dir</span> <span class="o">/</span> <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">stem</span><span class="si">}</span><span class="s2">.tex&quot;</span>
        <span class="n">matplot2tikz</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">tex_path</span><span class="p">))</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;TikZ saved to </span><span class="si">{</span><span class="n">tex_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">except</span> <span class="ne">ImportError</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;  matplot2tikz not installed, skipping .tex export.&quot;</span><span class="p">)</span>

    <span class="n">plt</span><span class="o">.</span><span class="n">close</span><span class="p">(</span><span class="n">fig</span><span class="p">)</span>

<span class="n">gaussian_rows</span>  <span class="o">=</span> <span class="p">[</span><span class="n">r</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span> <span class="k">if</span> <span class="s2">&quot;gaussian&quot;</span>  <span class="ow">in</span> <span class="n">r</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]]</span>
<span class="n">dcg_rows</span>       <span class="o">=</span> <span class="p">[</span><span class="n">r</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span> <span class="k">if</span> <span class="s2">&quot;dcg&quot;</span>       <span class="ow">in</span> <span class="n">r</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]]</span>
<span class="n">kronecker_rows</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span> <span class="k">if</span> <span class="s2">&quot;kronecker&quot;</span> <span class="ow">in</span> <span class="n">r</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]]</span>

<span class="n">make_chart</span><span class="p">(</span><span class="n">gaussian_rows</span><span class="p">,</span>  <span class="s2">&quot;Gaussian&quot;</span><span class="p">,</span>  <span class="n">results_dir</span><span class="p">)</span>
<span class="n">make_chart</span><span class="p">(</span><span class="n">dcg_rows</span><span class="p">,</span>       <span class="s2">&quot;DCG&quot;</span><span class="p">,</span>       <span class="n">results_dir</span><span class="p">)</span>
<span class="n">make_chart</span><span class="p">(</span><span class="n">kronecker_rows</span><span class="p">,</span> <span class="s2">&quot;Kronecker&quot;</span><span class="p">,</span> <span class="n">results_dir</span><span class="p">)</span>
<span class="n">EOF</span>

<span class="n">echo</span> <span class="s2">&quot;Done. Results in $</span><span class="si">{RESULTS_DIR}</span><span class="s2">/&quot;</span>
</pre></div>

</div>
</details>

## Config

`2-detection/experiments/sar/sar_bench_time.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
