<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sar_bench_memory</span>
</nav>

# sar_bench_memory

Memory benchmark for offline Gaussian and DCG GLRT detectors (CPU memray + GPU torch)

**Tags:** `benchmark`  `memory`  `SAR`

## Run

```sh
bash 2-detection/sar_experiments/benchmarks/memory_benchmark.sh
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sar_experiments/benchmarks/memory_benchmark.sh" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">127 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sar_experiments/benchmarks/memory_benchmark.sh</p>
<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/env bash</span>
<span class="c1"># Memory benchmark for detection scripts.</span>
<span class="c1"># CPU: uses memray -- generates flamegraph + peak summary.</span>
<span class="c1"># GPU: uses torch.cuda.max_memory_allocated() reported by the script itself.</span>
<span class="c1"># Results saved per config, summary printed at end.</span>

<span class="nb">set</span> <span class="o">-</span><span class="n">euo</span> <span class="n">pipefail</span>

<span class="n">SCRIPT_DIR</span><span class="o">=</span><span class="s2">&quot;$(cd &quot;</span><span class="err">$</span><span class="p">(</span><span class="n">dirname</span> <span class="s2">&quot;$</span><span class="si">{BASH_SOURCE[0]}</span><span class="s2">&quot;</span><span class="p">)</span><span class="s2">&quot; &amp;&amp; pwd)&quot;</span>
<span class="n">DATA</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{SCRIPT_DIR}</span><span class="s2">/../../data/SAR/Scene_1.npy&quot;</span>
<span class="n">WINDOW_SIZE</span><span class="o">=</span><span class="mi">7</span>
<span class="n">RESULTS_DIR</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{SCRIPT_DIR}</span><span class="s2">/memory_results&quot;</span>

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

<span class="c1"># ---- CPU: run under memray, extract peak RSS from stats output ---------------</span>
<span class="n">run_cpu_memory</span><span class="p">()</span> <span class="p">{</span>
  <span class="n">local</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;$1&quot;</span>
  <span class="n">local</span> <span class="n">script</span><span class="o">=</span><span class="s2">&quot;$2&quot;</span>
  <span class="n">local</span> <span class="n">extra_args</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{3:-}</span><span class="s2">&quot;</span>
  <span class="n">CURRENT</span><span class="o">=</span><span class="err">$</span><span class="p">((</span><span class="n">CURRENT</span> <span class="o">+</span> <span class="mi">1</span><span class="p">))</span>
  <span class="n">echo</span> <span class="s2">&quot;&quot;</span>
  <span class="n">echo</span> <span class="s2">&quot;[$CURRENT/$TOTAL] === $label (CPU memray) ===&quot;</span>

  <span class="n">local</span> <span class="nb">bin</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/$</span><span class="si">{label}</span><span class="s2">.bin&quot;</span>
  <span class="n">local</span> <span class="n">html</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/$</span><span class="si">{label}</span><span class="s2">.html&quot;</span>

  <span class="n">uv</span> <span class="n">run</span> <span class="n">python</span> <span class="o">-</span><span class="n">m</span> <span class="n">memray</span> <span class="n">run</span> <span class="o">--</span><span class="n">force</span> <span class="o">-</span><span class="n">o</span> <span class="s2">&quot;$bin&quot;</span> \
    <span class="s2">&quot;$script&quot;</span> <span class="s2">&quot;$DATA&quot;</span> <span class="s2">&quot;$WINDOW_SIZE&quot;</span> <span class="err">$</span><span class="n">extra_args</span> <span class="o">--</span><span class="n">quiet</span>

  <span class="n">uv</span> <span class="n">run</span> <span class="n">python</span> <span class="o">-</span><span class="n">m</span> <span class="n">memray</span> <span class="n">flamegraph</span> <span class="o">--</span><span class="n">force</span> <span class="o">-</span><span class="n">o</span> <span class="s2">&quot;$html&quot;</span> <span class="s2">&quot;$bin&quot;</span>
  <span class="n">echo</span> <span class="s2">&quot;  Flamegraph: $html&quot;</span>

  <span class="c1"># Extract peak memory from memray stats.</span>
  <span class="c1"># Use grep -E + head for macOS compatibility (grep -oP is GNU-only).</span>
  <span class="n">local</span> <span class="n">peak</span>
  <span class="n">peak</span><span class="o">=</span><span class="err">$</span><span class="p">(</span><span class="n">uv</span> <span class="n">run</span> <span class="n">python</span> <span class="o">-</span><span class="n">m</span> <span class="n">memray</span> <span class="n">stats</span> <span class="s2">&quot;$bin&quot;</span> <span class="mi">2</span><span class="o">&gt;/</span><span class="n">dev</span><span class="o">/</span><span class="n">null</span> \
    <span class="o">|</span> <span class="n">grep</span> <span class="o">-</span><span class="n">i</span> <span class="s2">&quot;peak memory usage&quot;</span> <span class="o">-</span><span class="n">A1</span> \
    <span class="o">|</span> <span class="n">tail</span> <span class="o">-</span><span class="mi">1</span> \
    <span class="o">|</span> <span class="n">grep</span> <span class="o">-</span><span class="n">Eo</span> <span class="s1">&#39;[0-9]+(\.[0-9]+)?[A-Za-z]+&#39;</span> \
    <span class="o">|</span> <span class="n">head</span> <span class="o">-</span><span class="mi">1</span> <span class="o">||</span> <span class="n">echo</span> <span class="s2">&quot;N/A&quot;</span><span class="p">)</span>
  <span class="n">echo</span> <span class="s2">&quot;  Peak memory: $peak&quot;</span>
  <span class="n">echo</span> <span class="s2">&quot;$</span><span class="si">{label}</span><span class="s2">,$</span><span class="si">{peak}</span><span class="s2">&quot;</span> <span class="o">&gt;&gt;</span><span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/memory_summary.csv&quot;</span>
<span class="p">}</span>

<span class="c1"># ---- GPU: script prints PEAK_GPU_MEMORY_BYTES=&lt;n&gt;, we parse it ---------------</span>
<span class="n">run_gpu_memory</span><span class="p">()</span> <span class="p">{</span>
  <span class="n">local</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;$1&quot;</span>
  <span class="n">local</span> <span class="n">script</span><span class="o">=</span><span class="s2">&quot;$2&quot;</span>
  <span class="n">local</span> <span class="n">extra_args</span><span class="o">=</span><span class="s2">&quot;$</span><span class="si">{3:-}</span><span class="s2">&quot;</span>
  <span class="n">CURRENT</span><span class="o">=</span><span class="err">$</span><span class="p">((</span><span class="n">CURRENT</span> <span class="o">+</span> <span class="mi">1</span><span class="p">))</span>
  <span class="n">echo</span> <span class="s2">&quot;&quot;</span>
  <span class="n">echo</span> <span class="s2">&quot;[$CURRENT/$TOTAL] === $label (GPU torch) ===&quot;</span>

  <span class="n">local</span> <span class="n">output</span>
  <span class="n">output</span><span class="o">=</span><span class="err">$</span><span class="p">(</span><span class="n">uv</span> <span class="n">run</span> <span class="n">python</span> <span class="s2">&quot;$script&quot;</span> <span class="s2">&quot;$DATA&quot;</span> <span class="s2">&quot;$WINDOW_SIZE&quot;</span> \
    <span class="o">--</span><span class="n">backend</span> <span class="n">torch</span><span class="o">-</span><span class="n">cuda</span> <span class="err">$</span><span class="n">extra_args</span> <span class="o">--</span><span class="n">quiet</span> <span class="o">--</span><span class="n">report</span><span class="o">-</span><span class="n">memory</span> <span class="mi">2</span><span class="o">&gt;&amp;</span><span class="mi">1</span><span class="p">)</span>

  <span class="n">local</span> <span class="n">peak_bytes</span>
  <span class="n">peak_bytes</span><span class="o">=</span><span class="err">$</span><span class="p">(</span><span class="n">echo</span> <span class="s2">&quot;$output&quot;</span> <span class="o">|</span> <span class="n">grep</span> <span class="s2">&quot;^PEAK_GPU_MEMORY_BYTES=&quot;</span> <span class="o">|</span> <span class="n">cut</span> <span class="o">-</span><span class="n">d</span><span class="o">=</span> <span class="o">-</span><span class="n">f2</span> <span class="o">||</span> <span class="n">echo</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>

  <span class="k">if</span> <span class="p">[</span> <span class="o">-</span><span class="n">n</span> <span class="s2">&quot;$peak_bytes&quot;</span> <span class="p">];</span> <span class="n">then</span>
    <span class="n">local</span> <span class="n">peak_mb</span>
    <span class="n">peak_mb</span><span class="o">=</span><span class="err">$</span><span class="p">(</span><span class="n">echo</span> <span class="s2">&quot;scale=1; $peak_bytes / 1048576&quot;</span> <span class="o">|</span> <span class="n">bc</span><span class="p">)</span>
    <span class="n">echo</span> <span class="s2">&quot;  Peak GPU memory: $</span><span class="si">{peak_mb}</span><span class="s2"> MB ($</span><span class="si">{peak_bytes}</span><span class="s2"> bytes)&quot;</span>
    <span class="n">echo</span> <span class="s2">&quot;$</span><span class="si">{label}</span><span class="s2">,$</span><span class="si">{peak_mb}</span><span class="s2"> MB&quot;</span> <span class="o">&gt;&gt;</span><span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/memory_summary.csv&quot;</span>
  <span class="k">else</span>
    <span class="n">echo</span> <span class="s2">&quot;  Could not parse GPU memory.&quot;</span>
    <span class="n">echo</span> <span class="s2">&quot;$</span><span class="si">{label}</span><span class="s2">,N/A&quot;</span> <span class="o">&gt;&gt;</span><span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/memory_summary.csv&quot;</span>
  <span class="n">fi</span>
<span class="p">}</span>

<span class="c1"># ---- Init summary CSV --------------------------------------------------------</span>
<span class="n">echo</span> <span class="s2">&quot;label,peak_memory&quot;</span> <span class="o">&gt;</span><span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/memory_summary.csv&quot;</span>

<span class="c1"># ---- CPU benchmarks ----------------------------------------------------------</span>
<span class="n">run_cpu_memory</span> <span class="s2">&quot;cpu_gaussian_no_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_GAUSSIAN&quot;</span> \
  <span class="s2">&quot;--backend torch-cpu&quot;</span>

<span class="n">run_cpu_memory</span> <span class="s2">&quot;cpu_gaussian_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_GAUSSIAN&quot;</span> \
  <span class="s2">&quot;--backend torch-cpu --wavelet&quot;</span>

<span class="n">run_cpu_memory</span> <span class="s2">&quot;cpu_dcg_no_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_DCG&quot;</span> \
  <span class="s2">&quot;--backend torch-cpu --iteration-chunk 512&quot;</span>

<span class="n">run_cpu_memory</span> <span class="s2">&quot;cpu_dcg_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_DCG&quot;</span> \
  <span class="s2">&quot;--backend torch-cpu --wavelet --iteration-chunk 512&quot;</span>

<span class="n">run_cpu_memory</span> <span class="s2">&quot;cpu_kronecker_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_KRONECKER&quot;</span> \
  <span class="s2">&quot;--backend torch-cpu --wavelet&quot;</span>

<span class="c1"># ---- GPU benchmarks ----------------------------------------------------------</span>
<span class="n">run_gpu_memory</span> <span class="s2">&quot;gpu_gaussian_no_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_GAUSSIAN&quot;</span> \
  <span class="s2">&quot;--splitting (1,1)&quot;</span>

<span class="n">run_gpu_memory</span> <span class="s2">&quot;gpu_gaussian_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_GAUSSIAN&quot;</span> \
  <span class="s2">&quot;--wavelet --splitting (1,1)&quot;</span>

<span class="n">run_gpu_memory</span> <span class="s2">&quot;gpu_dcg_no_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_DCG&quot;</span> \
  <span class="s2">&quot;--splitting (3,3) --iteration-chunk 512&quot;</span>

<span class="n">run_gpu_memory</span> <span class="s2">&quot;gpu_dcg_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_DCG&quot;</span> \
  <span class="s2">&quot;--wavelet --splitting (6,6) --iteration-chunk 512&quot;</span>

<span class="n">run_gpu_memory</span> <span class="s2">&quot;gpu_kronecker_wavelet&quot;</span> <span class="s2">&quot;$SCRIPT_KRONECKER&quot;</span> \
  <span class="s2">&quot;--wavelet --splitting (6,6)&quot;</span>

<span class="n">echo</span> <span class="s2">&quot;&quot;</span>
<span class="n">echo</span> <span class="s2">&quot;Memory benchmark done. Summary:&quot;</span>
<span class="n">cat</span> <span class="s2">&quot;$</span><span class="si">{RESULTS_DIR}</span><span class="s2">/memory_summary.csv&quot;</span>
<span class="n">echo</span> <span class="s2">&quot;&quot;</span>
<span class="n">echo</span> <span class="s2">&quot;Flamegraphs and raw .bin profiles saved in $</span><span class="si">{RESULTS_DIR}</span><span class="s2">/&quot;</span>
</pre></div>

</div>
</details>

## Config

`2-detection/experiments/sar/sar_bench_memory.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
