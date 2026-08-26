<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sar_mc_kron_struct</span>
</nav>

# sar_mc_kron_struct

Ce que la structure Kronecker achete : erreur vs taille de fenetre N

**Tags:** `detection`  `kronecker`  `estimation`  `structure`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_kron_structure_vs_n.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sar_experiments/mc_simulations/mc_kron_structure_vs_n.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">219 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sar_experiments/mc_simulations/mc_kron_structure_vs_n.py</p>
<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/env python</span>
<span class="sd">&quot;&quot;&quot;What the Kronecker structure buys: estimation error against the patch size N.</span>

<span class="sd">At a fixed number of dates T, the same H0 data is fitted twice with the same</span>
<span class="sd">geometry, the same gradient and the same line search, once with the shape</span>
<span class="sd">matrix constrained to A (x) B and once with it free in sH++(p).  The error</span>
<span class="sd">reported is the total squared Riemannian distance d^2_M of equation (18),</span>
<span class="sd">averaged over trials, together with the intrinsic Cramer-Rao bound of each</span>
<span class="sd">parametrisation.</span>

<span class="sd">The gap between the two bounds is a ratio of dimensions,</span>
<span class="sd">((a^2-1) + (b^2-1) + N) / ((p^2-1) + N): wide when the patch is small, closing</span>
<span class="sd">as N grows.  This is the same trade -- constrain the model to lower the sample</span>
<span class="sd">support needed -- that the multi-ping sonar section makes with Kronecker plus</span>
<span class="sd">Toeplitz, measured here on the change detection model.</span>

<span class="sd">Backend selection:</span>
<span class="sd">  numpy     → multiprocessing.Pool, one trial per worker</span>
<span class="sd">  all other → trials in leading batch dim, single-pass on device</span>
<span class="sd">&quot;&quot;&quot;</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">multiprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Pool</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">rich.progress</span><span class="w"> </span><span class="kn">import</span> <span class="n">BarColumn</span><span class="p">,</span> <span class="n">Progress</span><span class="p">,</span> <span class="n">TextColumn</span><span class="p">,</span> <span class="n">TimeElapsedColumn</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="n">scaled_gaussian_riemannian_gd_h0</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">MCResultExporter</span><span class="p">,</span>
    <span class="n">init_logging</span><span class="p">,</span>
    <span class="n">make_mc_parser</span><span class="p">,</span>
    <span class="n">maybe_empty_cache</span><span class="p">,</span>
    <span class="n">timed_run</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.simulation</span><span class="w"> </span><span class="kn">import</span> <span class="n">make_ab_toeplitz</span><span class="p">,</span> <span class="n">generate_kronecker_data</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.estimation_kronecker</span><span class="w"> </span><span class="kn">import</span> <span class="n">kronecker_mm_h0</span><span class="p">,</span> <span class="n">kronecker_riemannian_gd_h0</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.icrb</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">icrb_kronecker_scaled_gaussian</span><span class="p">,</span>
    <span class="n">icrb_scaled_gaussian</span><span class="p">,</span>
    <span class="n">kronecker_component_errors</span><span class="p">,</span>
    <span class="n">scaled_gaussian_component_errors</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.mc</span><span class="w"> </span><span class="kn">import</span> <span class="n">_MC_PLOT_TEMPLATE_STRUCT</span><span class="p">,</span> <span class="n">add_mc_base_args</span><span class="p">,</span> <span class="n">finish_struct</span>

<span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># One (N, trial batch) evaluation</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_fit_both</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Fit structured and unstructured models on X of shape (..., T, N, p).&quot;&quot;&quot;</span>
    <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">tau_true</span> <span class="o">=</span> <span class="n">truth</span>
    <span class="n">p</span> <span class="o">=</span> <span class="n">a</span> <span class="o">*</span> <span class="n">b</span>

    <span class="k">if</span> <span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;offline&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;gd&quot;</span><span class="p">:</span>
        <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau_k</span> <span class="o">=</span> <span class="n">kronecker_riemannian_gd_h0</span><span class="p">(</span>
            <span class="n">X</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;gd_iter_max&quot;</span><span class="p">],</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;gd_tol&quot;</span><span class="p">],</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau_flat</span> <span class="o">=</span> <span class="n">kronecker_mm_h0</span><span class="p">(</span>
            <span class="n">X</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;mm_tol&quot;</span><span class="p">],</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;mm_iter_max&quot;</span><span class="p">],</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>
        <span class="n">tau_k</span> <span class="o">=</span> <span class="n">tau_flat</span><span class="p">[</span><span class="o">...</span><span class="p">,</span> <span class="kc">None</span><span class="p">]</span> <span class="k">if</span> <span class="n">tau_flat</span><span class="o">.</span><span class="n">ndim</span> <span class="o">==</span> <span class="n">X</span><span class="o">.</span><span class="n">ndim</span> <span class="o">-</span> <span class="mi">2</span> <span class="k">else</span> <span class="n">tau_flat</span>
    <span class="n">err_k</span> <span class="o">=</span> <span class="n">kronecker_component_errors</span><span class="p">(</span>
        <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau_k</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>

    <span class="n">Sigma</span><span class="p">,</span> <span class="n">tau_f</span> <span class="o">=</span> <span class="n">scaled_gaussian_riemannian_gd_h0</span><span class="p">(</span>
        <span class="n">X</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;gd_iter_max&quot;</span><span class="p">],</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;gd_tol&quot;</span><span class="p">],</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">Sigma_true</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">kron</span><span class="p">(</span><span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">)</span>
    <span class="n">err_f</span> <span class="o">=</span> <span class="n">scaled_gaussian_component_errors</span><span class="p">(</span>
        <span class="n">Sigma</span><span class="p">,</span> <span class="n">tau_f</span><span class="p">,</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">Sigma_true</span><span class="p">,</span> <span class="n">backend</span><span class="p">),</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">p</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span>
        <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">err_k</span><span class="p">[</span><span class="s2">&quot;total&quot;</span><span class="p">],</span> <span class="n">err_f</span><span class="p">[</span><span class="s2">&quot;total&quot;</span><span class="p">]</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_worker</span><span class="p">(</span><span class="n">worker_args</span><span class="p">):</span>
    <span class="n">X</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span> <span class="o">=</span> <span class="n">worker_args</span>
    <span class="n">e_k</span><span class="p">,</span> <span class="n">e_f</span> <span class="o">=</span> <span class="n">_fit_both</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="kc">None</span><span class="p">],</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="p">(</span><span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">),</span> <span class="n">cfg</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>
    <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">to_numpy</span><span class="p">(</span><span class="n">e_k</span><span class="p">))),</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">to_numpy</span><span class="p">(</span><span class="n">e_f</span><span class="p">)))</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_run_pool</span><span class="p">(</span><span class="n">datasets</span><span class="p">,</span> <span class="n">N_vec</span><span class="p">,</span> <span class="n">n_workers</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">):</span>
    <span class="n">kron_err</span><span class="p">,</span> <span class="n">full_err</span> <span class="o">=</span> <span class="p">{},</span> <span class="p">{}</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Starting structured/unstructured comparison over </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">N_vec</span><span class="p">)</span><span class="si">}</span><span class="s2"> values of N (Pool)...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;[progress.description]</span><span class="si">{task.description}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">BarColumn</span><span class="p">(),</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{task.completed}</span><span class="s2">/</span><span class="si">{task.total}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">TimeElapsedColumn</span><span class="p">(),</span>
    <span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="s2">&quot;[cyan]N values...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">N_vec</span><span class="p">))</span>
        <span class="k">for</span> <span class="n">N</span> <span class="ow">in</span> <span class="n">N_vec</span><span class="p">:</span>
            <span class="n">data</span><span class="p">,</span> <span class="n">tau_true</span> <span class="o">=</span> <span class="n">datasets</span><span class="p">[</span><span class="n">N</span><span class="p">]</span>
            <span class="n">worker_args</span> <span class="o">=</span> <span class="p">[</span>
                <span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">tau_true</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">)</span>
                <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
            <span class="p">]</span>
            <span class="k">with</span> <span class="n">Pool</span><span class="p">(</span><span class="n">processes</span><span class="o">=</span><span class="n">n_workers</span><span class="p">)</span> <span class="k">as</span> <span class="n">pool</span><span class="p">:</span>
                <span class="n">results</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="n">pool</span><span class="o">.</span><span class="n">imap_unordered</span><span class="p">(</span><span class="n">_worker</span><span class="p">,</span> <span class="n">worker_args</span><span class="p">))</span>
            <span class="n">kron_err</span><span class="p">[</span><span class="n">N</span><span class="p">]</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">r</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">results</span><span class="p">])</span>
            <span class="n">full_err</span><span class="p">[</span><span class="n">N</span><span class="p">]</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">r</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">results</span><span class="p">])</span>
            <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">kron_err</span><span class="p">,</span> <span class="n">full_err</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_run_batched</span><span class="p">(</span><span class="n">datasets</span><span class="p">,</span> <span class="n">N_vec</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">):</span>
    <span class="n">kron_err</span><span class="p">,</span> <span class="n">full_err</span> <span class="o">=</span> <span class="p">{},</span> <span class="p">{}</span>
    <span class="n">A_t</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">A_true</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">B_t</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">B_true</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Starting batched structured/unstructured comparison on </span><span class="si">{</span><span class="n">backend</span><span class="si">}</span><span class="s2">...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;[progress.description]</span><span class="si">{task.description}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">BarColumn</span><span class="p">(),</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{task.completed}</span><span class="s2">/</span><span class="si">{task.total}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">TimeElapsedColumn</span><span class="p">(),</span>
    <span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="s2">&quot;[cyan]N values...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">N_vec</span><span class="p">))</span>
        <span class="k">for</span> <span class="n">N</span> <span class="ow">in</span> <span class="n">N_vec</span><span class="p">:</span>
            <span class="n">data</span><span class="p">,</span> <span class="n">tau_true</span> <span class="o">=</span> <span class="n">datasets</span><span class="p">[</span><span class="n">N</span><span class="p">]</span>
            <span class="n">X</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
            <span class="n">truth</span> <span class="o">=</span> <span class="p">(</span><span class="n">A_t</span><span class="p">,</span> <span class="n">B_t</span><span class="p">,</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">tau_true</span><span class="p">,</span> <span class="n">backend</span><span class="p">))</span>
            <span class="n">e_k</span><span class="p">,</span> <span class="n">e_f</span> <span class="o">=</span> <span class="n">_fit_both</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
            <span class="n">kron_err</span><span class="p">[</span><span class="n">N</span><span class="p">]</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">e_k</span><span class="p">)</span>
            <span class="n">full_err</span><span class="p">[</span><span class="n">N</span><span class="p">]</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">e_f</span><span class="p">)</span>
            <span class="n">maybe_empty_cache</span><span class="p">(</span><span class="n">backend</span><span class="p">)</span>
            <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">kron_err</span><span class="p">,</span> <span class="n">full_err</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Entry point</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span><span class="vm">__doc__</span><span class="p">)</span>
    <span class="n">add_mc_base_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">set_defaults</span><span class="p">(</span><span class="n">n_trials</span><span class="o">=</span><span class="mi">200</span><span class="p">)</span>

    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--a&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Size of the first Kronecker factor.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--b&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Size of the second Kronecker factor.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--T&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">25</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of dates, held fixed while N varies (default 25).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--N-list&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">4</span><span class="p">,</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">9</span><span class="p">,</span> <span class="mi">13</span><span class="p">,</span> <span class="mi">20</span><span class="p">,</span> <span class="mi">30</span><span class="p">,</span> <span class="mi">45</span><span class="p">,</span> <span class="mi">70</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Patch sizes to sweep (default 4 6 9 13 20 30 45 70; p = a*b = 12).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--nu&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-a&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.3+0.7j&quot;</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of A.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-b&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.3+0.6j&quot;</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of B.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--offline&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;gd&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;mm&quot;</span><span class="p">,</span> <span class="s2">&quot;gd&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Structured estimator: &#39;gd&#39; Riemannian gradient descent, term for term &quot;</span>
             <span class="s2">&quot;comparable with the unstructured one (default), or &#39;mm&#39;.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--mm-iter-max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Max MM iterations.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--mm-tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-8</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;MM tolerance.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--gd-iter-max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Max GD iterations.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--gd-tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-8</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;GD tolerance.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--debug&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Tiny configuration (6 trials, 3 values of N, T=8) to validate the &quot;</span>
             <span class="s2">&quot;pipeline in seconds. Results are NOT publication grade.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">N_list</span> <span class="o">=</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="p">[</span><span class="mi">6</span><span class="p">,</span> <span class="mi">13</span><span class="p">,</span> <span class="mi">30</span><span class="p">]</span>
        <span class="n">args</span><span class="o">.</span><span class="n">gd_iter_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">mm_iter_max</span> <span class="o">=</span> <span class="mi">50</span><span class="p">,</span> <span class="mi">20</span>

    <span class="n">init_logging</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span><span class="s2">&quot;--debug: 6 trials, T=8, N in {6, 13, 30}. Pipeline check only.&quot;</span><span class="p">)</span>

    <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">p</span><span class="p">,</span> <span class="n">T</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">a</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">b</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">a</span> <span class="o">*</span> <span class="n">args</span><span class="o">.</span><span class="n">b</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T</span>
    <span class="n">N_vec</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">(</span><span class="nb">set</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">N_list</span><span class="p">))</span>

    <span class="n">cfg</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;offline&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">offline</span><span class="p">,</span>
        <span class="s2">&quot;mm_iter_max&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">mm_iter_max</span><span class="p">,</span>
        <span class="s2">&quot;mm_tol&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">mm_tol</span><span class="p">,</span>
        <span class="s2">&quot;gd_iter_max&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">gd_iter_max</span><span class="p">,</span>
        <span class="s2">&quot;gd_tol&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">gd_tol</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Structure vs N: a=</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">, b=</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">, p=</span><span class="si">{</span><span class="n">p</span><span class="si">}</span><span class="s2">, T=</span><span class="si">{</span><span class="n">T</span><span class="si">}</span><span class="s2">, N=</span><span class="si">{</span><span class="n">N_vec</span><span class="si">}</span><span class="s2">, &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;n_trials=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">, backend=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  texture ~ Gamma(</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="mi">1</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">:</span><span class="s2">.3g</span><span class="si">}</span><span class="s2">) | structured estimator=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">offline</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span> <span class="o">=</span> <span class="n">make_ab_toeplitz</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_a</span><span class="p">),</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_b</span><span class="p">))</span>

    <span class="n">datasets</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">N</span> <span class="ow">in</span> <span class="n">N_vec</span><span class="p">:</span>
        <span class="n">datasets</span><span class="p">[</span><span class="n">N</span><span class="p">]</span> <span class="o">=</span> <span class="n">generate_kronecker_data</span><span class="p">(</span>
            <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">T</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span>
            <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="n">N</span><span class="p">,</span> <span class="n">tau_shape</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="p">,</span> <span class="n">tau_scale</span><span class="o">=</span><span class="mf">1.0</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="p">,</span> <span class="n">return_tau</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
        <span class="p">)</span>

    <span class="n">exporter</span> <span class="o">=</span> <span class="n">MCResultExporter</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span> <span class="n">Path</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span> <span class="sa">f</span><span class="s2">&quot;a</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">_b</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">_T</span><span class="si">{</span><span class="n">T</span><span class="si">}</span><span class="s2">_n</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span>
        <span class="n">plot_template</span><span class="o">=</span><span class="n">_MC_PLOT_TEMPLATE_STRUCT</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="p">(</span><span class="n">kron_err</span><span class="p">,</span> <span class="n">full_err</span><span class="p">),</span> <span class="n">elapsed</span> <span class="o">=</span> <span class="n">timed_run</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_pool</span><span class="p">(</span><span class="n">datasets</span><span class="p">,</span> <span class="n">N_vec</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_workers</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">),</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_batched</span><span class="p">(</span><span class="n">datasets</span><span class="p">,</span> <span class="n">N_vec</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">),</span>
    <span class="p">)</span>

    <span class="n">icrb_kron</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">icrb_kronecker_scaled_gaussian</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">T</span><span class="p">)[</span><span class="s2">&quot;total&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">N</span> <span class="ow">in</span> <span class="n">N_vec</span><span class="p">])</span>
    <span class="n">icrb_full</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">icrb_scaled_gaussian</span><span class="p">(</span><span class="n">p</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">T</span><span class="p">)[</span><span class="s2">&quot;total&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">N</span> <span class="ow">in</span> <span class="n">N_vec</span><span class="p">])</span>

    <span class="n">title</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">&quot;Ce que la structure achète  (a=</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">, b=</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">, T=</span><span class="si">{</span><span class="n">T</span><span class="si">}</span><span class="s2">, nu=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">}</span><span class="s2">)&quot;</span>
    <span class="n">finish_struct</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">exporter</span><span class="p">,</span> <span class="n">kron_err</span><span class="p">,</span> <span class="n">full_err</span><span class="p">,</span> <span class="n">icrb_kron</span><span class="p">,</span> <span class="n">icrb_full</span><span class="p">,</span> <span class="n">N_vec</span><span class="p">,</span>
                  <span class="s2">&quot;mc_kron_struct&quot;</span><span class="p">,</span> <span class="n">title</span><span class="p">,</span> <span class="n">elapsed</span><span class="p">)</span>


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
<span class="param-flag">--T</span><span class="param-type">int</span><span class="param-default">default <b>25</b></span>
</div>
<p class="param-help">Number of dates, held fixed while N varies (default 25).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--N-list</span><span class="param-type">int</span><span class="param-default">default <b>[4, 6, 9, 13, 20, 30, 45, 70]</b></span>
</div>
<p class="param-help">Patch sizes to sweep (default 4 6 9 13 20 30 45 70; p = a*b = 12).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--nu</span><span class="param-type">float</span><span class="param-default">default <b>1.0</b></span>
</div>
<p class="param-help">Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-a</span><span class="param-type">str</span><span class="param-default">default <b>0.3+0.7j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of A.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-b</span><span class="param-type">str</span><span class="param-default">default <b>0.3+0.6j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of B.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--offline</span><span class="param-type">str</span><span class="param-default">default <b>gd</b></span>
</div>
<p class="param-help">Structured estimator: &#x27;gd&#x27; Riemannian gradient descent, term for term comparable with the unstructured one (default), or &#x27;mm&#x27;.</p><p class="param-choices">choices: mm, gd</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--mm-iter-max</span><span class="param-type">int</span><span class="param-default">default <b>50</b></span>
</div>
<p class="param-help">Max MM iterations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--mm-tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-08</b></span>
</div>
<p class="param-help">MM tolerance.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--gd-iter-max</span><span class="param-type">int</span><span class="param-default">default <b>200</b></span>
</div>
<p class="param-help">Max GD iterations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--gd-tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-08</b></span>
</div>
<p class="param-help">GD tolerance.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--debug</span><span class="param-type">flag</span>
</div>
<p class="param-help">Tiny configuration (6 trials, 3 values of N, T=8) to validate the pipeline in seconds. Results are NOT publication grade.</p>
</div>
</div>

## Config

`2-detection/experiments/sar/sar_mc_kron_struct.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
