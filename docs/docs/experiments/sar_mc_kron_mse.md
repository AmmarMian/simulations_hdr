<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/2-detection/">2 · Detection</a>
<span class="sep">/</span>
<span class="here">sar_mc_kron_mse</span>
</nav>

# sar_mc_kron_mse

Mean squared error of the Kronecker estimators, offline and recursive, against the intrinsic Cramer-Rao bounds

**Tags:** `detection`  `kronecker`  `estimation`  `icrb`  `monte-carlo`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_kron_mse_icrb.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--n-trials 1000</code><br>
  <code>--seed 42</code><br>
  <code>--backend numpy</code><br>
  <code>--n-features 8</code><br>
  <code>--T-max 1000</code><br>
  <code>--T-min 2</code><br>
  <code>--n-T 12</code><br>
  <code>--sigma-seed 0</code><br>
  <code>--a 3</code><br>
  <code>--b 4</code><br>
  <code>--nu 1.0</code><br>
  <code>--rho-a 0.3+0.7j</code><br>
  <code>--rho-b 0.3+0.6j</code><br>
  <code>--offline mm</code><br>
  <code>--step-rule fixed</code><br>
  <code>--init-mode mm</code><br>
  <code>--mm-iter-max 50</code><br>
  <code>--mm-tol 1e-08</code><br>
  <code>--gd-iter-max 200</code><br>
  <code>--gd-tol 1e-08</code><br>
  <span class="mn-date">9512517 · 2026-08-25</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/2-detection/sar_experiments/mc_simulations/mc_kron_mse_icrb.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">279 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">2-detection/sar_experiments/mc_simulations/mc_kron_mse_icrb.py</p>
<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/env python</span>
<span class="sd">&quot;&quot;&quot;MSE vs T of the Kronecker scaled-Gaussian estimators, against the ICRB.</span>

<span class="sd">Reproduces Figures 2 and 3 of Mian et al., Signal Processing 224 (2024), with</span>
<span class="sd">the setup of the released configuration rather than of the body text: Toeplitz</span>
<span class="sd">factors of unit determinant, a=3, b=4, n=a*b+1=13, K-distributed data (texture</span>
<span class="sd">Gamma(nu, 1/nu), nu=1). The body text of Section 5.1 announces a=4, b=3, n=8</span>
<span class="sd">and random factors of condition number 10; the published figure captions and</span>
<span class="sd">the released code both use the setup implemented here.</span>

<span class="sd">Measured quantities, per component of theta = (A, B, tau): the squared</span>
<span class="sd">geodesic distances of equation (18), averaged over trials, for</span>

<span class="sd">  * the offline MLE (MM by default, Riemannian gradient descent with --offline gd),</span>
<span class="sd">  * the recursive estimator of equation (19), one gradient step per new date,</span>

<span class="sd">together with the intrinsic Cramer-Rao bounds of equation (25).</span>

<span class="sd">Backend selection:</span>
<span class="sd">  numpy     → multiprocessing.Pool, one trial per worker</span>
<span class="sd">  all other → trials in leading batch dim, single-pass on device</span>
<span class="sd">&quot;&quot;&quot;</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">multiprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Pool</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">rich.progress</span><span class="w"> </span><span class="kn">import</span> <span class="n">BarColumn</span><span class="p">,</span> <span class="n">Progress</span> <span class="k">as</span> <span class="n">RichProgress</span><span class="p">,</span> <span class="n">TextColumn</span><span class="p">,</span> <span class="n">TimeElapsedColumn</span>

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
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.simulation</span><span class="w"> </span><span class="kn">import</span> <span class="n">make_ab_toeplitz</span><span class="p">,</span> <span class="n">generate_kronecker_data</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.estimation_kronecker</span><span class="w"> </span><span class="kn">import</span> <span class="n">kronecker_mm_h0</span><span class="p">,</span> <span class="n">kronecker_riemannian_gd_h0</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.estimation_online</span><span class="w"> </span><span class="kn">import</span> <span class="n">OnlineKroneckerEstimator</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.icrb</span><span class="w"> </span><span class="kn">import</span> <span class="n">icrb_kronecker_scaled_gaussian</span><span class="p">,</span> <span class="n">kronecker_component_errors</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.sar.mc</span><span class="w"> </span><span class="kn">import</span> <span class="n">_MC_PLOT_TEMPLATE_MSE</span><span class="p">,</span> <span class="n">add_mc_args</span><span class="p">,</span> <span class="n">finish_mse</span>

<span class="n">logger</span> <span class="o">=</span> <span class="n">logging</span><span class="o">.</span><span class="n">getLogger</span><span class="p">(</span><span class="vm">__name__</span><span class="p">)</span>

<span class="n">_COMPONENTS</span> <span class="o">=</span> <span class="p">(</span><span class="s2">&quot;A&quot;</span><span class="p">,</span> <span class="s2">&quot;B&quot;</span><span class="p">,</span> <span class="s2">&quot;tau&quot;</span><span class="p">,</span> <span class="s2">&quot;total&quot;</span><span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Estimation helpers</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_offline_estimate</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Offline MLE on X of shape (..., T, N, p).&quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;offline&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;gd&quot;</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">kronecker_riemannian_gd_h0</span><span class="p">(</span>
            <span class="n">X</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;gd_iter_max&quot;</span><span class="p">],</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;gd_tol&quot;</span><span class="p">],</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span> <span class="o">=</span> <span class="n">kronecker_mm_h0</span><span class="p">(</span>
        <span class="n">X</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;mm_tol&quot;</span><span class="p">],</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;mm_iter_max&quot;</span><span class="p">],</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span><span class="p">[</span><span class="o">...</span><span class="p">,</span> <span class="kc">None</span><span class="p">]</span> <span class="k">if</span> <span class="n">tau</span><span class="o">.</span><span class="n">ndim</span> <span class="o">==</span> <span class="n">X</span><span class="o">.</span><span class="n">ndim</span> <span class="o">-</span> <span class="mi">2</span> <span class="k">else</span> <span class="n">tau</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_errors</span><span class="p">(</span><span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
    <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">tau_true</span> <span class="o">=</span> <span class="n">truth</span>
    <span class="k">return</span> <span class="n">kronecker_component_errors</span><span class="p">(</span>
        <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_online_checkpoints</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Stream X (..., T_max, N, p) through the recursive estimator.</span>

<span class="sd">    Returns {component: {T: array}} evaluated at each T in T_vec.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">T_set</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span>
    <span class="n">T_max</span> <span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="o">-</span><span class="mi">3</span><span class="p">]</span>
    <span class="n">est</span> <span class="o">=</span> <span class="n">OnlineKroneckerEstimator</span><span class="p">(</span>
        <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span>
        <span class="n">step_rule</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;step_rule&quot;</span><span class="p">],</span> <span class="n">init_mode</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;init_mode&quot;</span><span class="p">],</span> <span class="n">alpha_0</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;alpha_0&quot;</span><span class="p">],</span>
        <span class="n">iter_max</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;mm_iter_max&quot;</span><span class="p">],</span> <span class="n">tol</span><span class="o">=</span><span class="n">cfg</span><span class="p">[</span><span class="s2">&quot;mm_tol&quot;</span><span class="p">],</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">est</span><span class="o">.</span><span class="n">reset</span><span class="p">()</span>
    <span class="n">out</span> <span class="o">=</span> <span class="p">{</span><span class="n">c</span><span class="p">:</span> <span class="p">{}</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">_COMPONENTS</span><span class="p">}</span>
    <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">T_max</span><span class="p">):</span>
        <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span> <span class="o">=</span> <span class="n">est</span><span class="o">.</span><span class="n">update</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="o">...</span><span class="p">,</span> <span class="n">t</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:])</span>
        <span class="n">T_current</span> <span class="o">=</span> <span class="n">t</span> <span class="o">+</span> <span class="mi">1</span>
        <span class="k">if</span> <span class="n">T_current</span> <span class="ow">in</span> <span class="n">T_set</span><span class="p">:</span>
            <span class="n">err</span> <span class="o">=</span> <span class="n">_errors</span><span class="p">(</span><span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">_COMPONENTS</span><span class="p">:</span>
                <span class="n">out</span><span class="p">[</span><span class="n">c</span><span class="p">][</span><span class="n">T_current</span><span class="p">]</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">err</span><span class="p">[</span><span class="n">c</span><span class="p">])</span>
    <span class="k">return</span> <span class="n">out</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Pool worker (numpy only — one trial per worker process)</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_worker</span><span class="p">(</span><span class="n">worker_args</span><span class="p">):</span>
    <span class="n">X</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span> <span class="o">=</span> <span class="n">worker_args</span>
    <span class="n">truth</span> <span class="o">=</span> <span class="p">(</span><span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">)</span>
    <span class="n">online</span> <span class="o">=</span> <span class="n">_online_checkpoints</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>

    <span class="n">offline</span> <span class="o">=</span> <span class="p">{</span><span class="n">c</span><span class="p">:</span> <span class="p">{}</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">_COMPONENTS</span><span class="p">}</span>
    <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">:</span>
        <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span> <span class="o">=</span> <span class="n">_offline_estimate</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="kc">None</span><span class="p">,</span> <span class="p">:</span><span class="n">T</span><span class="p">],</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>
        <span class="n">err</span> <span class="o">=</span> <span class="n">_errors</span><span class="p">(</span><span class="n">A</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">B</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">tau</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">truth</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="s2">&quot;numpy&quot;</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">_COMPONENTS</span><span class="p">:</span>
            <span class="n">offline</span><span class="p">[</span><span class="n">c</span><span class="p">][</span><span class="n">T</span><span class="p">]</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">err</span><span class="p">[</span><span class="n">c</span><span class="p">]))</span>
    <span class="k">return</span> <span class="n">online</span><span class="p">,</span> <span class="n">offline</span>


<span class="k">def</span><span class="w"> </span><span class="nf">_run_pool</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">n_workers</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span>
              <span class="n">storage_path</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="n">n_trials</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">worker_args</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">tau_true</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n_trials</span><span class="p">)</span>
    <span class="p">]</span>
    <span class="n">all_online</span><span class="p">,</span> <span class="n">all_offline</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Starting MSE computation: </span><span class="si">{</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> trials via Pool, </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span><span class="si">}</span><span class="s2"> T-checkpoints...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">(</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">n_trials</span><span class="p">,</span>
                  <span class="n">description</span><span class="o">=</span><span class="s2">&quot;MC trials (Pool)&quot;</span><span class="p">,</span> <span class="n">unit</span><span class="o">=</span><span class="s2">&quot;trials&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="k">with</span> <span class="n">Pool</span><span class="p">(</span><span class="n">processes</span><span class="o">=</span><span class="n">n_workers</span><span class="p">)</span> <span class="k">as</span> <span class="n">pool</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">on</span><span class="p">,</span> <span class="n">off</span> <span class="ow">in</span> <span class="n">pool</span><span class="o">.</span><span class="n">imap_unordered</span><span class="p">(</span><span class="n">_worker</span><span class="p">,</span> <span class="n">worker_args</span><span class="p">):</span>
                <span class="n">all_online</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">on</span><span class="p">)</span>
                <span class="n">all_offline</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">off</span><span class="p">)</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_stack</span><span class="p">(</span><span class="n">dicts</span><span class="p">):</span>
        <span class="k">return</span> <span class="p">{</span><span class="n">c</span><span class="p">:</span> <span class="p">{</span><span class="n">T</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(</span><span class="n">d</span><span class="p">[</span><span class="n">c</span><span class="p">][</span><span class="n">T</span><span class="p">])</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">dicts</span><span class="p">])</span> <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">}</span>
                <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">_COMPONENTS</span><span class="p">}</span>

    <span class="k">return</span> <span class="n">_stack</span><span class="p">(</span><span class="n">all_online</span><span class="p">),</span> <span class="n">_stack</span><span class="p">(</span><span class="n">all_offline</span><span class="p">)</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Batched path (non-numpy backends)</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">_run_batched</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">):</span>
    <span class="n">X</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">A_t</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">A_true</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">B_t</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">B_true</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">tau_t</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">tau_true</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">truth</span> <span class="o">=</span> <span class="p">(</span><span class="n">A_t</span><span class="p">,</span> <span class="n">B_t</span><span class="p">,</span> <span class="n">tau_t</span><span class="p">)</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Starting batched MSE computation on </span><span class="si">{</span><span class="n">backend</span><span class="si">}</span><span class="s2"> (</span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span><span class="si">}</span><span class="s2"> T-points)...&quot;</span><span class="p">)</span>
    <span class="k">with</span> <span class="n">RichProgress</span><span class="p">(</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;[progress.description]</span><span class="si">{task.description}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">BarColumn</span><span class="p">(),</span>
        <span class="n">TextColumn</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{task.completed}</span><span class="s2">/</span><span class="si">{task.total}</span><span class="s2">&quot;</span><span class="p">),</span>
        <span class="n">TimeElapsedColumn</span><span class="p">(),</span>
    <span class="p">)</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task_on</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="s2">&quot;[cyan]Online single-pass...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">online</span> <span class="o">=</span> <span class="n">_online_checkpoints</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
        <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task_on</span><span class="p">)</span>

        <span class="n">task_off</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;[green]Offline (</span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span><span class="si">}</span><span class="s2"> points)...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">))</span>
        <span class="n">offline</span> <span class="o">=</span> <span class="p">{</span><span class="n">c</span><span class="p">:</span> <span class="p">{}</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">_COMPONENTS</span><span class="p">}</span>
        <span class="k">for</span> <span class="n">T</span> <span class="ow">in</span> <span class="n">T_vec</span><span class="p">:</span>
            <span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span> <span class="o">=</span> <span class="n">_offline_estimate</span><span class="p">(</span><span class="n">X</span><span class="p">[</span><span class="o">...</span><span class="p">,</span> <span class="p">:</span><span class="n">T</span><span class="p">,</span> <span class="p">:,</span> <span class="p">:],</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
            <span class="n">err</span> <span class="o">=</span> <span class="n">_errors</span><span class="p">(</span><span class="n">A</span><span class="p">,</span> <span class="n">B</span><span class="p">,</span> <span class="n">tau</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">_COMPONENTS</span><span class="p">:</span>
                <span class="n">offline</span><span class="p">[</span><span class="n">c</span><span class="p">][</span><span class="n">T</span><span class="p">]</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">err</span><span class="p">[</span><span class="n">c</span><span class="p">])</span>
            <span class="n">maybe_empty_cache</span><span class="p">(</span><span class="n">backend</span><span class="p">)</span>
            <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task_off</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">online</span><span class="p">,</span> <span class="n">offline</span>


<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># Entry point</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span><span class="vm">__doc__</span><span class="p">)</span>
    <span class="n">add_mc_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">set_defaults</span><span class="p">(</span><span class="n">T_max</span><span class="o">=</span><span class="mi">1000</span><span class="p">,</span> <span class="n">T_min</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">n_T</span><span class="o">=</span><span class="mi">12</span><span class="p">,</span> <span class="n">n_trials</span><span class="o">=</span><span class="mi">1000</span><span class="p">)</span>

    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--a&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Size of first Kronecker factor (default 3, as in the released config).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--b&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Size of second Kronecker factor (default 4).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--n-samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Samples per date (default: p+1 = a*b+1 = 13).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--nu&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0). &quot;</span>
             <span class="s2">&quot;Large nu approaches the Gaussian case.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-a&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.3+0.7j&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of A, as a complex literal (default 0.3+0.7j).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--rho-b&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.3+0.6j&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Toeplitz coefficient of B, as a complex literal (default 0.3+0.6j).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--offline&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;mm&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;mm&quot;</span><span class="p">,</span> <span class="s2">&quot;gd&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Offline reference: &#39;mm&#39; majorisation-minimisation (fast, default) or &quot;</span>
             <span class="s2">&quot;&#39;gd&#39; Riemannian gradient descent (the reference of the paper). &quot;</span>
             <span class="s2">&quot;Both target the same MLE and agree numerically.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--step-rule&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;fixed&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;fixed&quot;</span><span class="p">,</span> <span class="s2">&quot;armijo&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Step of the recursive estimator: &#39;fixed&#39; is alpha_0/t, the schedule of &quot;</span>
             <span class="s2">&quot;equation (19) (default); &#39;armijo&#39; is a line search at each update.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--init-mode&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;mm&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;mm&quot;</span><span class="p">,</span> <span class="s2">&quot;identity&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Initialisation of the recursive estimator: &#39;mm&#39; warm-starts on the first &quot;</span>
             <span class="s2">&quot;date (default), &#39;identity&#39; starts from (I, I, 1) as the released code does.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--alpha-0&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Initial step (default: 1.0 for &#39;fixed&#39;, 0.1 for &#39;armijo&#39;).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--mm-iter-max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Max MM iterations (default 50).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--mm-tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-8</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;MM convergence tolerance (default 1e-8).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--gd-iter-max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Max iterations of the offline Riemannian gradient descent (default 200).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--gd-tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-8</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Tolerance of the offline Riemannian gradient descent (default 1e-8).&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--debug&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Tiny configuration (8 trials, T up to 50) to validate the pipeline in &quot;</span>
             <span class="s2">&quot;seconds. Results are NOT publication grade.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_T</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T_min</span> <span class="o">=</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">50</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">2</span>
        <span class="n">args</span><span class="o">.</span><span class="n">mm_iter_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">gd_iter_max</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span> <span class="mi">50</span>

    <span class="n">init_logging</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">debug</span><span class="p">:</span>
        <span class="n">logger</span><span class="o">.</span><span class="n">warning</span><span class="p">(</span><span class="s2">&quot;--debug: 8 trials, T_max=50. Pipeline check only, not a result.&quot;</span><span class="p">)</span>

    <span class="n">a</span><span class="p">,</span> <span class="n">b</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">a</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">b</span>
    <span class="n">p</span> <span class="o">=</span> <span class="n">a</span> <span class="o">*</span> <span class="n">b</span>
    <span class="n">n_samples</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span> <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span> <span class="k">else</span> <span class="n">p</span> <span class="o">+</span> <span class="mi">1</span>
    <span class="n">T_vec</span> <span class="o">=</span> <span class="n">T_vec_logspace</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">T_min</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">T_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_T</span><span class="p">)</span>
    <span class="n">T_max</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span>

    <span class="n">cfg</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;offline&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">offline</span><span class="p">,</span>
        <span class="s2">&quot;step_rule&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">step_rule</span><span class="p">,</span>
        <span class="s2">&quot;init_mode&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">init_mode</span><span class="p">,</span>
        <span class="s2">&quot;alpha_0&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">alpha_0</span><span class="p">,</span>
        <span class="s2">&quot;mm_iter_max&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">mm_iter_max</span><span class="p">,</span>
        <span class="s2">&quot;mm_tol&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">mm_tol</span><span class="p">,</span>
        <span class="s2">&quot;gd_iter_max&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">gd_iter_max</span><span class="p">,</span>
        <span class="s2">&quot;gd_tol&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">gd_tol</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Kronecker MSE/ICRB: a=</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">, b=</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">, p=</span><span class="si">{</span><span class="n">p</span><span class="si">}</span><span class="s2">, n_samples=</span><span class="si">{</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">, &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;n_trials=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">, T=[</span><span class="si">{</span><span class="n">T_vec</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s2">..</span><span class="si">{</span><span class="n">T_vec</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">] (</span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">T_vec</span><span class="p">)</span><span class="si">}</span><span class="s2"> pts), &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;backend=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;  texture ~ Gamma(</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="mi">1</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">:</span><span class="s2">.3g</span><span class="si">}</span><span class="s2">) | offline=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">offline</span><span class="si">}</span><span class="s2"> | &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;online step=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">step_rule</span><span class="si">}</span><span class="s2">, init=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">init_mode</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span> <span class="o">=</span> <span class="n">make_ab_toeplitz</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_a</span><span class="p">),</span> <span class="nb">complex</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho_b</span><span class="p">))</span>
    <span class="n">logger</span><span class="o">.</span><span class="n">info</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Generating Kronecker data (</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">T_max</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">p</span><span class="si">}</span><span class="s2">) complex128...&quot;</span><span class="p">)</span>
    <span class="n">data</span><span class="p">,</span> <span class="n">tau_true</span> <span class="o">=</span> <span class="n">generate_kronecker_data</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">T_max</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">tau_shape</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="p">,</span> <span class="n">tau_scale</span><span class="o">=</span><span class="mf">1.0</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="p">,</span> <span class="n">return_tau</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">exporter</span> <span class="o">=</span> <span class="n">MCResultExporter</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span> <span class="n">Path</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span> <span class="sa">f</span><span class="s2">&quot;a</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">_b</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">_T</span><span class="si">{</span><span class="n">T_max</span><span class="si">}</span><span class="s2">_n</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span>
        <span class="n">plot_template</span><span class="o">=</span><span class="n">_MC_PLOT_TEMPLATE_MSE</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="p">(</span><span class="n">online_err</span><span class="p">,</span> <span class="n">offline_err</span><span class="p">),</span> <span class="n">elapsed</span> <span class="o">=</span> <span class="n">timed_run</span><span class="p">(</span>
        <span class="n">args</span><span class="p">,</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_pool</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_workers</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">,</span>
                          <span class="n">args</span><span class="o">.</span><span class="n">export_path</span><span class="p">),</span>
        <span class="k">lambda</span><span class="p">:</span> <span class="n">_run_batched</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">tau_true</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">A_true</span><span class="p">,</span> <span class="n">B_true</span><span class="p">,</span> <span class="n">cfg</span><span class="p">),</span>
    <span class="p">)</span>

    <span class="n">icrb</span> <span class="o">=</span> <span class="n">icrb_kronecker_scaled_gaussian</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">T_vec</span><span class="p">))</span>
    <span class="n">title</span> <span class="o">=</span> <span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Estimation Kronecker gaussienne à échelle  (a=</span><span class="si">{</span><span class="n">a</span><span class="si">}</span><span class="s2">, b=</span><span class="si">{</span><span class="n">b</span><span class="si">}</span><span class="s2">, n=</span><span class="si">{</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">, &quot;</span>
             <span class="sa">f</span><span class="s2">&quot;nu=</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">nu</span><span class="si">}</span><span class="s2">)&quot;</span><span class="p">)</span>
    <span class="n">finish_mse</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">exporter</span><span class="p">,</span> <span class="n">online_err</span><span class="p">,</span> <span class="n">offline_err</span><span class="p">,</span> <span class="n">icrb</span><span class="p">,</span> <span class="n">T_vec</span><span class="p">,</span> <span class="s2">&quot;mc_kron_mse&quot;</span><span class="p">,</span> <span class="n">title</span><span class="p">,</span> <span class="n">elapsed</span><span class="p">)</span>


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
<p class="param-help">Size of first Kronecker factor (default 3, as in the released config).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--b</span><span class="param-type">int</span><span class="param-default">default <b>4</b></span>
</div>
<p class="param-help">Size of second Kronecker factor (default 4).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-samples</span><span class="param-type">int</span>
</div>
<p class="param-help">Samples per date (default: p+1 = a*b+1 = 13).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--nu</span><span class="param-type">float</span><span class="param-default">default <b>1.0</b></span>
</div>
<p class="param-help">Shape of the K-distribution texture, tau ~ Gamma(nu, 1/nu) (default 1.0). Large nu approaches the Gaussian case.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-a</span><span class="param-type">str</span><span class="param-default">default <b>0.3+0.7j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of A, as a complex literal (default 0.3+0.7j).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho-b</span><span class="param-type">str</span><span class="param-default">default <b>0.3+0.6j</b></span>
</div>
<p class="param-help">Toeplitz coefficient of B, as a complex literal (default 0.3+0.6j).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--offline</span><span class="param-type">str</span><span class="param-default">default <b>mm</b></span>
</div>
<p class="param-help">Offline reference: &#x27;mm&#x27; majorisation-minimisation (fast, default) or &#x27;gd&#x27; Riemannian gradient descent (the reference of the paper). Both target the same MLE and agree numerically.</p><p class="param-choices">choices: mm, gd</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--step-rule</span><span class="param-type">str</span><span class="param-default">default <b>fixed</b></span>
</div>
<p class="param-help">Step of the recursive estimator: &#x27;fixed&#x27; is alpha_0/t, the schedule of equation (19) (default); &#x27;armijo&#x27; is a line search at each update.</p><p class="param-choices">choices: fixed, armijo</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--init-mode</span><span class="param-type">str</span><span class="param-default">default <b>mm</b></span>
</div>
<p class="param-help">Initialisation of the recursive estimator: &#x27;mm&#x27; warm-starts on the first date (default), &#x27;identity&#x27; starts from (I, I, 1) as the released code does.</p><p class="param-choices">choices: mm, identity</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--alpha-0</span><span class="param-type">float</span>
</div>
<p class="param-help">Initial step (default: 1.0 for &#x27;fixed&#x27;, 0.1 for &#x27;armijo&#x27;).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--mm-iter-max</span><span class="param-type">int</span><span class="param-default">default <b>50</b></span>
</div>
<p class="param-help">Max MM iterations (default 50).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--mm-tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-08</b></span>
</div>
<p class="param-help">MM convergence tolerance (default 1e-8).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--gd-iter-max</span><span class="param-type">int</span><span class="param-default">default <b>200</b></span>
</div>
<p class="param-help">Max iterations of the offline Riemannian gradient descent (default 200).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--gd-tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-08</b></span>
</div>
<p class="param-help">Tolerance of the offline Riemannian gradient descent (default 1e-8).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--debug</span><span class="param-type">flag</span>
</div>
<p class="param-help">Tiny configuration (8 trials, T up to 50) to validate the pipeline in seconds. Results are NOT publication grade.</p>
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

`2-detection/experiments/sar/sar_mc_kron_mse.yaml`

<a class="back-link" href="../../chapters/2-detection/">← All experiments in 2 · Detection</a>
