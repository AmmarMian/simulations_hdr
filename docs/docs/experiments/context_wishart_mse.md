<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_wishart_mse</span>
</nav>

# context_wishart_mse

Monte-Carlo check of the closed-form MSE of the SCM under a Gaussian model, swept in the sample support and in the dimension

**Tags:** `context`  `scm`  `wishart`  `monte-carlo`

## Run

```sh
uv run python 1-context/wishart_mse/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/wishart_mse/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">252 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/wishart_mse/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Monte-Carlo check of the closed-form MSE of the SCM under a Gaussian model.</span>
<span class="c1">#</span>
<span class="c1"># Under x_k ~ N(mu, Sigma) i.i.d., the unbiased SCM S/(N-1) is Wishart</span>
<span class="c1"># distributed, which gives the exact mean squared error</span>
<span class="c1">#</span>
<span class="c1">#     E ||Sigma_hat - Sigma||_F^2 = (||Sigma||_F^2 + tr(Sigma)^2) / (N - 1).</span>
<span class="c1">#</span>
<span class="c1"># Two sweeps confirm both regimes of that expression: the 1/N decay at fixed</span>
<span class="c1"># dimension, and the d^2 growth at fixed sample support.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">scipy.linalg</span><span class="w"> </span><span class="kn">import</span> <span class="n">toeplitz</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">rich.progress</span><span class="w"> </span><span class="kn">import</span> <span class="n">Progress</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">matplot2tikz</span><span class="w"> </span><span class="kn">import</span> <span class="n">save</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="n">SCMEstimator</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.simulation</span><span class="w"> </span><span class="kn">import</span> <span class="n">T_vec_logspace</span>


<span class="k">def</span><span class="w"> </span><span class="nf">make_covariance</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">kind</span><span class="p">,</span> <span class="n">rho</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Real SPD covariance matrix, matching the regimes of the context chapter.&quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="n">kind</span> <span class="o">==</span> <span class="s2">&quot;identity&quot;</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">eye</span><span class="p">(</span><span class="n">d</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">kind</span> <span class="o">==</span> <span class="s2">&quot;toeplitz&quot;</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">toeplitz</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">power</span><span class="p">(</span><span class="n">rho</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="n">d</span><span class="p">)))</span>
    <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Unknown covariance kind: </span><span class="si">{</span><span class="n">kind</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">theoretical_mse</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Exact MSE of the unbiased SCM, from the Wishart moments.&quot;&quot;&quot;</span>
    <span class="k">return</span> <span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="nb">ord</span><span class="o">=</span><span class="s2">&quot;fro&quot;</span><span class="p">)</span> <span class="o">**</span> <span class="mi">2</span> <span class="o">+</span> <span class="n">np</span><span class="o">.</span><span class="n">trace</span><span class="p">(</span><span class="n">cov</span><span class="p">)</span> <span class="o">**</span> <span class="mi">2</span>
    <span class="p">)</span> <span class="o">/</span> <span class="p">(</span><span class="n">n_samples</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">empirical_mse</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">n_trials</span><span class="p">,</span> <span class="n">rng</span><span class="p">,</span> <span class="n">max_elements</span><span class="o">=</span><span class="mi">4_000_000</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Monte-Carlo mean and standard error of the SCM squared error.</span>

<span class="sd">    Trials are processed in chunks so memory stays bounded when both</span>
<span class="sd">    n_samples and n_trials are large.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">d</span> <span class="o">=</span> <span class="n">cov</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">cholesky</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">cholesky</span><span class="p">(</span><span class="n">cov</span><span class="p">)</span>
    <span class="c1"># Unbiased SCM with estimated mean: SCMEstimator divides by n_samples, so</span>
    <span class="c1"># the n_samples / (n_samples - 1) factor restores the (N-1) normalisation.</span>
    <span class="n">estimator</span> <span class="o">=</span> <span class="n">SCMEstimator</span><span class="p">(</span><span class="n">assume_centered</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">correction</span> <span class="o">=</span> <span class="n">n_samples</span> <span class="o">/</span> <span class="p">(</span><span class="n">n_samples</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span>

    <span class="n">chunk</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="nb">min</span><span class="p">(</span><span class="n">n_trials</span><span class="p">,</span> <span class="nb">int</span><span class="p">(</span><span class="n">max_elements</span> <span class="o">//</span> <span class="p">(</span><span class="n">n_samples</span> <span class="o">*</span> <span class="n">d</span><span class="p">))))</span>
    <span class="n">errors</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">start</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">n_trials</span><span class="p">,</span> <span class="n">chunk</span><span class="p">):</span>
        <span class="n">size</span> <span class="o">=</span> <span class="nb">min</span><span class="p">(</span><span class="n">chunk</span><span class="p">,</span> <span class="n">n_trials</span> <span class="o">-</span> <span class="n">start</span><span class="p">)</span>
        <span class="n">data</span> <span class="o">=</span> <span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">size</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">d</span><span class="p">))</span> <span class="o">@</span> <span class="n">cholesky</span><span class="o">.</span><span class="n">T</span>
        <span class="n">estimates</span> <span class="o">=</span> <span class="n">correction</span> <span class="o">*</span> <span class="n">estimator</span><span class="o">.</span><span class="n">compute</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
        <span class="n">errors</span><span class="o">.</span><span class="n">append</span><span class="p">(</span>
            <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">estimates</span> <span class="o">-</span> <span class="n">cov</span><span class="p">,</span> <span class="nb">ord</span><span class="o">=</span><span class="s2">&quot;fro&quot;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="p">(</span><span class="o">-</span><span class="mi">2</span><span class="p">,</span> <span class="o">-</span><span class="mi">1</span><span class="p">))</span> <span class="o">**</span> <span class="mi">2</span>
        <span class="p">)</span>
    <span class="n">errors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">(</span><span class="n">errors</span><span class="p">)</span>
    <span class="c1"># Standard error of the mean: we are checking the expectation itself, not</span>
    <span class="c1"># the per-trial spread, which for a chi-squared-like quantity is of the</span>
    <span class="c1"># same order as the mean and would go negative on a log axis.</span>
    <span class="k">return</span> <span class="n">errors</span><span class="o">.</span><span class="n">mean</span><span class="p">(),</span> <span class="n">errors</span><span class="o">.</span><span class="n">std</span><span class="p">()</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">errors</span><span class="o">.</span><span class="n">size</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Monte-Carlo verification of the closed-form MSE of the SCM.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_trials&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of MC-trials per point.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--d_sweep_values&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">7</span><span class="p">,</span> <span class="mi">20</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Dimensions shown in the N-sweep panel.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_sweep_values&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">500</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Sample supports shown in the d-sweep panel.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_min&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">30</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Smallest N of the N-sweep.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Largest N of the N-sweep.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_points&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">12</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of points per sweep.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--d_max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">40</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Largest dimension of the d-sweep.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--covariance&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;toeplitz&quot;</span><span class="p">,</span> <span class="n">choices</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;toeplitz&quot;</span><span class="p">,</span> <span class="s2">&quot;identity&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;True covariance regime. Toeplitz exercises the full formula, &quot;</span>
             <span class="s2">&quot;identity reduces it to d(d+1)/(N-1).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--rho&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Correlation of the Toeplitz regime.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span>
        <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/wishart_mse&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Output directory for LaTeX exports (injected by qanat, or set manually).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--show-interactive&quot;</span><span class="p">,</span>
        <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Show plots interactively with matplotlib.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--export&quot;</span><span class="p">,</span>
        <span class="n">action</span><span class="o">=</span><span class="n">argparse</span><span class="o">.</span><span class="n">BooleanOptionalAction</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Save TikZ/PGFPlots figure (.tex) (default: True).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;random seed generation base seed&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>

    <span class="n">N_vec</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">T_vec_logspace</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_min</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_points</span><span class="p">))</span>
    <span class="n">d_vec</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">T_vec_logspace</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">d_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_points</span><span class="p">))</span>

    <span class="n">n_points_total</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span><span class="p">)</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">N_vec</span><span class="p">)</span> <span class="o">+</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span><span class="p">)</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">d_vec</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Launching simulation&quot;</span><span class="p">)</span>

    <span class="c1"># Sweep 1: error against N, at fixed dimension</span>
    <span class="n">mse_vs_N</span><span class="p">,</span> <span class="n">std_vs_N</span><span class="p">,</span> <span class="n">theory_vs_N</span> <span class="o">=</span> <span class="p">{},</span> <span class="p">{},</span> <span class="p">{}</span>
    <span class="c1"># Sweep 2: error against d, at fixed sample support</span>
    <span class="n">mse_vs_d</span><span class="p">,</span> <span class="n">std_vs_d</span><span class="p">,</span> <span class="n">theory_vs_d</span> <span class="o">=</span> <span class="p">{},</span> <span class="p">{},</span> <span class="p">{}</span>

    <span class="k">with</span> <span class="n">Progress</span><span class="p">()</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task_id</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="s2">&quot;[cyan]Working...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="n">n_points_total</span><span class="p">)</span>

        <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span><span class="p">:</span>
            <span class="n">cov</span> <span class="o">=</span> <span class="n">make_covariance</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">covariance</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">)</span>
            <span class="n">means</span><span class="p">,</span> <span class="n">stds</span><span class="p">,</span> <span class="n">theory</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[],</span> <span class="p">[]</span>
            <span class="k">for</span> <span class="n">n_samples</span> <span class="ow">in</span> <span class="n">N_vec</span><span class="p">:</span>
                <span class="n">mean</span><span class="p">,</span> <span class="n">std</span> <span class="o">=</span> <span class="n">empirical_mse</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="nb">int</span><span class="p">(</span><span class="n">n_samples</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">rng</span><span class="p">)</span>
                <span class="n">means</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">mean</span><span class="p">)</span>
                <span class="n">stds</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">std</span><span class="p">)</span>
                <span class="n">theory</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">theoretical_mse</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="nb">int</span><span class="p">(</span><span class="n">n_samples</span><span class="p">)))</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task_id</span><span class="p">)</span>
            <span class="n">mse_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">],</span> <span class="n">std_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">],</span> <span class="n">theory_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
                <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">means</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">stds</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">theory</span><span class="p">)</span>
            <span class="p">)</span>

        <span class="k">for</span> <span class="n">n_samples</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span><span class="p">:</span>
            <span class="n">means</span><span class="p">,</span> <span class="n">stds</span><span class="p">,</span> <span class="n">theory</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[],</span> <span class="p">[]</span>
            <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">d_vec</span><span class="p">:</span>
                <span class="n">cov</span> <span class="o">=</span> <span class="n">make_covariance</span><span class="p">(</span><span class="nb">int</span><span class="p">(</span><span class="n">d</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">covariance</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">)</span>
                <span class="n">mean</span><span class="p">,</span> <span class="n">std</span> <span class="o">=</span> <span class="n">empirical_mse</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">rng</span><span class="p">)</span>
                <span class="n">means</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">mean</span><span class="p">)</span>
                <span class="n">stds</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">std</span><span class="p">)</span>
                <span class="n">theory</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">theoretical_mse</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">))</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task_id</span><span class="p">)</span>
            <span class="n">mse_vs_d</span><span class="p">[</span><span class="n">n_samples</span><span class="p">],</span> <span class="n">std_vs_d</span><span class="p">[</span><span class="n">n_samples</span><span class="p">],</span> <span class="n">theory_vs_d</span><span class="p">[</span><span class="n">n_samples</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span>
                <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">means</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">stds</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">theory</span><span class="p">)</span>
            <span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Done.&quot;</span><span class="p">)</span>

    <span class="c1"># Largest relative deviation, as a scalar sanity check</span>
    <span class="n">deviations</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">mse_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">]</span> <span class="o">/</span> <span class="n">theory_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">]</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">max</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span>
    <span class="p">]</span> <span class="o">+</span> <span class="p">[</span>
        <span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">mse_vs_d</span><span class="p">[</span><span class="n">n</span><span class="p">]</span> <span class="o">/</span> <span class="n">theory_vs_d</span><span class="p">[</span><span class="n">n</span><span class="p">]</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">max</span><span class="p">()</span> <span class="k">for</span> <span class="n">n</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span>
    <span class="p">]</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Largest relative deviation from theory: </span><span class="si">{</span><span class="nb">max</span><span class="p">(</span><span class="n">deviations</span><span class="p">)</span><span class="si">:</span><span class="s2">.2%</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="c1"># Save results</span>
    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span>
        <span class="n">n_trials</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span>
        <span class="n">covariance</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">covariance</span><span class="p">,</span>
        <span class="n">rho</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span>
        <span class="n">N_vec</span><span class="o">=</span><span class="n">N_vec</span><span class="p">,</span>
        <span class="n">d_vec</span><span class="o">=</span><span class="n">d_vec</span><span class="p">,</span>
        <span class="n">d_sweep_values</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span><span class="p">),</span>
        <span class="n">n_sweep_values</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span><span class="p">),</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;mse_vs_N_d</span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">mse_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">]</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span><span class="p">},</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;std_vs_N_d</span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">std_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">]</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span><span class="p">},</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;theory_vs_N_d</span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">theory_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">]</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span><span class="p">},</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;mse_vs_d_N</span><span class="si">{</span><span class="n">n</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">mse_vs_d</span><span class="p">[</span><span class="n">n</span><span class="p">]</span> <span class="k">for</span> <span class="n">n</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span><span class="p">},</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;std_vs_d_N</span><span class="si">{</span><span class="n">n</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">std_vs_d</span><span class="p">[</span><span class="n">n</span><span class="p">]</span> <span class="k">for</span> <span class="n">n</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span><span class="p">},</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;theory_vs_d_N</span><span class="si">{</span><span class="n">n</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">theory_vs_d</span><span class="p">[</span><span class="n">n</span><span class="p">]</span> <span class="k">for</span> <span class="n">n</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span><span class="p">},</span>
    <span class="p">)</span>

    <span class="c1"># Plotting: ratio of the Monte-Carlo estimate to the closed-form value.</span>
    <span class="c1"># Plotting both in absolute value on a log axis spanning several decades</span>
    <span class="c1"># would hide any disagreement — the ratio is what actually tests the</span>
    <span class="c1"># formula, with the error bars giving the scale of what counts as a</span>
    <span class="c1"># deviation.</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">11</span><span class="p">,</span> <span class="mi">4</span><span class="p">),</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">plot_ratio</span><span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">mse</span><span class="p">,</span> <span class="n">standard_error</span><span class="p">,</span> <span class="n">theory</span><span class="p">,</span> <span class="n">color</span><span class="p">,</span> <span class="n">label</span><span class="p">):</span>
        <span class="n">ratio</span> <span class="o">=</span> <span class="n">mse</span> <span class="o">/</span> <span class="n">theory</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
            <span class="n">x</span><span class="p">,</span> <span class="n">ratio</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">s</span><span class="o">=</span><span class="mi">22</span><span class="p">,</span>
            <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">edgecolors</span><span class="o">=</span><span class="n">color</span><span class="p">,</span> <span class="n">linewidths</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="n">label</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">errline</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">ax</span><span class="o">.</span><span class="n">errorbar</span><span class="p">(</span>
            <span class="n">x</span><span class="p">,</span> <span class="n">ratio</span><span class="p">,</span> <span class="n">yerr</span><span class="o">=</span><span class="mi">2</span> <span class="o">*</span> <span class="n">standard_error</span> <span class="o">/</span> <span class="n">theory</span><span class="p">,</span>
            <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;&quot;</span><span class="p">,</span> <span class="n">capsize</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">ecolor</span><span class="o">=</span><span class="n">color</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="c1"># matplot2tikz exports an empty linestyle as a solid connecting line</span>
        <span class="n">errline</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">d</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">d_sweep_values</span><span class="p">):</span>
        <span class="n">plot_ratio</span><span class="p">(</span>
            <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">N_vec</span><span class="p">,</span> <span class="n">mse_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">],</span> <span class="n">std_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">],</span> <span class="n">theory_vs_N</span><span class="p">[</span><span class="n">d</span><span class="p">],</span>
            <span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="sa">rf</span><span class="s2">&quot;$d = </span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">axhline</span><span class="p">(</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C7&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$N$&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\mathrm</span><span class="si">{EQM}</span><span class="s2">_{\mathrm</span><span class="si">{MC}</span><span class="s2">} / \mathrm</span><span class="si">{EQM}</span><span class="s2">_{\mathrm</span><span class="si">{th}</span><span class="s2">}$&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>

    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">n_samples</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_sweep_values</span><span class="p">):</span>
        <span class="n">plot_ratio</span><span class="p">(</span>
            <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">d_vec</span><span class="p">,</span> <span class="n">mse_vs_d</span><span class="p">[</span><span class="n">n_samples</span><span class="p">],</span> <span class="n">std_vs_d</span><span class="p">[</span><span class="n">n_samples</span><span class="p">],</span>
            <span class="n">theory_vs_d</span><span class="p">[</span><span class="n">n_samples</span><span class="p">],</span> <span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="sa">rf</span><span class="s2">&quot;$N = </span><span class="si">{</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">axhline</span><span class="p">(</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C7&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$d$&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;wishart_mse.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved MSE verification in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_trials</span><span class="param-type">int</span><span class="param-default">default <b>10000</b></span>
</div>
<p class="param-help">Number of MC-trials per point.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--d_sweep_values</span><span class="param-type">int</span><span class="param-default">default <b>[7, 20]</b></span>
</div>
<p class="param-help">Dimensions shown in the N-sweep panel.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_sweep_values</span><span class="param-type">int</span><span class="param-default">default <b>[100, 500]</b></span>
</div>
<p class="param-help">Sample supports shown in the d-sweep panel.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_min</span><span class="param-type">int</span><span class="param-default">default <b>30</b></span>
</div>
<p class="param-help">Smallest N of the N-sweep.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_max</span><span class="param-type">int</span><span class="param-default">default <b>10000</b></span>
</div>
<p class="param-help">Largest N of the N-sweep.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_points</span><span class="param-type">int</span><span class="param-default">default <b>12</b></span>
</div>
<p class="param-help">Number of points per sweep.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--d_max</span><span class="param-type">int</span><span class="param-default">default <b>40</b></span>
</div>
<p class="param-help">Largest dimension of the d-sweep.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--covariance</span><span class="param-type">str</span><span class="param-default">default <b>toeplitz</b></span>
</div>
<p class="param-help">True covariance regime. Toeplitz exercises the full formula, identity reduces it to d(d+1)/(N-1).</p><p class="param-choices">choices: toeplitz, identity</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho</span><span class="param-type">float</span><span class="param-default">default <b>0.8</b></span>
</div>
<p class="param-help">Correlation of the Toeplitz regime.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/wishart_mse</b></span>
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
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">random seed generation base seed</p>
</div>
</div>

## Config

`1-context/experiments/context_wishart_mse.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
