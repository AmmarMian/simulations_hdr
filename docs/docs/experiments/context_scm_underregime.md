<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_scm_underregime</span>
</nav>

# context_scm_underregime

SCM estimator behavior in the under-regime (d > N) — mean, covariance, condition number errors

**Tags:** `context`  `scm`  `under-regime`  `monte-carlo`

## Run

```sh
uv run python 1-context/scm_underregime/main.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--N 30</code><br>
  <code>--n_trials 10000</code><br>
  <code>--seed 42</code><br>
  <span class="mn-date">5dd225a · 2026-07-02</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/scm_underregime/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">165 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/scm_underregime/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Showing behavior of SCM on simple data with case dim  &gt; N_obs</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">rich.progress</span><span class="w"> </span><span class="kn">import</span> <span class="n">Progress</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">multiprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Pool</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">itertools</span><span class="w"> </span><span class="kn">import</span> <span class="n">product</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">matplot2tikz</span><span class="w"> </span><span class="kn">import</span> <span class="n">save</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>


<span class="k">def</span><span class="w"> </span><span class="nf">compute_estimation</span><span class="p">(</span><span class="n">params</span><span class="p">):</span>
    <span class="n">d</span><span class="p">,</span> <span class="n">N</span><span class="p">,</span> <span class="n">trial_no</span><span class="p">,</span> <span class="n">base_seed</span> <span class="o">=</span> <span class="n">params</span>
    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span>
        <span class="n">d</span> <span class="o">+</span> <span class="n">N</span> <span class="o">+</span> <span class="n">trial_no</span> <span class="o">+</span> <span class="n">base_seed</span>
    <span class="p">)</span>  <span class="c1"># To be sure the generation of data is different</span>
    <span class="n">mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">((</span><span class="n">d</span><span class="p">,</span> <span class="n">N</span><span class="p">))</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">mean</span> <span class="o">+</span> <span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">d</span><span class="p">,</span> <span class="n">N</span><span class="p">))</span>

    <span class="n">estimate_mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
    <span class="n">estimate_cov</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">cov</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">[</span>
        <span class="n">d</span><span class="p">,</span>
        <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">estimate_mean</span> <span class="o">-</span> <span class="n">mean</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]),</span>
        <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">estimate_cov</span> <span class="o">-</span> <span class="n">np</span><span class="o">.</span><span class="n">eye</span><span class="p">(</span><span class="n">d</span><span class="p">),</span> <span class="nb">ord</span><span class="o">=</span><span class="s2">&quot;fro&quot;</span><span class="p">),</span>
        <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">cond</span><span class="p">(</span><span class="n">estimate_cov</span><span class="p">),</span>
    <span class="p">]</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;MC simulation of estimating mean and covariance with increasing number of variables.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--N&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">30</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of observations.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--n_trials&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of MC-trials.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span>
        <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/error_estimation_scm_underregime&quot;</span><span class="p">,</span>
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
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Save TikZ/PGFPlots figures (.tex) (default: True).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;random seed generation base seed&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>
    <span class="n">args</span><span class="o">.</span><span class="n">output_dir</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span>  <span class="c1"># alias for references below</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="c1"># Create output directory if not existing</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">exists</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">output_dir</span><span class="p">):</span>
        <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">output_dir</span><span class="p">)</span>

    <span class="c1"># Constants</span>
    <span class="n">N</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">N</span>
    <span class="n">d_vec</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">unique</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">logspace</span><span class="p">(</span><span class="mf">0.5</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">N</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="nb">int</span><span class="p">))</span>
    <span class="n">n_trials</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span>

    <span class="c1"># Montecarlo with progressbar.</span>
    <span class="c1"># Inspired from: https://github.com/Textualize/rich/discussions/884#discussioncomment-269200</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Launching simulation&quot;</span><span class="p">)</span>
    <span class="n">param_grid</span> <span class="o">=</span> <span class="n">product</span><span class="p">(</span><span class="n">d_vec</span><span class="p">,</span> <span class="p">[</span><span class="n">N</span><span class="p">],</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">n_trials</span><span class="p">)),</span> <span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">])</span>
    <span class="n">results</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">with</span> <span class="n">Progress</span><span class="p">()</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task_id</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span><span class="s2">&quot;[cyan]Working...&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">d_vec</span><span class="p">)</span> <span class="o">*</span> <span class="n">n_trials</span><span class="p">)</span>
        <span class="k">with</span> <span class="n">Pool</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">cpu_count</span><span class="p">())</span> <span class="k">as</span> <span class="n">pool</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">result</span> <span class="ow">in</span> <span class="n">pool</span><span class="o">.</span><span class="n">imap</span><span class="p">(</span><span class="n">compute_estimation</span><span class="p">,</span> <span class="n">param_grid</span><span class="p">):</span>
                <span class="n">results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">result</span><span class="p">)</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task_id</span><span class="p">)</span>

        <span class="n">results</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">results</span><span class="p">)</span>

    <span class="c1"># Averaging monte-carlo</span>
    <span class="n">error_mean_mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">results</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">==</span> <span class="n">d</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">d_vec</span><span class="p">])</span>
    <span class="n">error_mean_std</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">results</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">==</span> <span class="n">d</span><span class="p">,</span> <span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">std</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">d_vec</span><span class="p">])</span>

    <span class="n">error_cov_mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">results</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">==</span> <span class="n">d</span><span class="p">,</span> <span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">d_vec</span><span class="p">])</span>
    <span class="n">error_cov_std</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">results</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">==</span> <span class="n">d</span><span class="p">,</span> <span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">std</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">d_vec</span><span class="p">])</span>

    <span class="n">cond_cov_mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">results</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">==</span> <span class="n">d</span><span class="p">,</span> <span class="mi">3</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">d_vec</span><span class="p">])</span>
    <span class="n">cond_cov_std</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">results</span><span class="p">[</span><span class="n">results</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">==</span> <span class="n">d</span><span class="p">,</span> <span class="mi">3</span><span class="p">]</span><span class="o">.</span><span class="n">std</span><span class="p">()</span> <span class="k">for</span> <span class="n">d</span> <span class="ow">in</span> <span class="n">d_vec</span><span class="p">])</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Done.&quot;</span><span class="p">)</span>

    <span class="c1"># Save results</span>
    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">output_dir</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">d_vec</span><span class="o">=</span><span class="n">d_vec</span><span class="p">,</span> <span class="n">N</span><span class="o">=</span><span class="n">N</span><span class="p">,</span> <span class="n">n_trials</span><span class="o">=</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span>
        <span class="n">error_mean_mean</span><span class="o">=</span><span class="n">error_mean_mean</span><span class="p">,</span> <span class="n">error_mean_std</span><span class="o">=</span><span class="n">error_mean_std</span><span class="p">,</span>
        <span class="n">error_cov_mean</span><span class="o">=</span><span class="n">error_cov_mean</span><span class="p">,</span> <span class="n">error_cov_std</span><span class="o">=</span><span class="n">error_cov_std</span><span class="p">,</span>
        <span class="n">cond_cov_mean</span><span class="o">=</span><span class="n">cond_cov_mean</span><span class="p">,</span> <span class="n">cond_cov_std</span><span class="o">=</span><span class="n">cond_cov_std</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="c1"># Plotting</span>
    <span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">d_vec</span><span class="p">,</span> <span class="n">error_mean_mean</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">edgecolors</span><span class="o">=</span><span class="s2">&quot;k&quot;</span><span class="p">)</span>
    <span class="n">errline</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">errorbar</span><span class="p">(</span><span class="n">d_vec</span><span class="p">,</span> <span class="n">error_mean_mean</span><span class="p">,</span> <span class="n">yerr</span><span class="o">=</span><span class="n">error_mean_std</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;&quot;</span><span class="p">,</span> <span class="n">capsize</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
    <span class="n">errline</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$d$&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span>
        <span class="sa">r</span><span class="s2">&quot;$\|\hat{\boldsymbol{\mu}}_\mathcal</span><span class="si">{X}</span><span class="s2"> - \boldsymbol{\mu}_\mathcal</span><span class="si">{X}</span><span class="s2">\|_2$&quot;</span>
    <span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Error of mean estimation with </span><span class="si">{</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> Monte-carlo trials&quot;</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">output_dir</span><span class="p">,</span> <span class="s2">&quot;mean.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved mean error in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">d_vec</span><span class="p">,</span> <span class="n">error_cov_mean</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">edgecolors</span><span class="o">=</span><span class="s2">&quot;k&quot;</span><span class="p">)</span>
    <span class="n">errline</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">errorbar</span><span class="p">(</span><span class="n">d_vec</span><span class="p">,</span> <span class="n">error_cov_mean</span><span class="p">,</span> <span class="n">yerr</span><span class="o">=</span><span class="n">error_cov_std</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;&quot;</span><span class="p">,</span> <span class="n">capsize</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
    <span class="n">errline</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;$d$&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span>
        <span class="sa">r</span><span class="s2">&quot;$\|\hat{\boldsymbol{\Sigma}}_\mathcal</span><span class="si">{X}</span><span class="s2"> - \boldsymbol{\Sigma}_\mathcal</span><span class="si">{X}</span><span class="s2">\|_2$&quot;</span>
    <span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Error of covariance estimation with </span><span class="si">{</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> Monte-carlo trials&quot;</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">output_dir</span><span class="p">,</span> <span class="s2">&quot;cov.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved cov error in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">d_vec</span><span class="p">,</span> <span class="n">cond_cov_mean</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">edgecolors</span><span class="o">=</span><span class="s2">&quot;k&quot;</span><span class="p">)</span>
    <span class="c1"># plt.errorbar(d_vec, cond_cov_mean, yerr=cond_cov_std, linestyle=&quot;&quot;, capsize=5)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">N</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;k&quot;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="sa">r</span><span class="s2">&quot;$d=N$&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;$d$&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span>
        <span class="sa">r</span><span class="s2">&quot;$\operatorname</span><span class="si">{Cond}</span><span class="s2">\left(\hat{\boldsymbol{\Sigma}}_\mathcal</span><span class="si">{X}</span><span class="s2">\right)$&quot;</span>
    <span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span>
        <span class="sa">f</span><span class="s2">&quot;Condition number of estimated covariance with </span><span class="si">{</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> Monte-carlo trials&quot;</span>
    <span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">output_dir</span><span class="p">,</span> <span class="s2">&quot;cond.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved cov error in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--N</span><span class="param-type">int</span><span class="param-default">default <b>30</b></span>
</div>
<p class="param-help">Number of observations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_trials</span><span class="param-type">int</span><span class="param-default">default <b>50</b></span>
</div>
<p class="param-help">Number of MC-trials.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/error_estimation_scm_underregime</b></span>
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
<p class="param-help">Save TikZ/PGFPlots figures (.tex) (default: True).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">random seed generation base seed</p>
</div>
</div>

## Results

<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_scm_underregime.json" data-title="context_scm_underregime"></div>
</div>

## Config

`1-context/experiments/context_scm_underregime.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
