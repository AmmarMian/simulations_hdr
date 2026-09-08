<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_example_covariances</span>
</nav>

# context_example_covariances

PGFPlots matrix visualisations of covariance regimes — identity, Toeplitz, random

**Tags:** `context`  `covariance`  `illustration`

## Run

```sh
uv run python 1-context/examples_covariances/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/examples_covariances/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">73 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/examples_covariances/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Illustation of different regime of covariance matrices</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">scipy.linalg</span><span class="w"> </span><span class="kn">import</span> <span class="n">toeplitz</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">matplot2tikz</span><span class="w"> </span><span class="kn">import</span> <span class="n">save</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Illustration of different covariance matrix regimes.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span>
        <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/example_covariances&quot;</span><span class="p">,</span>
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
    <span class="n">_args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>
    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">_args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span><span class="n">_args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>

    <span class="c1"># Constant(s)</span>
    <span class="n">d</span> <span class="o">=</span> <span class="mi">7</span>
    <span class="n">rho</span> <span class="o">=</span> <span class="mf">0.8</span>

    <span class="c1"># Covariance matrices</span>
    <span class="n">cov_id</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">eye</span><span class="p">(</span><span class="n">d</span><span class="p">)</span>
    <span class="n">cov_toeplitz</span> <span class="o">=</span> <span class="n">toeplitz</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">power</span><span class="p">(</span><span class="n">rho</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">d</span><span class="p">)))</span>
    <span class="n">cov_rd</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">empty</span><span class="p">((</span><span class="n">d</span><span class="p">,</span> <span class="n">d</span><span class="p">))</span>
    <span class="n">cov_rd</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">tril_indices</span><span class="p">(</span><span class="n">d</span><span class="p">)]</span> <span class="o">=</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">rng</span><span class="o">.</span><span class="n">random</span><span class="p">(</span><span class="nb">int</span><span class="p">(</span><span class="n">d</span> <span class="o">*</span> <span class="p">(</span><span class="n">d</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span> <span class="o">/</span> <span class="mi">2</span><span class="p">))</span> <span class="o">-</span> <span class="mi">1</span>
    <span class="n">cov_rd</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">triu_indices</span><span class="p">(</span><span class="n">d</span><span class="p">)]</span> <span class="o">=</span> <span class="n">cov_rd</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">tril_indices</span><span class="p">(</span><span class="n">d</span><span class="p">)]</span>
    <span class="n">cov_rd</span><span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">diag_indices</span><span class="p">(</span><span class="n">d</span><span class="p">)]</span> <span class="o">=</span> <span class="mi">1</span>
    <span class="n">cov_rd</span> <span class="o">=</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="p">(</span><span class="n">cov_rd</span> <span class="o">+</span> <span class="n">cov_rd</span><span class="o">.</span><span class="n">T</span><span class="p">)</span>

    <span class="c1"># Single figure with all three matrices</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">4</span><span class="p">))</span>

    <span class="n">matrices</span> <span class="o">=</span> <span class="p">[</span><span class="n">cov_id</span><span class="p">,</span> <span class="n">cov_toeplitz</span><span class="p">,</span> <span class="n">cov_rd</span><span class="p">]</span>
    <span class="n">titles</span> <span class="o">=</span> <span class="p">[</span><span class="sa">r</span><span class="s2">&quot;$\mathbf</span><span class="si">{I}</span><span class="s2">_d$&quot;</span><span class="p">,</span> <span class="sa">r</span><span class="s2">&quot;Toeplitz($\rho$)&quot;</span><span class="p">,</span> <span class="sa">r</span><span class="s2">&quot;Random&quot;</span><span class="p">]</span>

    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">mat</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">matrices</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
        <span class="n">im</span> <span class="o">=</span> <span class="n">ax</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">mat</span><span class="p">,</span> <span class="n">aspect</span><span class="o">=</span><span class="s2">&quot;equal&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;variable $i$&quot;</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;variable $j$&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">title</span><span class="p">)</span>
        <span class="n">fig</span><span class="o">.</span><span class="n">colorbar</span><span class="p">(</span><span class="n">im</span><span class="p">,</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">,</span> <span class="n">orientation</span><span class="o">=</span><span class="s2">&quot;horizontal&quot;</span><span class="p">,</span> <span class="n">pad</span><span class="o">=</span><span class="mf">0.15</span><span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">_args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">_args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;all_covariances.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">_args</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">_args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/example_covariances</b></span>
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

## Results

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--seed 42</code><br>
  <span class="mn-date">9bcab17 · 2026-07-02</span>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_example_covariances.json" data-title="context_example_covariances"></div>
</div>

## Config

`1-context/experiments/context_example_covariances.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
