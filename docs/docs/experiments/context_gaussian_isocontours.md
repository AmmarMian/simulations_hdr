<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_gaussian_isocontours</span>
</nav>

# context_gaussian_isocontours

Isodensity contours and samples of the bivariate Gaussian for three covariance regimes

**Tags:** `context`  `gaussian`  `distributions`  `illustration`

## Run

```sh
uv run python 1-context/probability_densities/main.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--n_samples 50</code><br>
  <code>--rho 0.8</code><br>
  <code>--condition 50.0</code><br>
  <code>--seed 42</code><br>
  <span class="mn-date">5d050cb · 2026-08-17</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/probability_densities/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">184 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/probability_densities/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Isodensity contours of the bivariate Gaussian for three covariance regimes</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">scipy.stats</span><span class="w"> </span><span class="kn">import</span> <span class="n">chi2</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">matplot2tikz</span><span class="w"> </span><span class="kn">import</span> <span class="n">save</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>


<span class="k">def</span><span class="w"> </span><span class="nf">gaussian_pdf</span><span class="p">(</span><span class="n">grid_x</span><span class="p">,</span> <span class="n">grid_y</span><span class="p">,</span> <span class="n">mean</span><span class="p">,</span> <span class="n">cov</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Evaluate the bivariate Gaussian d.d.p on a meshgrid.&quot;&quot;&quot;</span>
    <span class="n">points</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">grid_x</span><span class="o">.</span><span class="n">ravel</span><span class="p">(),</span> <span class="n">grid_y</span><span class="o">.</span><span class="n">ravel</span><span class="p">()],</span> <span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span> <span class="o">-</span> <span class="n">mean</span><span class="p">[:,</span> <span class="kc">None</span><span class="p">]</span>
    <span class="n">quad</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">einsum</span><span class="p">(</span><span class="s2">&quot;ik,ij,jk-&gt;k&quot;</span><span class="p">,</span> <span class="n">points</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">inv</span><span class="p">(</span><span class="n">cov</span><span class="p">),</span> <span class="n">points</span><span class="p">)</span>
    <span class="n">norm</span> <span class="o">=</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="p">(</span><span class="mi">2</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">cov</span><span class="p">)))</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">norm</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">quad</span><span class="p">))</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="n">grid_x</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">mahalanobis_levels</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="n">probabilities</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;d.d.p values whose contours enclose the given probability masses.</span>

<span class="sd">    For a bivariate Gaussian the squared Mahalanobis distance follows a</span>
<span class="sd">    chi-squared law with 2 degrees of freedom, which gives the radii</span>
<span class="sd">    directly.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">norm</span> <span class="o">=</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="p">(</span><span class="mi">2</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">cov</span><span class="p">)))</span>
    <span class="n">radii_squared</span> <span class="o">=</span> <span class="n">chi2</span><span class="o">.</span><span class="n">ppf</span><span class="p">(</span><span class="n">probabilities</span><span class="p">,</span> <span class="n">df</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">norm</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">radii_squared</span><span class="p">))</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Isodensity contours of the bivariate Gaussian for three covariance regimes.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of samples drawn per regime. Drawn as hollow markers so the &quot;</span>
             <span class="s2">&quot;isodensity contours stay readable underneath.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--rho&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Correlation coefficient of the correlated regime.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--condition&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">50.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Condition number of the ill-conditioned regime.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span>
        <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/gaussian_isocontours&quot;</span><span class="p">,</span>
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

    <span class="c1"># Constant(s)</span>
    <span class="n">d</span> <span class="o">=</span> <span class="mi">2</span>
    <span class="n">mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">d</span><span class="p">)</span>
    <span class="n">probabilities</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.9</span><span class="p">,</span> <span class="mf">0.99</span><span class="p">])</span>

    <span class="c1"># Covariance matrices: identity, correlated, ill-conditioned</span>
    <span class="n">cov_id</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">eye</span><span class="p">(</span><span class="n">d</span><span class="p">)</span>

    <span class="n">cov_corr</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">],</span> <span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">]])</span>

    <span class="n">angle</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span> <span class="o">/</span> <span class="mi">6</span>
    <span class="n">rotation</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span>
        <span class="p">[[</span><span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">angle</span><span class="p">),</span> <span class="o">-</span><span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">angle</span><span class="p">)],</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">angle</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">angle</span><span class="p">)]]</span>
    <span class="p">)</span>
    <span class="n">eigenvalues</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">1.0</span><span class="p">,</span> <span class="mf">1.0</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">])</span>
    <span class="n">cov_ill</span> <span class="o">=</span> <span class="n">rotation</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">eigenvalues</span><span class="p">)</span> <span class="o">@</span> <span class="n">rotation</span><span class="o">.</span><span class="n">T</span>

    <span class="n">covariances</span> <span class="o">=</span> <span class="p">[</span><span class="n">cov_id</span><span class="p">,</span> <span class="n">cov_corr</span><span class="p">,</span> <span class="n">cov_ill</span><span class="p">]</span>
    <span class="n">titles</span> <span class="o">=</span> <span class="p">[</span>
        <span class="sa">r</span><span class="s2">&quot;$\mathbf{\Sigma} = \mathbf</span><span class="si">{I}</span><span class="s2">_2$&quot;</span><span class="p">,</span>
        <span class="sa">rf</span><span class="s2">&quot;$\rho = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">,</span>
        <span class="sa">rf</span><span class="s2">&quot;$k(\mathbf</span><span class="se">{{</span><span class="s2">\Sigma</span><span class="se">}}</span><span class="s2">) = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="si">:</span><span class="s2">.0f</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">,</span>
    <span class="p">]</span>

    <span class="c1"># Sampling</span>
    <span class="n">samples</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">rng</span><span class="o">.</span><span class="n">multivariate_normal</span><span class="p">(</span><span class="n">mean</span><span class="p">,</span> <span class="n">cov</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">)</span> <span class="k">for</span> <span class="n">cov</span> <span class="ow">in</span> <span class="n">covariances</span>
    <span class="p">]</span>

    <span class="c1"># Common extent so the three panels are visually comparable. The outer</span>
    <span class="c1"># contour sets a floor so the framing stays stable across --seed and</span>
    <span class="c1"># --n_samples, and the draws widen it only when they fall outside.</span>
    <span class="n">outer_radius</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">chi2</span><span class="o">.</span><span class="n">ppf</span><span class="p">(</span><span class="n">probabilities</span><span class="o">.</span><span class="n">max</span><span class="p">(),</span> <span class="n">df</span><span class="o">=</span><span class="mi">2</span><span class="p">))</span>
    <span class="n">contour_extent</span> <span class="o">=</span> <span class="n">outer_radius</span> <span class="o">*</span> <span class="nb">max</span><span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigvalsh</span><span class="p">(</span><span class="n">cov</span><span class="p">))</span><span class="o">.</span><span class="n">max</span><span class="p">()</span> <span class="k">for</span> <span class="n">cov</span> <span class="ow">in</span> <span class="n">covariances</span>
    <span class="p">)</span>
    <span class="n">limit</span> <span class="o">=</span> <span class="mf">1.1</span> <span class="o">*</span> <span class="nb">max</span><span class="p">(</span><span class="n">contour_extent</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">(</span><span class="n">samples</span><span class="p">))</span><span class="o">.</span><span class="n">max</span><span class="p">())</span>
    <span class="n">axis_grid</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">,</span> <span class="mi">400</span><span class="p">)</span>
    <span class="n">grid_x</span><span class="p">,</span> <span class="n">grid_y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">meshgrid</span><span class="p">(</span><span class="n">axis_grid</span><span class="p">,</span> <span class="n">axis_grid</span><span class="p">)</span>

    <span class="c1"># Single figure with all three regimes</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">4</span><span class="p">),</span> <span class="n">sharex</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">cov</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">covariances</span><span class="p">,</span> <span class="n">samples</span><span class="p">,</span> <span class="n">titles</span><span class="p">)):</span>
        <span class="n">density</span> <span class="o">=</span> <span class="n">gaussian_pdf</span><span class="p">(</span><span class="n">grid_x</span><span class="p">,</span> <span class="n">grid_y</span><span class="p">,</span> <span class="n">mean</span><span class="p">,</span> <span class="n">cov</span><span class="p">)</span>

        <span class="c1"># Hollow markers so the isodensity contours stay readable underneath</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
            <span class="n">data</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">],</span>
            <span class="n">data</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">],</span>
            <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span>
            <span class="n">s</span><span class="o">=</span><span class="mi">14</span><span class="p">,</span>
            <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span>
            <span class="n">edgecolors</span><span class="o">=</span><span class="s2">&quot;C0&quot;</span><span class="p">,</span>
            <span class="n">linewidths</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">contour</span><span class="p">(</span>
            <span class="n">grid_x</span><span class="p">,</span>
            <span class="n">grid_y</span><span class="p">,</span>
            <span class="n">density</span><span class="p">,</span>
            <span class="n">levels</span><span class="o">=</span><span class="n">mahalanobis_levels</span><span class="p">(</span><span class="n">cov</span><span class="p">,</span> <span class="n">probabilities</span><span class="p">),</span>
            <span class="n">colors</span><span class="o">=</span><span class="s2">&quot;C1&quot;</span><span class="p">,</span>
            <span class="n">linewidths</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span>
        <span class="p">)</span>

        <span class="c1"># Principal axes, scaled by the square root of the eigenvalues</span>
        <span class="n">eigvals</span><span class="p">,</span> <span class="n">eigvecs</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">(</span><span class="n">cov</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">eigval</span><span class="p">,</span> <span class="n">eigvec</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">eigvals</span><span class="p">,</span> <span class="n">eigvecs</span><span class="o">.</span><span class="n">T</span><span class="p">):</span>
            <span class="n">axis_end</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">eigval</span><span class="p">)</span> <span class="o">*</span> <span class="n">eigvec</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
                <span class="p">[</span><span class="o">-</span><span class="n">axis_end</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">axis_end</span><span class="p">[</span><span class="mi">0</span><span class="p">]],</span>
                <span class="p">[</span><span class="o">-</span><span class="n">axis_end</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">axis_end</span><span class="p">[</span><span class="mi">1</span><span class="p">]],</span>
                <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C3&quot;</span><span class="p">,</span>
                <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
                <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span>
            <span class="p">)</span>

        <span class="n">ax</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="s2">&quot;equal&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_1$&quot;</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_2$&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">title</span><span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="c1"># Save results</span>
    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span>
        <span class="n">n_samples</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span>
        <span class="n">rho</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span>
        <span class="n">condition</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span>
        <span class="n">probabilities</span><span class="o">=</span><span class="n">probabilities</span><span class="p">,</span>
        <span class="n">covariances</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">covariances</span><span class="p">),</span>
        <span class="n">samples</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">samples</span><span class="p">),</span>
        <span class="n">titles</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">titles</span><span class="p">),</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;gaussian_isocontours.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved isocontours in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>50</b></span>
</div>
<p class="param-help">Number of samples drawn per regime. Drawn as hollow markers so the isodensity contours stay readable underneath.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho</span><span class="param-type">float</span><span class="param-default">default <b>0.8</b></span>
</div>
<p class="param-help">Correlation coefficient of the correlated regime.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--condition</span><span class="param-type">float</span><span class="param-default">default <b>50.0</b></span>
</div>
<p class="param-help">Condition number of the ill-conditioned regime.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/gaussian_isocontours</b></span>
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

<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_gaussian_isocontours.json" data-title="context_gaussian_isocontours"></div>
</div>

## Config

`1-context/experiments/context_gaussian_isocontours.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
