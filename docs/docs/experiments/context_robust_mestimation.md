<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_robust_mestimation</span>
</nav>

# context_robust_mestimation

Concentration ellipses of the SCM, the model MLE and Tyler's estimator on a single draw

**Tags:** `context`  `robust`  `m-estimation`  `illustration`

## Run

```sh
uv run python 1-context/robust_mestimation/main.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--distributions [&#x27;gaussian&#x27;, &#x27;student&#x27;, &#x27;k&#x27;, &#x27;gengauss&#x27;]</code><br>
  <code>--n_samples 50</code><br>
  <code>--rho 0.8</code><br>
  <code>--dof_student 2.1</code><br>
  <code>--dof_k 0.1</code><br>
  <code>--shape_gengauss 0.15</code><br>
  <code>--iter_max 100</code><br>
  <code>--tol 1e-08</code><br>
  <code>--backend numpy</code><br>
  <code>--seed 42</code><br>
  <span class="mn-date">f81a05e · 2026-08-17</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/robust_mestimation/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">282 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/robust_mestimation/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># A single estimation, seen in the plane</span>
<span class="c1">#</span>
<span class="c1"># One panel per elliptical model, all sharing the same true shape matrix. Each</span>
<span class="c1"># shows one draw of the data together with the concentration ellipse of three</span>
<span class="c1"># estimators: the SCM, the maximum-likelihood M-estimator of that model, and</span>
<span class="c1"># Tyler&#39;s distribution-free estimator.</span>
<span class="c1">#</span>
<span class="c1"># The fixed-point engine and Tyler&#39;s estimator come from hdrlib.core.estimation</span>
<span class="c1"># unchanged; each model supplies its own weight function through the public</span>
<span class="c1"># m_estimator_function hook.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">matplot2tikz</span><span class="w"> </span><span class="kn">import</span> <span class="n">save</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">SCMEstimator</span><span class="p">,</span>
    <span class="n">TylerEstimator</span><span class="p">,</span>
    <span class="n">fixed_point_m_estimation_centered</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.elliptical</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">GaussianDistribution</span><span class="p">,</span>
    <span class="n">StudentTDistribution</span><span class="p">,</span>
    <span class="n">KDistribution</span><span class="p">,</span>
    <span class="n">GeneralizedGaussianDistribution</span><span class="p">,</span>
    <span class="n">sample_elliptical</span><span class="p">,</span>
<span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">build_distributions</span><span class="p">(</span><span class="n">names</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">dof_student</span><span class="p">,</span> <span class="n">dof_k</span><span class="p">,</span> <span class="n">shape_gengauss</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
    <span class="n">factories</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;gaussian&quot;</span><span class="p">:</span> <span class="k">lambda</span><span class="p">:</span> <span class="n">GaussianDistribution</span><span class="p">(</span><span class="n">n_features</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">),</span>
        <span class="s2">&quot;student&quot;</span><span class="p">:</span> <span class="k">lambda</span><span class="p">:</span> <span class="n">StudentTDistribution</span><span class="p">(</span>
            <span class="n">n_features</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">dof_student</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span>
        <span class="p">),</span>
        <span class="s2">&quot;k&quot;</span><span class="p">:</span> <span class="k">lambda</span><span class="p">:</span> <span class="n">KDistribution</span><span class="p">(</span><span class="n">n_features</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">dof_k</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span><span class="p">),</span>
        <span class="s2">&quot;gengauss&quot;</span><span class="p">:</span> <span class="k">lambda</span><span class="p">:</span> <span class="n">GeneralizedGaussianDistribution</span><span class="p">(</span>
            <span class="n">n_features</span><span class="p">,</span> <span class="n">shape</span><span class="o">=</span><span class="n">shape_gengauss</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">backend</span>
        <span class="p">),</span>
    <span class="p">}</span>
    <span class="n">unknown</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">names</span><span class="p">)</span> <span class="o">-</span> <span class="nb">set</span><span class="p">(</span><span class="n">factories</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">unknown</span><span class="p">:</span>
        <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Unknown distribution(s): </span><span class="si">{</span><span class="nb">sorted</span><span class="p">(</span><span class="n">unknown</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">[</span><span class="n">factories</span><span class="p">[</span><span class="n">name</span><span class="p">]()</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">names</span><span class="p">]</span>


<span class="k">def</span><span class="w"> </span><span class="nf">panel_title</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">distribution</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Math-only title: matplotlib&#39;s mathtext does not take LaTeX accents.&quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="n">name</span> <span class="o">==</span> <span class="s2">&quot;gaussian&quot;</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">r</span><span class="s2">&quot;$\mathcal</span><span class="si">{N}</span><span class="s2">$&quot;</span>
    <span class="k">if</span> <span class="n">name</span> <span class="o">==</span> <span class="s2">&quot;student&quot;</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">rf</span><span class="s2">&quot;$t,\ \nu = </span><span class="si">{</span><span class="n">distribution</span><span class="o">.</span><span class="n">dof</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">$&quot;</span>
    <span class="k">if</span> <span class="n">name</span> <span class="o">==</span> <span class="s2">&quot;k&quot;</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">rf</span><span class="s2">&quot;$K,\ \nu = </span><span class="si">{</span><span class="n">distribution</span><span class="o">.</span><span class="n">dof</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">$&quot;</span>
    <span class="k">return</span> <span class="sa">rf</span><span class="s2">&quot;$\mathcal</span><span class="se">{{</span><span class="s2">GG</span><span class="se">}}</span><span class="s2">,\ s = </span><span class="si">{</span><span class="n">distribution</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">$&quot;</span>


<span class="k">def</span><span class="w"> </span><span class="nf">normalize_shape</span><span class="p">(</span><span class="n">matrix</span><span class="p">,</span> <span class="n">n_features</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Normalise by the trace so only the shape is compared.&quot;&quot;&quot;</span>
    <span class="k">return</span> <span class="n">n_features</span> <span class="o">*</span> <span class="n">matrix</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">trace</span><span class="p">(</span><span class="n">matrix</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">concentration_ellipse</span><span class="p">(</span><span class="n">shape</span><span class="p">,</span> <span class="n">radius</span><span class="p">,</span> <span class="n">n_points</span><span class="o">=</span><span class="mi">300</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Ellipse {x : x^T shape^{-1} x = radius^2}, as a (2, n_points) array.&quot;&quot;&quot;</span>
    <span class="n">angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="p">,</span> <span class="n">n_points</span><span class="p">)</span>
    <span class="n">circle</span> <span class="o">=</span> <span class="n">radius</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">angles</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">angles</span><span class="p">)])</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">cholesky</span><span class="p">(</span><span class="n">shape</span><span class="p">)</span> <span class="o">@</span> <span class="n">circle</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Concentration ellipses of the SCM, the model MLE and Tyler&#39;s estimator.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--distributions&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;gaussian&quot;</span><span class="p">,</span> <span class="s2">&quot;student&quot;</span><span class="p">,</span> <span class="s2">&quot;k&quot;</span><span class="p">,</span> <span class="s2">&quot;gengauss&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Models to show, one panel each.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of observations of the single estimation shown. Kept small &quot;</span>
             <span class="s2">&quot;on purpose: with a large support every estimator is accurate and &quot;</span>
             <span class="s2">&quot;the ellipses become indistinguishable.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--rho&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Correlation of the shared true shape matrix.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dof_student&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">2.1</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Degrees of freedom of the t model. Just above 2, where the &quot;</span>
             <span class="s2">&quot;covariance still exists but the tails are very heavy.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dof_k&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Texture shape of the K model; the smaller, the heavier.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--shape_gengauss&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.15</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Exponent s of the generalized Gaussian; s&lt;1 gives heavier tails.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--iter_max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Fixed-point iterations.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-8</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Fixed-point tolerance.&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/robust_mestimation&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Output directory for LaTeX exports (injected by qanat, or set manually).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--show-interactive&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Show plots interactively with matplotlib.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--export&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="n">argparse</span><span class="o">.</span><span class="n">BooleanOptionalAction</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Save TikZ/PGFPlots figure (.tex) (default: True).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--backend&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;numpy&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Compute backend (numpy, torch-cpu, torch-mps, ...).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;random seed generation base seed&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="c1"># Constant(s)</span>
    <span class="n">d</span> <span class="o">=</span> <span class="mi">2</span>
    <span class="n">mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">d</span><span class="p">)</span>
    <span class="c1"># Radius of the drawn ellipses, in units of the Mahalanobis distance</span>
    <span class="n">radius</span> <span class="o">=</span> <span class="mf">2.0</span>

    <span class="n">shape_true</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">],</span> <span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">]])</span>
    <span class="n">shape_true</span> <span class="o">=</span> <span class="n">normalize_shape</span><span class="p">(</span><span class="n">shape_true</span><span class="p">,</span> <span class="n">d</span><span class="p">)</span>

    <span class="n">distributions</span> <span class="o">=</span> <span class="n">build_distributions</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">dof_student</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">dof_k</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">shape_gengauss</span><span class="p">,</span>
        <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">mean_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">mean</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">shape_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">shape_true</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>

    <span class="n">samples</span><span class="p">,</span> <span class="n">estimates</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">offset</span><span class="p">,</span> <span class="n">distribution</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">distributions</span><span class="p">):</span>
        <span class="n">data</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span>
            <span class="n">sample_elliptical</span><span class="p">(</span>
                <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span> <span class="n">mean_device</span><span class="p">,</span> <span class="n">shape_device</span><span class="p">,</span> <span class="n">distribution</span><span class="p">,</span>
                <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="n">offset</span><span class="p">,</span>
            <span class="p">)</span>
        <span class="p">)</span>
        <span class="n">samples</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>

        <span class="n">scm</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">SCMEstimator</span><span class="p">()</span><span class="o">.</span><span class="n">compute</span><span class="p">(</span><span class="n">data</span><span class="p">))</span>
        <span class="n">tyler</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span>
            <span class="n">TylerEstimator</span><span class="p">(</span>
                <span class="n">normalization</span><span class="o">=</span><span class="s2">&quot;trace&quot;</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">tol</span>
            <span class="p">)</span><span class="o">.</span><span class="n">compute</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
        <span class="p">)</span>
        <span class="c1"># The engine is reused as-is; only the weight changes, through the</span>
        <span class="c1"># public m_estimator_function hook.</span>
        <span class="c1">#</span>
        <span class="c1"># No normalisation during the iterations, unlike Tyler: the scale of a</span>
        <span class="c1"># genuine MLE is identifiable, so renormalising at each step would move</span>
        <span class="c1"># the fixed point and bias the estimate. The result is trace-normalised</span>
        <span class="c1"># afterwards, only to compare shapes.</span>
        <span class="n">mle</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span>
            <span class="n">fixed_point_m_estimation_centered</span><span class="p">(</span>
                <span class="n">data</span><span class="p">,</span>
                <span class="n">m_estimator_function</span><span class="o">=</span><span class="n">distribution</span><span class="o">.</span><span class="n">weight_function</span><span class="p">,</span>
                <span class="n">iter_max</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span>
                <span class="n">tol</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">tol</span><span class="p">,</span>
                <span class="n">normalization</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
            <span class="p">)</span>
        <span class="p">)</span>
        <span class="n">estimates</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
            <span class="s2">&quot;scm&quot;</span><span class="p">:</span> <span class="n">normalize_shape</span><span class="p">(</span><span class="n">scm</span><span class="p">,</span> <span class="n">d</span><span class="p">),</span>
            <span class="s2">&quot;mle&quot;</span><span class="p">:</span> <span class="n">normalize_shape</span><span class="p">(</span><span class="n">mle</span><span class="p">,</span> <span class="n">d</span><span class="p">),</span>
            <span class="s2">&quot;tyler&quot;</span><span class="p">:</span> <span class="n">normalize_shape</span><span class="p">(</span><span class="n">tyler</span><span class="p">,</span> <span class="n">d</span><span class="p">),</span>
        <span class="p">})</span>

    <span class="n">styles</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;true&quot;</span><span class="p">:</span> <span class="nb">dict</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="s2">&quot;C7&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="sa">r</span><span class="s2">&quot;vraie $\xi$&quot;</span><span class="p">),</span>
        <span class="s2">&quot;scm&quot;</span><span class="p">:</span> <span class="nb">dict</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="s2">&quot;C1&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;-&quot;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;scm&quot;</span><span class="p">),</span>
        <span class="s2">&quot;mle&quot;</span><span class="p">:</span> <span class="nb">dict</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="s2">&quot;C2&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;-&quot;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;mle&quot;</span><span class="p">),</span>
        <span class="s2">&quot;tyler&quot;</span><span class="p">:</span> <span class="nb">dict</span><span class="p">(</span><span class="n">color</span><span class="o">=</span><span class="s2">&quot;C3&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;-&quot;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;Tyler&quot;</span><span class="p">),</span>
    <span class="p">}</span>

    <span class="c1"># Frame on a high quantile rather than the maximum: a single extreme draw</span>
    <span class="c1"># would otherwise shrink the cloud to a dot. Outliers beyond the frame are</span>
    <span class="c1"># still what drags the SCM ellipse, they are simply not all drawn.</span>
    <span class="n">limit</span> <span class="o">=</span> <span class="mf">1.1</span> <span class="o">*</span> <span class="nb">max</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">quantile</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="mf">0.995</span><span class="p">)</span> <span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">samples</span><span class="p">)</span>

    <span class="n">n_panels</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">distributions</span><span class="p">)</span>
    <span class="n">n_cols</span> <span class="o">=</span> <span class="nb">min</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">n_panels</span><span class="p">)</span>
    <span class="n">n_rows</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">n_panels</span> <span class="o">/</span> <span class="n">n_cols</span><span class="p">))</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span>
        <span class="n">n_rows</span><span class="p">,</span> <span class="n">n_cols</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="n">n_cols</span><span class="p">,</span> <span class="mf">3.4</span> <span class="o">*</span> <span class="n">n_rows</span><span class="p">),</span>
        <span class="n">sharex</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">axes</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">atleast_1d</span><span class="p">(</span><span class="n">axes</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>

    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">name</span><span class="p">,</span> <span class="n">distribution</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">estimate</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span>
        <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">,</span> <span class="n">distributions</span><span class="p">,</span> <span class="n">samples</span><span class="p">,</span> <span class="n">estimates</span><span class="p">)</span>
    <span class="p">):</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
            <span class="n">data</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">],</span> <span class="n">data</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">],</span>
            <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">s</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span> <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">edgecolors</span><span class="o">=</span><span class="s2">&quot;C0&quot;</span><span class="p">,</span> <span class="n">linewidths</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span>
            <span class="n">zorder</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="k">for</span> <span class="n">key</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">&quot;true&quot;</span><span class="p">,</span> <span class="s2">&quot;scm&quot;</span><span class="p">,</span> <span class="s2">&quot;mle&quot;</span><span class="p">,</span> <span class="s2">&quot;tyler&quot;</span><span class="p">):</span>
            <span class="n">shape</span> <span class="o">=</span> <span class="n">shape_true</span> <span class="k">if</span> <span class="n">key</span> <span class="o">==</span> <span class="s2">&quot;true&quot;</span> <span class="k">else</span> <span class="n">estimate</span><span class="p">[</span><span class="n">key</span><span class="p">]</span>
            <span class="n">curve</span> <span class="o">=</span> <span class="n">concentration_ellipse</span><span class="p">(</span><span class="n">shape</span><span class="p">,</span> <span class="n">radius</span><span class="p">)</span>
            <span class="n">style</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="n">styles</span><span class="p">[</span><span class="n">key</span><span class="p">])</span>
            <span class="n">label</span> <span class="o">=</span> <span class="n">style</span><span class="o">.</span><span class="n">pop</span><span class="p">(</span><span class="s2">&quot;label&quot;</span><span class="p">)</span>
            <span class="n">label</span> <span class="o">=</span> <span class="n">label</span> <span class="k">if</span> <span class="n">i</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="kc">None</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">curve</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">curve</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.3</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">label</span><span class="p">,</span> <span class="o">**</span><span class="n">style</span><span class="p">)</span>

        <span class="n">ax</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="s2">&quot;equal&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o">//</span> <span class="n">n_cols</span> <span class="o">==</span> <span class="n">n_rows</span> <span class="o">-</span> <span class="mi">1</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_1$&quot;</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o">%</span> <span class="n">n_cols</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_2$&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">panel_title</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">distribution</span><span class="p">))</span>

    <span class="k">for</span> <span class="n">ax</span> <span class="ow">in</span> <span class="n">axes</span><span class="p">[</span><span class="n">n_panels</span><span class="p">:]:</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>

    <span class="c1"># Legend outside the grid, below it. It is attached to a single axis</span>
    <span class="c1"># rather than to the figure because matplot2tikz exports axis legends but</span>
    <span class="c1"># silently drops figure-level ones.</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;lower left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">1.14</span><span class="p">),</span>
        <span class="n">ncol</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">styles</span><span class="p">),</span> <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">9</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="c1"># Errors, printed and stored, so the visual reading can be checked</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Shape estimation error (Frobenius), N = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">:&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">estimate</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">,</span> <span class="n">estimates</span><span class="p">):</span>
        <span class="n">errors</span> <span class="o">=</span> <span class="p">{</span>
            <span class="n">key</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">value</span> <span class="o">-</span> <span class="n">shape_true</span><span class="p">,</span> <span class="nb">ord</span><span class="o">=</span><span class="s2">&quot;fro&quot;</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">estimate</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
        <span class="p">}</span>
        <span class="nb">print</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;  </span><span class="si">{</span><span class="n">name</span><span class="si">:</span><span class="s2">9</span><span class="si">}</span><span class="s2"> &quot;</span>
            <span class="o">+</span> <span class="s2">&quot;  &quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">key</span><span class="si">}</span><span class="s2">=</span><span class="si">{</span><span class="n">value</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span> <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">errors</span><span class="o">.</span><span class="n">items</span><span class="p">())</span>
        <span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span> <span class="n">rho</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="n">radius</span><span class="o">=</span><span class="n">radius</span><span class="p">,</span>
        <span class="n">dof_student</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof_student</span><span class="p">,</span> <span class="n">dof_k</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof_k</span><span class="p">,</span>
        <span class="n">shape_gengauss</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">shape_gengauss</span><span class="p">,</span>
        <span class="n">names</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">),</span>
        <span class="n">shape_true</span><span class="o">=</span><span class="n">shape_true</span><span class="p">,</span>
        <span class="n">samples</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">samples</span><span class="p">),</span>
        <span class="o">**</span><span class="p">{</span>
            <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">key</span><span class="si">}</span><span class="s2">_</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">estimate</span><span class="p">[</span><span class="n">key</span><span class="p">]</span>
            <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">estimate</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">,</span> <span class="n">estimates</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">key</span> <span class="ow">in</span> <span class="n">estimate</span>
        <span class="p">},</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;scmvstyler.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved ellipses in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--distributions</span><span class="param-type">str</span><span class="param-default">default <b>[&#x27;gaussian&#x27;, &#x27;student&#x27;, &#x27;k&#x27;, &#x27;gengauss&#x27;]</b></span>
</div>
<p class="param-help">Models to show, one panel each.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>50</b></span>
</div>
<p class="param-help">Number of observations of the single estimation shown. Kept small on purpose: with a large support every estimator is accurate and the ellipses become indistinguishable.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho</span><span class="param-type">float</span><span class="param-default">default <b>0.8</b></span>
</div>
<p class="param-help">Correlation of the shared true shape matrix.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dof_student</span><span class="param-type">float</span><span class="param-default">default <b>2.1</b></span>
</div>
<p class="param-help">Degrees of freedom of the t model. Just above 2, where the covariance still exists but the tails are very heavy.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dof_k</span><span class="param-type">float</span><span class="param-default">default <b>0.1</b></span>
</div>
<p class="param-help">Texture shape of the K model; the smaller, the heavier.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--shape_gengauss</span><span class="param-type">float</span><span class="param-default">default <b>0.15</b></span>
</div>
<p class="param-help">Exponent s of the generalized Gaussian; s&lt;1 gives heavier tails.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--iter_max</span><span class="param-type">int</span><span class="param-default">default <b>100</b></span>
</div>
<p class="param-help">Fixed-point iterations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-08</b></span>
</div>
<p class="param-help">Fixed-point tolerance.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/robust_mestimation</b></span>
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
<span class="param-flag">--backend</span><span class="param-type">str</span><span class="param-default">default <b>numpy</b></span>
</div>
<p class="param-help">Compute backend (numpy, torch-cpu, torch-mps, ...).</p>
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
<div class="plotly-wrap" data-src="../../assets/data/context_robust_mestimation.json" data-title="context_robust_mestimation"></div>
</div>

## Config

`1-context/experiments/context_robust_mestimation.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
