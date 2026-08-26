<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_riemann_frobenius_rao</span>
</nav>

# context_riemann_frobenius_rao

Estimation error of the SCM and of Tyler's estimator, in Frobenius norm and in Rao distance

**Tags:** `context`  `riemann`  `robust`  `monte-carlo`

## Run

```sh
uv run python 1-context/riemann_frobenius_rao/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/riemann_frobenius_rao/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">223 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/riemann_frobenius_rao/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># The same two estimators, measured with two different rulers</span>
<span class="c1">#</span>
<span class="c1"># The estimation error of the SCM and of Tyler&#39;s estimator is followed as a</span>
<span class="c1"># function of the number of observations, on Student data, and measured twice:</span>
<span class="c1"># once with the Frobenius norm of the difference, once with the Rao distance,</span>
<span class="c1"># which for the centred Gaussian model is the affine-invariant distance of the</span>
<span class="c1"># cone.</span>
<span class="c1">#</span>
<span class="c1"># Both estimators and the truth are normalised to unit determinant before the</span>
<span class="c1"># comparison: Tyler&#39;s estimator only identifies a shape, and the two rulers</span>
<span class="c1"># would otherwise be comparing scales rather than models.</span>
<span class="c1">#</span>
<span class="c1"># What the figure shows is that the two rulers do not cross at the same place.</span>
<span class="c1"># In the very short-sample regime — a handful of observations more than the</span>
<span class="c1"># dimension — the Frobenius norm already prefers Tyler while the Rao distance</span>
<span class="c1"># still prefers the SCM: the ordering of two estimators is a property of the</span>
<span class="c1"># metric, and not of the estimators alone.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">rich.progress</span><span class="w"> </span><span class="kn">import</span> <span class="n">Progress</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="n">SCMEstimator</span><span class="p">,</span> <span class="n">minimize_tyler_fixed_point</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.elliptical</span><span class="w"> </span><span class="kn">import</span> <span class="n">StudentTDistribution</span><span class="p">,</span> <span class="n">sample_elliptical</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.manifolds</span><span class="w"> </span><span class="kn">import</span> <span class="n">HermitianPositiveDefinite</span>


<span class="k">def</span><span class="w"> </span><span class="nf">normalize_determinant</span><span class="p">(</span><span class="n">matrix</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Rescale to unit determinant, the only scale Tyler&#39;s estimator fixes.&quot;&quot;&quot;</span>
    <span class="n">n_features</span> <span class="o">=</span> <span class="n">matrix</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">matrix</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">matrix</span><span class="p">)</span> <span class="o">**</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">/</span> <span class="n">n_features</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">scatter_matrix</span><span class="p">(</span><span class="n">n_features</span><span class="p">,</span> <span class="n">condition</span><span class="p">,</span> <span class="n">rng</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Ill-conditioned scatter matrix of unit determinant.&quot;&quot;&quot;</span>
    <span class="n">eigenvalues</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">logspace</span><span class="p">(</span>
        <span class="o">-</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">condition</span><span class="p">),</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">condition</span><span class="p">),</span> <span class="n">n_features</span>
    <span class="p">)</span>
    <span class="n">rotation</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">qr</span><span class="p">(</span><span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">n_features</span><span class="p">,</span> <span class="n">n_features</span><span class="p">)))[</span><span class="mi">0</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">normalize_determinant</span><span class="p">(</span><span class="n">rotation</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">eigenvalues</span><span class="p">)</span> <span class="o">@</span> <span class="n">rotation</span><span class="o">.</span><span class="n">T</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Estimation error of the SCM and of Tyler&#39;s estimator, in Frobenius &quot;</span>
        <span class="s2">&quot;norm and in Rao distance.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">7</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Dimension of the observations.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">9</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">12</span><span class="p">,</span> <span class="mi">14</span><span class="p">,</span> <span class="mi">18</span><span class="p">,</span> <span class="mi">25</span><span class="p">,</span> <span class="mi">40</span><span class="p">,</span> <span class="mi">70</span><span class="p">,</span> <span class="mi">120</span><span class="p">,</span> <span class="mi">200</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Sample sizes at which the error is evaluated. The first ones are &quot;</span>
             <span class="s2">&quot;just above the dimension, which is where the two metrics &quot;</span>
             <span class="s2">&quot;disagree.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_trials&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">500</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of MC-trials per sample size.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dof&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">3.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Degrees of freedom of the Student data. Heavy enough for the SCM &quot;</span>
             <span class="s2">&quot;to suffer, light enough for its covariance to exist.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--condition&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">50.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Condition number of the true scatter matrix.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--iter_max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">300</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Fixed-point iterations for Tyler.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/riemann_frobenius_rao&quot;</span><span class="p">,</span>
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
        <span class="s2">&quot;--axis_width&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.45</span><span class="se">\\</span><span class="s2">textwidth&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of a single panel in the exported PGFPlots figure. Set &quot;</span>
             <span class="s2">&quot;here rather than patched into the .tex afterwards, so that a &quot;</span>
             <span class="s2">&quot;re-sync into the dissertation does not undo it.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_height&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;4.6cm&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Height of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
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

    <span class="n">d</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span>
    <span class="n">manifold</span> <span class="o">=</span> <span class="n">HermitianPositiveDefinite</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>

    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>
    <span class="n">scatter</span> <span class="o">=</span> <span class="n">scatter_matrix</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">rng</span><span class="p">)</span>
    <span class="n">scatter_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">scatter</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">mean_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">d</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">distribution</span> <span class="o">=</span> <span class="n">StudentTDistribution</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>

    <span class="n">estimators</span> <span class="o">=</span> <span class="p">(</span><span class="s2">&quot;scm&quot;</span><span class="p">,</span> <span class="s2">&quot;Tyler&quot;</span><span class="p">)</span>
    <span class="n">errors</span> <span class="o">=</span> <span class="p">{</span>
        <span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">metric</span><span class="p">):</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">))</span>
        <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">estimators</span>
        <span class="k">for</span> <span class="n">metric</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">&quot;frobenius&quot;</span><span class="p">,</span> <span class="s2">&quot;rao&quot;</span><span class="p">)</span>
    <span class="p">}</span>

    <span class="k">with</span> <span class="n">Progress</span><span class="p">()</span> <span class="k">as</span> <span class="n">progress</span><span class="p">:</span>
        <span class="n">task</span> <span class="o">=</span> <span class="n">progress</span><span class="o">.</span><span class="n">add_task</span><span class="p">(</span>
            <span class="s2">&quot;Monte-Carlo&quot;</span><span class="p">,</span> <span class="n">total</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">)</span> <span class="o">*</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span>
        <span class="p">)</span>
        <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="n">n_samples</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">):</span>
            <span class="k">for</span> <span class="n">trial</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">):</span>
                <span class="n">data</span> <span class="o">=</span> <span class="n">sample_elliptical</span><span class="p">(</span>
                    <span class="n">n_samples</span><span class="p">,</span> <span class="n">mean_device</span><span class="p">,</span> <span class="n">scatter_device</span><span class="p">,</span> <span class="n">distribution</span><span class="p">,</span>
                    <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="mi">1000</span> <span class="o">*</span> <span class="n">index</span> <span class="o">+</span> <span class="n">trial</span><span class="p">,</span>
                <span class="p">)</span>
                <span class="n">tyler</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">minimize_tyler_fixed_point</span><span class="p">(</span>
                    <span class="n">data</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span>
                <span class="p">)</span>
                <span class="n">estimates</span> <span class="o">=</span> <span class="p">{</span>
                    <span class="s2">&quot;scm&quot;</span><span class="p">:</span> <span class="n">normalize_determinant</span><span class="p">(</span>
                        <span class="n">to_numpy</span><span class="p">(</span><span class="n">SCMEstimator</span><span class="p">(</span><span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span><span class="o">.</span><span class="n">compute</span><span class="p">(</span><span class="n">data</span><span class="p">))</span>
                    <span class="p">),</span>
                    <span class="s2">&quot;Tyler&quot;</span><span class="p">:</span> <span class="n">normalize_determinant</span><span class="p">(</span><span class="n">to_numpy</span><span class="p">(</span><span class="n">tyler</span><span class="p">)),</span>
                <span class="p">}</span>
                <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">estimate</span> <span class="ow">in</span> <span class="n">estimates</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
                    <span class="n">errors</span><span class="p">[(</span><span class="n">name</span><span class="p">,</span> <span class="s2">&quot;frobenius&quot;</span><span class="p">)][</span><span class="n">index</span><span class="p">,</span> <span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span>
                        <span class="n">estimate</span> <span class="o">-</span> <span class="n">scatter</span><span class="p">,</span> <span class="nb">ord</span><span class="o">=</span><span class="s2">&quot;fro&quot;</span>
                    <span class="p">)</span>
                    <span class="n">errors</span><span class="p">[(</span><span class="n">name</span><span class="p">,</span> <span class="s2">&quot;rao&quot;</span><span class="p">)][</span><span class="n">index</span><span class="p">,</span> <span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span>
                        <span class="n">manifold</span><span class="o">.</span><span class="n">dist</span><span class="p">(</span>
                            <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">estimate</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span> <span class="n">scatter_device</span>
                        <span class="p">)</span>
                    <span class="p">)</span>
                <span class="n">progress</span><span class="o">.</span><span class="n">advance</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>

    <span class="n">means</span> <span class="o">=</span> <span class="p">{</span><span class="n">key</span><span class="p">:</span> <span class="n">value</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span> <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">errors</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span>

    <span class="n">colors</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;scm&quot;</span><span class="p">:</span> <span class="s2">&quot;C1&quot;</span><span class="p">,</span> <span class="s2">&quot;Tyler&quot;</span><span class="p">:</span> <span class="s2">&quot;C2&quot;</span><span class="p">}</span>
    <span class="n">markers</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;scm&quot;</span><span class="p">:</span> <span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="s2">&quot;Tyler&quot;</span><span class="p">:</span> <span class="s2">&quot;s&quot;</span><span class="p">}</span>
    <span class="n">titles</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;frobenius&quot;</span><span class="p">:</span> <span class="sa">r</span><span class="s2">&quot;norme de Frobenius&quot;</span><span class="p">,</span>
        <span class="s2">&quot;rao&quot;</span><span class="p">:</span> <span class="sa">r</span><span class="s2">&quot;distance de Rao&quot;</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">n_samples</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">)</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">))</span>
    <span class="k">for</span> <span class="n">column</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">metric</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="p">(</span><span class="s2">&quot;frobenius&quot;</span><span class="p">,</span> <span class="s2">&quot;rao&quot;</span><span class="p">))):</span>
        <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">estimators</span><span class="p">:</span>
            <span class="c1"># Labelled on the left panel only: matplot2tikz gathers the</span>
            <span class="c1"># labelled curves of every axis into the one exported legend.</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
                <span class="n">n_samples</span><span class="p">,</span> <span class="n">means</span><span class="p">[(</span><span class="n">name</span><span class="p">,</span> <span class="n">metric</span><span class="p">)],</span>
                <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">marker</span><span class="o">=</span><span class="n">markers</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
                <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.4</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">name</span> <span class="k">if</span> <span class="n">column</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
            <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$N$&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">titles</span><span class="p">[</span><span class="n">metric</span><span class="p">])</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;erreur moyenne&quot;</span><span class="p">)</span>
    <span class="c1"># Legend below the panels rather than inside one of them, see the other</span>
    <span class="c1"># Riemannian figures: at this width an inner legend covers the curves.</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;upper left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.45</span><span class="p">),</span> <span class="n">ncol</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
        <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;d = </span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">, Student nu = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, condition </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> trials&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="n">n</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">):</span>
        <span class="n">frobenius</span> <span class="o">=</span> <span class="p">{</span><span class="n">name</span><span class="p">:</span> <span class="n">means</span><span class="p">[(</span><span class="n">name</span><span class="p">,</span> <span class="s2">&quot;frobenius&quot;</span><span class="p">)][</span><span class="n">index</span><span class="p">]</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">estimators</span><span class="p">}</span>
        <span class="n">rao</span> <span class="o">=</span> <span class="p">{</span><span class="n">name</span><span class="p">:</span> <span class="n">means</span><span class="p">[(</span><span class="n">name</span><span class="p">,</span> <span class="s2">&quot;rao&quot;</span><span class="p">)][</span><span class="n">index</span><span class="p">]</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">estimators</span><span class="p">}</span>
        <span class="nb">print</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;  N = </span><span class="si">{</span><span class="n">n</span><span class="si">:</span><span class="s2">4d</span><span class="si">}</span><span class="s2">   Frobenius: scm </span><span class="si">{</span><span class="n">frobenius</span><span class="p">[</span><span class="s1">&#39;scm&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">8.3f</span><span class="si">}</span><span class="s2"> &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;Tyler </span><span class="si">{</span><span class="n">frobenius</span><span class="p">[</span><span class="s1">&#39;Tyler&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">8.3f</span><span class="si">}</span><span class="s2"> -&gt; &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="nb">min</span><span class="p">(</span><span class="n">frobenius</span><span class="p">,</span><span class="w"> </span><span class="n">key</span><span class="o">=</span><span class="n">frobenius</span><span class="o">.</span><span class="n">get</span><span class="p">)</span><span class="si">:</span><span class="s2">5</span><span class="si">}</span><span class="s2">   |   &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;Rao: scm </span><span class="si">{</span><span class="n">rao</span><span class="p">[</span><span class="s1">&#39;scm&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">6.3f</span><span class="si">}</span><span class="s2"> Tyler </span><span class="si">{</span><span class="n">rao</span><span class="p">[</span><span class="s1">&#39;Tyler&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">6.3f</span><span class="si">}</span><span class="s2"> -&gt; &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="nb">min</span><span class="p">(</span><span class="n">rao</span><span class="p">,</span><span class="w"> </span><span class="n">key</span><span class="o">=</span><span class="n">rao</span><span class="o">.</span><span class="n">get</span><span class="p">)</span><span class="si">:</span><span class="s2">5</span><span class="si">}</span><span class="s2">&quot;</span>
        <span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_features</span><span class="o">=</span><span class="n">d</span><span class="p">,</span> <span class="n">n_trials</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="p">,</span>
        <span class="n">condition</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="n">n_samples</span><span class="p">,</span> <span class="n">scatter</span><span class="o">=</span><span class="n">scatter</span><span class="p">,</span>
        <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2">_</span><span class="si">{</span><span class="n">metric</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">errors</span><span class="p">[(</span><span class="n">name</span><span class="p">,</span> <span class="n">metric</span><span class="p">)]</span>
           <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">estimators</span> <span class="k">for</span> <span class="n">metric</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">&quot;frobenius&quot;</span><span class="p">,</span> <span class="s2">&quot;rao&quot;</span><span class="p">)},</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;erreurrao.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved error curves in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_features</span><span class="param-type">int</span><span class="param-default">default <b>7</b></span>
</div>
<p class="param-help">Dimension of the observations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>[9, 10, 12, 14, 18, 25, 40, 70, 120, 200]</b></span>
</div>
<p class="param-help">Sample sizes at which the error is evaluated. The first ones are just above the dimension, which is where the two metrics disagree.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_trials</span><span class="param-type">int</span><span class="param-default">default <b>500</b></span>
</div>
<p class="param-help">Number of MC-trials per sample size.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dof</span><span class="param-type">float</span><span class="param-default">default <b>3.0</b></span>
</div>
<p class="param-help">Degrees of freedom of the Student data. Heavy enough for the SCM to suffer, light enough for its covariance to exist.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--condition</span><span class="param-type">float</span><span class="param-default">default <b>50.0</b></span>
</div>
<p class="param-help">Condition number of the true scatter matrix.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--iter_max</span><span class="param-type">int</span><span class="param-default">default <b>300</b></span>
</div>
<p class="param-help">Fixed-point iterations for Tyler.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/riemann_frobenius_rao</b></span>
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
<span class="param-flag">--axis_width</span><span class="param-type">str</span><span class="param-default">default <b>0.45\textwidth</b></span>
</div>
<p class="param-help">Width of a single panel in the exported PGFPlots figure. Set here rather than patched into the .tex afterwards, so that a re-sync into the dissertation does not undo it.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_height</span><span class="param-type">str</span><span class="param-default">default <b>4.6cm</b></span>
</div>
<p class="param-help">Height of a single panel in the exported PGFPlots figure.</p>
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

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-08-19</span><br>
  <code>--n_features</code> <span class='mn-default'>7</span><br>
  <code>--n_samples</code> <span class='mn-default'>[9, 10, 12, 14, 18, 25, 40, 70, 120, 200]</span><br>
  <code>--n_trials</code> <span class='mn-default'>500</span><br>
  <code>--dof</code> <span class='mn-default'>3.0</span><br>
  <code>--condition</code> <span class='mn-default'>50.0</span><br>
  <code>--iter_max</code> <span class='mn-default'>300</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--axis_width</code> <span class='mn-default'>0.45\textwidth</span><br>
  <code>--axis_height</code> <span class='mn-default'>4.6cm</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_frobenius_rao.json" data-title="context_riemann_frobenius_rao"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Monte-Carlo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
d = 7, Student nu = 3, condition 50, 500 trials
  N =    9   Frobenius: scm   32.571 Tyler   31.735 -&gt; Tyler   |   Rao: scm  4.372 Tyler  4.965 -&gt; scm  
  N =   10   Frobenius: scm   24.057 Tyler   20.013 -&gt; Tyler   |   Rao: scm  3.987 Tyler  4.153 -&gt; scm  
  N =   12   Frobenius: scm   18.076 Tyler   13.678 -&gt; Tyler   |   Rao: scm  3.470 Tyler  3.345 -&gt; Tyler
  N =   14   Frobenius: scm   18.641 Tyler    9.738 -&gt; Tyler   |   Rao: scm  3.201 Tyler  2.887 -&gt; Tyler
  N =   18   Frobenius: scm   12.911 Tyler    6.852 -&gt; Tyler   |   Rao: scm  2.777 Tyler  2.365 -&gt; Tyler
  N =   25   Frobenius: scm   10.349 Tyler    5.000 -&gt; Tyler   |   Rao: scm  2.354 Tyler  1.851 -&gt; Tyler
  N =   40   Frobenius: scm    8.084 Tyler    3.369 -&gt; Tyler   |   Rao: scm  1.973 Tyler  1.397 -&gt; Tyler
  N =   70   Frobenius: scm    4.907 Tyler    2.352 -&gt; Tyler   |   Rao: scm  1.568 Tyler  1.033 -&gt; Tyler
  N =  120   Frobenius: scm    4.149 Tyler    1.711 -&gt; Tyler   |   Rao: scm  1.335 Tyler  0.768 -&gt; Tyler
  N =  200   Frobenius: scm    3.766 Tyler    1.329 -&gt; Tyler   |   Rao: scm  1.149 Tyler  0.589 -&gt; Tyler
Saved error curves in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_riemann_frobenius_rao/run_33/erreurrao.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_riemann_frobenius_rao.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
