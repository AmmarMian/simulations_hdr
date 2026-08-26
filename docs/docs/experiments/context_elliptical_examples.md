<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_elliptical_examples</span>
</nav>

# context_elliptical_examples

Isodensity contours and draws for elliptical distributions sharing one scatter matrix

**Tags:** `context`  `elliptical`  `distributions`  `illustration`

## Run

```sh
uv run python 1-context/elliptical_examples/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/elliptical_examples/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">210 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/elliptical_examples/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Isodensity contours and draws for several elliptical distributions</span>
<span class="c1">#</span>
<span class="c1"># All panels share the same scatter matrix, so the elliptical symmetry is</span>
<span class="c1"># common to all of them: only the law of the modular variate — that is, the</span>
<span class="c1"># density generator — changes, which is what the tails show.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">matplot2tikz</span><span class="w"> </span><span class="kn">import</span> <span class="n">save</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.elliptical</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">GaussianDistribution</span><span class="p">,</span>
    <span class="n">StudentTDistribution</span><span class="p">,</span>
    <span class="n">KDistribution</span><span class="p">,</span>
    <span class="n">GeneralizedGaussianDistribution</span><span class="p">,</span>
    <span class="n">sample_elliptical</span><span class="p">,</span>
    <span class="n">isodensity_ellipse</span><span class="p">,</span>
<span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">build_distributions</span><span class="p">(</span><span class="n">names</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">dof_student</span><span class="p">,</span> <span class="n">dof_k</span><span class="p">,</span> <span class="n">shape_gengauss</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Instantiate the requested distributions, all normalised to Cov = Xi.&quot;&quot;&quot;</span>
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
        <span class="k">return</span> <span class="sa">rf</span><span class="s2">&quot;$t,\ \nu = </span><span class="si">{</span><span class="n">distribution</span><span class="o">.</span><span class="n">dof</span><span class="si">:</span><span class="s2">.0f</span><span class="si">}</span><span class="s2">$&quot;</span>
    <span class="k">if</span> <span class="n">name</span> <span class="o">==</span> <span class="s2">&quot;k&quot;</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">rf</span><span class="s2">&quot;$K,\ \nu = </span><span class="si">{</span><span class="n">distribution</span><span class="o">.</span><span class="n">dof</span><span class="si">:</span><span class="s2">.0f</span><span class="si">}</span><span class="s2">$&quot;</span>
    <span class="k">return</span> <span class="sa">rf</span><span class="s2">&quot;$\mathcal</span><span class="se">{{</span><span class="s2">GG</span><span class="se">}}</span><span class="s2">,\ s = </span><span class="si">{</span><span class="n">distribution</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">$&quot;</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Isodensity contours of elliptical distributions sharing a scatter matrix.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--distributions&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;gaussian&quot;</span><span class="p">,</span> <span class="s2">&quot;student&quot;</span><span class="p">,</span> <span class="s2">&quot;k&quot;</span><span class="p">,</span> <span class="s2">&quot;gengauss&quot;</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Distributions to show, one panel each. The Gaussian acts as the &quot;</span>
             <span class="s2">&quot;reference against which the tails are read.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of samples drawn per panel. Drawn as hollow markers so the &quot;</span>
             <span class="s2">&quot;isodensity contours stay readable underneath.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--rho&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Correlation coefficient of the shared scatter matrix.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dof_student&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">3.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Degrees of freedom of the t distribution (&gt;2 for a finite covariance).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dof_k&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">2.0</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Texture shape of the K distribution.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--shape_gengauss&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Exponent s of the generalized Gaussian (s&lt;1 gives heavier tails).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span>
        <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/elliptical_examples&quot;</span><span class="p">,</span>
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
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--backend&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;numpy&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Compute backend for the draws (numpy, torch-cpu, torch-mps, ...).&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;random seed generation base seed&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="c1"># Constant(s)</span>
    <span class="n">d</span> <span class="o">=</span> <span class="mi">2</span>
    <span class="n">mean</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">d</span><span class="p">)</span>
    <span class="n">probabilities</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.9</span><span class="p">,</span> <span class="mf">0.99</span><span class="p">])</span>

    <span class="c1"># A single scatter matrix shared by every panel</span>
    <span class="n">scatter</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">],</span> <span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">]])</span>

    <span class="n">distributions</span> <span class="o">=</span> <span class="n">build_distributions</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">,</span> <span class="n">d</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">dof_student</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">dof_k</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">shape_gengauss</span><span class="p">,</span>
        <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">scatter_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">scatter</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">mean_device</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">mean</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">samples</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">to_numpy</span><span class="p">(</span>
            <span class="n">sample_elliptical</span><span class="p">(</span>
                <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span> <span class="n">mean_device</span><span class="p">,</span> <span class="n">scatter_device</span><span class="p">,</span> <span class="n">distribution</span><span class="p">,</span>
                <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="n">offset</span><span class="p">,</span>
            <span class="p">)</span>
        <span class="p">)</span>
        <span class="k">for</span> <span class="n">offset</span><span class="p">,</span> <span class="n">distribution</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">distributions</span><span class="p">)</span>
    <span class="p">]</span>
    <span class="n">contours</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">[</span><span class="n">isodensity_ellipse</span><span class="p">(</span><span class="n">scatter</span><span class="p">,</span> <span class="n">distribution</span><span class="p">,</span> <span class="n">p</span><span class="p">)</span> <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">probabilities</span><span class="p">]</span>
        <span class="k">for</span> <span class="n">distribution</span> <span class="ow">in</span> <span class="n">distributions</span>
    <span class="p">]</span>

    <span class="c1"># Common extent, driven by the widest outer contour so that the panels stay</span>
    <span class="c1"># comparable and the framing does not move with --seed.</span>
    <span class="n">limit</span> <span class="o">=</span> <span class="mf">1.1</span> <span class="o">*</span> <span class="nb">max</span><span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">curves</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span><span class="o">.</span><span class="n">max</span><span class="p">()</span> <span class="k">for</span> <span class="n">curves</span> <span class="ow">in</span> <span class="n">contours</span>
    <span class="p">)</span>

    <span class="c1"># A grid rather than a single row: squarer figures sit better both in the</span>
    <span class="c1"># dissertation&#39;s text width and on the docs page.</span>
    <span class="n">n_panels</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">distributions</span><span class="p">)</span>
    <span class="n">n_cols</span> <span class="o">=</span> <span class="nb">min</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">n_panels</span><span class="p">)</span>
    <span class="n">n_rows</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">n_panels</span> <span class="o">/</span> <span class="n">n_cols</span><span class="p">))</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span>
        <span class="n">n_rows</span><span class="p">,</span> <span class="n">n_cols</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="n">n_cols</span><span class="p">,</span> <span class="mf">3.4</span> <span class="o">*</span> <span class="n">n_rows</span><span class="p">),</span>
        <span class="n">sharex</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">axes</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">atleast_1d</span><span class="p">(</span><span class="n">axes</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>

    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">name</span><span class="p">,</span> <span class="n">distribution</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">curves</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span>
        <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">,</span> <span class="n">distributions</span><span class="p">,</span> <span class="n">samples</span><span class="p">,</span> <span class="n">contours</span><span class="p">)</span>
    <span class="p">):</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
            <span class="n">data</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">],</span> <span class="n">data</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">],</span>
            <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">s</span><span class="o">=</span><span class="mi">14</span><span class="p">,</span> <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">edgecolors</span><span class="o">=</span><span class="s2">&quot;C0&quot;</span><span class="p">,</span> <span class="n">linewidths</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span>
            <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="k">for</span> <span class="n">curve</span> <span class="ow">in</span> <span class="n">curves</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">curve</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">curve</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C1&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>

        <span class="n">ax</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="s2">&quot;equal&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o">//</span> <span class="n">n_cols</span> <span class="o">==</span> <span class="n">n_rows</span> <span class="o">-</span> <span class="mi">1</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_1$&quot;</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o">%</span> <span class="n">n_cols</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$x_2$&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">panel_title</span><span class="p">(</span><span class="n">name</span><span class="p">,</span> <span class="n">distribution</span><span class="p">))</span>

    <span class="c1"># Hide any leftover cell when the panel count does not fill the grid</span>
    <span class="k">for</span> <span class="n">ax</span> <span class="ow">in</span> <span class="n">axes</span><span class="p">[</span><span class="n">n_panels</span><span class="p">:]:</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="c1"># Save results</span>
    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span>
        <span class="n">n_samples</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span>
        <span class="n">rho</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span>
        <span class="n">dof_student</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof_student</span><span class="p">,</span>
        <span class="n">dof_k</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof_k</span><span class="p">,</span>
        <span class="n">shape_gengauss</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">shape_gengauss</span><span class="p">,</span>
        <span class="n">probabilities</span><span class="o">=</span><span class="n">probabilities</span><span class="p">,</span>
        <span class="n">scatter</span><span class="o">=</span><span class="n">scatter</span><span class="p">,</span>
        <span class="n">names</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">distributions</span><span class="p">),</span>
        <span class="n">samples</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">samples</span><span class="p">),</span>
        <span class="n">contours</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">curves</span><span class="p">)</span> <span class="k">for</span> <span class="n">curves</span> <span class="ow">in</span> <span class="n">contours</span><span class="p">]),</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;elliptical_tails.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved elliptical examples in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

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
<p class="param-help">Distributions to show, one panel each. The Gaussian acts as the reference against which the tails are read.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>50</b></span>
</div>
<p class="param-help">Number of samples drawn per panel. Drawn as hollow markers so the isodensity contours stay readable underneath.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--rho</span><span class="param-type">float</span><span class="param-default">default <b>0.8</b></span>
</div>
<p class="param-help">Correlation coefficient of the shared scatter matrix.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dof_student</span><span class="param-type">float</span><span class="param-default">default <b>3.0</b></span>
</div>
<p class="param-help">Degrees of freedom of the t distribution (&gt;2 for a finite covariance).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dof_k</span><span class="param-type">float</span><span class="param-default">default <b>2.0</b></span>
</div>
<p class="param-help">Texture shape of the K distribution.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--shape_gengauss</span><span class="param-type">float</span><span class="param-default">default <b>0.5</b></span>
</div>
<p class="param-help">Exponent s of the generalized Gaussian (s&lt;1 gives heavier tails).</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/elliptical_examples</b></span>
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
<p class="param-help">Compute backend for the draws (numpy, torch-cpu, torch-mps, ...).</p>
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
  <span class="mn-date">Generated: 2026-08-17</span><br>
  <code>--distributions</code> <span class='mn-default'>['gaussian', 'student', 'k', 'gengauss']</span><br>
  <code>--n_samples</code> <span class='mn-default'>50</span><br>
  <code>--rho</code> <span class='mn-default'>0.8</span><br>
  <code>--dof_student</code> <span class='mn-default'>3.0</span><br>
  <code>--dof_k</code> <span class='mn-default'>2.0</span><br>
  <code>--shape_gengauss</code> <span class='mn-default'>0.5</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_elliptical_examples.json" data-title="context_elliptical_examples"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Saved elliptical examples in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_elliptical_examples/run_13/elliptical_tails.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_elliptical_examples.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
