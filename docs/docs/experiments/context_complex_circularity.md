<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_complex_circularity</span>
</nav>

# context_complex_circularity

Same covariance, four pseudo-covariances — what circularity buys and what it hides

**Tags:** `context`  `complex`  `circularity`  `illustration`

## Run

```sh
uv run python 1-context/complex_circularity/main.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--rho [0.0, 0.5, 0.5, 0.9]</code><br>
  <code>--phase [0.0, 0.0, 0.3333, 0.3333]</code><br>
  <code>--gamma 1.0</code><br>
  <code>--n_samples 600</code><br>
  <code>--probability 0.9</code><br>
  <code>--axis_width 0.45\textwidth</code><br>
  <code>--axis_height 4.6cm</code><br>
  <code>--seed 42</code><br>
  <span class="mn-date">7812a73 · 2026-08-18</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/complex_circularity/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">232 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/complex_circularity/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># What the covariance of a complex vector does not say</span>
<span class="c1">#</span>
<span class="c1"># One panel per value of the pseudo-covariance, all sharing the *same*</span>
<span class="c1"># covariance. The dashed circle is what the covariance alone predicts; the</span>
<span class="c1"># solid ellipse is the actual concentration curve. They coincide only in the</span>
<span class="c1"># first panel, where the pseudo-covariance vanishes — that is, only under</span>
<span class="c1"># circularity.</span>
<span class="c1">#</span>
<span class="c1"># Everything is done in the scalar case d = 1, where the whole second-order</span>
<span class="c1"># structure is two numbers: the real variance Gamma and the complex</span>
<span class="c1"># pseudo-variance C. Picinbono&#39;s admissibility condition then reads</span>
<span class="c1"># |C| &lt;= Gamma, and the panels sweep that segment.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">matplot2tikz</span><span class="w"> </span><span class="kn">import</span> <span class="n">save</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>


<span class="k">def</span><span class="w"> </span><span class="nf">real_covariance</span><span class="p">(</span><span class="n">gamma</span><span class="p">,</span> <span class="n">pseudo</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Real 2x2 covariance of (Re z, Im z) from the pair (Gamma, C).</span>

<span class="sd">    Inverts the block relations of the dissertation: with</span>
<span class="sd">    Gamma = E{z z*} and C = E{z z}, one has</span>
<span class="sd">    Gamma_x = Re(Gamma + C)/2, Gamma_y = Re(Gamma - C)/2 and</span>
<span class="sd">    Gamma_xy = Im(C - Gamma)/2, which is Im(C)/2 for a real Gamma.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">var_x</span> <span class="o">=</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="p">(</span><span class="n">gamma</span> <span class="o">+</span> <span class="n">pseudo</span><span class="o">.</span><span class="n">real</span><span class="p">)</span>
    <span class="n">var_y</span> <span class="o">=</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="p">(</span><span class="n">gamma</span> <span class="o">-</span> <span class="n">pseudo</span><span class="o">.</span><span class="n">real</span><span class="p">)</span>
    <span class="n">cov_xy</span> <span class="o">=</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="n">pseudo</span><span class="o">.</span><span class="n">imag</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="n">var_x</span><span class="p">,</span> <span class="n">cov_xy</span><span class="p">],</span> <span class="p">[</span><span class="n">cov_xy</span><span class="p">,</span> <span class="n">var_y</span><span class="p">]])</span>


<span class="k">def</span><span class="w"> </span><span class="nf">is_admissible</span><span class="p">(</span><span class="n">gamma</span><span class="p">,</span> <span class="n">pseudo</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Picinbono&#39;s condition, which reads |C| &lt;= Gamma in the scalar case.</span>

<span class="sd">    The error variance of the widely linear predictor of z* from z is</span>
<span class="sd">    P = Gamma - |C|^2 / Gamma, and it must be non-negative.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="k">return</span> <span class="n">gamma</span> <span class="o">-</span> <span class="nb">abs</span><span class="p">(</span><span class="n">pseudo</span><span class="p">)</span> <span class="o">**</span> <span class="mi">2</span> <span class="o">/</span> <span class="n">gamma</span> <span class="o">&gt;=</span> <span class="o">-</span><span class="mf">1e-12</span>


<span class="k">def</span><span class="w"> </span><span class="nf">concentration_ellipse</span><span class="p">(</span><span class="n">covariance</span><span class="p">,</span> <span class="n">probability</span><span class="p">,</span> <span class="n">n_points</span><span class="o">=</span><span class="mi">400</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Curve {v : v^T Sigma^-1 v = q} enclosing a given probability mass.</span>

<span class="sd">    For a bivariate Gaussian the quadratic form is chi-squared with 2 degrees</span>
<span class="sd">    of freedom, whose quantile is available in closed form — which avoids a</span>
<span class="sd">    scipy dependency for a single number.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">radius</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="o">-</span><span class="mf">2.0</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="mf">1.0</span> <span class="o">-</span> <span class="n">probability</span><span class="p">))</span>
    <span class="n">angles</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="p">,</span> <span class="n">n_points</span><span class="p">)</span>
    <span class="n">circle</span> <span class="o">=</span> <span class="n">radius</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">([</span><span class="n">np</span><span class="o">.</span><span class="n">cos</span><span class="p">(</span><span class="n">angles</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">sin</span><span class="p">(</span><span class="n">angles</span><span class="p">)])</span>
    <span class="c1"># A degenerate covariance (|C| = Gamma) is only positive *semi*-definite,</span>
    <span class="c1"># so the Cholesky factor is taken through the eigendecomposition.</span>
    <span class="n">values</span><span class="p">,</span> <span class="n">vectors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">(</span><span class="n">covariance</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">vectors</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">maximum</span><span class="p">(</span><span class="n">values</span><span class="p">,</span> <span class="mf">0.0</span><span class="p">)))</span> <span class="o">@</span> <span class="n">circle</span>


<span class="k">def</span><span class="w"> </span><span class="nf">panel_title</span><span class="p">(</span><span class="n">rho</span><span class="p">,</span> <span class="n">phase</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">rho</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="k">return</span> <span class="sa">r</span><span class="s2">&quot;$C = 0$&quot;</span>
    <span class="k">return</span> <span class="sa">rf</span><span class="s2">&quot;$|C|/\Gamma = </span><span class="si">{</span><span class="n">rho</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">,\ \arg C = </span><span class="si">{</span><span class="n">phase</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">\pi$&quot;</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Same covariance, four pseudo-covariances: circularity seen in the plane.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--rho&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.9</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Moduli |C|/Gamma of the pseudo-covariance, one panel each. Must &quot;</span>
             <span class="s2">&quot;lie in [0,1]: 0 is circular, 1 is a degenerate (real) variable.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--phase&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">0.0</span><span class="p">,</span> <span class="mf">0.3333</span><span class="p">,</span> <span class="mf">0.3333</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Arguments of the pseudo-covariance, in units of pi, one per rho.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--gamma&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;The covariance, shared by every panel. It is what the panels hold &quot;</span>
             <span class="s2">&quot;fixed, so that only the pseudo-covariance distinguishes them.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">600</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Observations drawn per panel.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--probability&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Probability mass enclosed by the drawn concentration curves.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/complex_circularity&quot;</span><span class="p">,</span>
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
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;random seed generation base seed&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">)</span> <span class="o">!=</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">phase</span><span class="p">):</span>
        <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span><span class="s2">&quot;--rho and --phase must have the same length&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">gamma</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">gamma</span>
    <span class="n">phases</span> <span class="o">=</span> <span class="p">[</span><span class="n">p</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">pi</span> <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">phase</span><span class="p">]</span>
    <span class="n">pseudos</span> <span class="o">=</span> <span class="p">[</span><span class="n">rho</span> <span class="o">*</span> <span class="n">gamma</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="mi">1</span><span class="n">j</span> <span class="o">*</span> <span class="n">phase</span><span class="p">)</span> <span class="k">for</span> <span class="n">rho</span><span class="p">,</span> <span class="n">phase</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="n">phases</span><span class="p">)]</span>

    <span class="k">for</span> <span class="n">rho</span><span class="p">,</span> <span class="n">pseudo</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="n">pseudos</span><span class="p">):</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="n">is_admissible</span><span class="p">(</span><span class="n">gamma</span><span class="p">,</span> <span class="n">pseudo</span><span class="p">):</span>
            <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;|C|/Gamma = </span><span class="si">{</span><span class="n">rho</span><span class="si">}</span><span class="s2"> violates Picinbono&#39;s condition |C| &lt;= Gamma&quot;</span>
            <span class="p">)</span>

    <span class="c1"># The reference a reader would draw from the covariance alone: the circular</span>
    <span class="c1"># variable of the same Gamma, whose real covariance is (Gamma/2) I.</span>
    <span class="n">reference</span> <span class="o">=</span> <span class="n">concentration_ellipse</span><span class="p">(</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">gamma</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">eye</span><span class="p">(</span><span class="mi">2</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">probability</span><span class="p">)</span>

    <span class="n">samples</span><span class="p">,</span> <span class="n">ellipses</span><span class="p">,</span> <span class="n">empirical</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[],</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">offset</span><span class="p">,</span> <span class="n">pseudo</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">pseudos</span><span class="p">):</span>
        <span class="n">covariance</span> <span class="o">=</span> <span class="n">real_covariance</span><span class="p">(</span><span class="n">gamma</span><span class="p">,</span> <span class="n">pseudo</span><span class="p">)</span>
        <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="n">offset</span><span class="p">)</span>
        <span class="n">values</span><span class="p">,</span> <span class="n">vectors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">(</span><span class="n">covariance</span><span class="p">)</span>
        <span class="n">factor</span> <span class="o">=</span> <span class="n">vectors</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">maximum</span><span class="p">(</span><span class="n">values</span><span class="p">,</span> <span class="mf">0.0</span><span class="p">)))</span>
        <span class="n">data</span> <span class="o">=</span> <span class="p">(</span><span class="n">factor</span> <span class="o">@</span> <span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="mi">2</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">)))</span><span class="o">.</span><span class="n">T</span>
        <span class="n">z</span> <span class="o">=</span> <span class="n">data</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">+</span> <span class="mi">1</span><span class="n">j</span> <span class="o">*</span> <span class="n">data</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span>

        <span class="n">samples</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
        <span class="n">ellipses</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">concentration_ellipse</span><span class="p">(</span><span class="n">covariance</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">probability</span><span class="p">))</span>
        <span class="n">empirical</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">z</span><span class="p">)</span> <span class="o">**</span> <span class="mi">2</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">z</span> <span class="o">*</span> <span class="n">z</span><span class="p">)))</span>

    <span class="c1"># A 2-column grid rather than a single row: the dissertation&#39;s text block</span>
    <span class="c1"># is narrow, and four panels side by side overflow it.</span>
    <span class="n">n_panels</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">pseudos</span><span class="p">)</span>
    <span class="n">n_cols</span> <span class="o">=</span> <span class="nb">min</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">n_panels</span><span class="p">)</span>
    <span class="n">n_rows</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">ceil</span><span class="p">(</span><span class="n">n_panels</span> <span class="o">/</span> <span class="n">n_cols</span><span class="p">))</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span>
        <span class="n">n_rows</span><span class="p">,</span> <span class="n">n_cols</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="n">n_cols</span><span class="p">,</span> <span class="mf">3.4</span> <span class="o">*</span> <span class="n">n_rows</span><span class="p">),</span>
        <span class="n">sharex</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">axes</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">atleast_1d</span><span class="p">(</span><span class="n">axes</span><span class="p">)</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>

    <span class="n">limit</span> <span class="o">=</span> <span class="mf">1.15</span> <span class="o">*</span> <span class="nb">max</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">quantile</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">abs</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="mf">0.999</span><span class="p">)</span> <span class="k">for</span> <span class="n">data</span> <span class="ow">in</span> <span class="n">samples</span><span class="p">)</span>

    <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">rho</span><span class="p">,</span> <span class="n">phase</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">ellipse</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span>
        <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="n">phases</span><span class="p">,</span> <span class="n">samples</span><span class="p">,</span> <span class="n">ellipses</span><span class="p">)</span>
    <span class="p">):</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span>
            <span class="n">data</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">],</span> <span class="n">data</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">],</span>
            <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">s</span><span class="o">=</span><span class="mi">6</span><span class="p">,</span> <span class="n">facecolors</span><span class="o">=</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="n">edgecolors</span><span class="o">=</span><span class="s2">&quot;C0&quot;</span><span class="p">,</span> <span class="n">linewidths</span><span class="o">=</span><span class="mf">0.4</span><span class="p">,</span>
            <span class="n">zorder</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">reference</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">reference</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span>
            <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C7&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="s2">&quot;prédit par $</span><span class="se">\\</span><span class="s2">Gamma$ seule&quot;</span> <span class="k">if</span> <span class="n">index</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">ellipse</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">ellipse</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span>
            <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C1&quot;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;-&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.4</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="s2">&quot;concentration réelle&quot;</span> <span class="k">if</span> <span class="n">index</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_aspect</span><span class="p">(</span><span class="s2">&quot;equal&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="o">-</span><span class="n">limit</span><span class="p">,</span> <span class="n">limit</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">index</span> <span class="o">//</span> <span class="n">n_cols</span> <span class="o">==</span> <span class="n">n_rows</span> <span class="o">-</span> <span class="mi">1</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\mathrm</span><span class="si">{Re}</span><span class="s2">\,z$&quot;</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">index</span> <span class="o">%</span> <span class="n">n_cols</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\mathrm</span><span class="si">{Im}</span><span class="s2">\,z$&quot;</span><span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">panel_title</span><span class="p">(</span><span class="n">rho</span><span class="p">,</span> <span class="n">phase</span><span class="p">))</span>

    <span class="k">for</span> <span class="n">ax</span> <span class="ow">in</span> <span class="n">axes</span><span class="p">[</span><span class="n">n_panels</span><span class="p">:]:</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_visible</span><span class="p">(</span><span class="kc">False</span><span class="p">)</span>

    <span class="c1"># Legend attached to an axis, not to the figure: matplot2tikz exports the</span>
    <span class="c1"># former and silently drops the latter.</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;lower left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="mf">1.14</span><span class="p">),</span>
        <span class="n">ncol</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">9</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Gamma = </span><span class="si">{</span><span class="n">gamma</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, shared by every panel; N = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">rho</span><span class="p">,</span> <span class="n">phase</span><span class="p">,</span> <span class="p">(</span><span class="n">gamma_hat</span><span class="p">,</span> <span class="n">pseudo_hat</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">,</span> <span class="n">phases</span><span class="p">,</span> <span class="n">empirical</span><span class="p">):</span>
        <span class="nb">print</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;  |C|/Gamma=</span><span class="si">{</span><span class="n">rho</span><span class="si">:</span><span class="s2">&lt;4g</span><span class="si">}</span><span class="s2"> arg C=</span><span class="si">{</span><span class="n">phase</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="si">:</span><span class="s2">&lt;7.4g</span><span class="si">}</span><span class="s2">pi  &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;Gamma_hat=</span><span class="si">{</span><span class="n">gamma_hat</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">  &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;C_hat=</span><span class="si">{</span><span class="nb">abs</span><span class="p">(</span><span class="n">pseudo_hat</span><span class="p">)</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2"> exp(j</span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">angle</span><span class="p">(</span><span class="n">pseudo_hat</span><span class="p">)</span><span class="w"> </span><span class="o">/</span><span class="w"> </span><span class="n">np</span><span class="o">.</span><span class="n">pi</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">pi)&quot;</span>
        <span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">gamma</span><span class="o">=</span><span class="n">gamma</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span>
        <span class="n">probability</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">probability</span><span class="p">,</span>
        <span class="n">rho</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">rho</span><span class="p">),</span> <span class="n">phase</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">phases</span><span class="p">),</span>
        <span class="n">samples</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">samples</span><span class="p">),</span>
        <span class="n">ellipses</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span><span class="n">ellipses</span><span class="p">),</span>
        <span class="n">reference</span><span class="o">=</span><span class="n">reference</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;circularity.tex&quot;</span><span class="p">)</span>
        <span class="n">save</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved circularity panels in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--rho</span><span class="param-type">float</span><span class="param-default">default <b>[0.0, 0.5, 0.5, 0.9]</b></span>
</div>
<p class="param-help">Moduli |C|/Gamma of the pseudo-covariance, one panel each. Must lie in [0,1]: 0 is circular, 1 is a degenerate (real) variable.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--phase</span><span class="param-type">float</span><span class="param-default">default <b>[0.0, 0.0, 0.3333, 0.3333]</b></span>
</div>
<p class="param-help">Arguments of the pseudo-covariance, in units of pi, one per rho.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--gamma</span><span class="param-type">float</span><span class="param-default">default <b>1.0</b></span>
</div>
<p class="param-help">The covariance, shared by every panel. It is what the panels hold fixed, so that only the pseudo-covariance distinguishes them.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>600</b></span>
</div>
<p class="param-help">Observations drawn per panel.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--probability</span><span class="param-type">float</span><span class="param-default">default <b>0.9</b></span>
</div>
<p class="param-help">Probability mass enclosed by the drawn concentration curves.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/complex_circularity</b></span>
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
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">random seed generation base seed</p>
</div>
</div>

## Results

<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_complex_circularity.json" data-title="context_complex_circularity"></div>
</div>

## Config

`1-context/experiments/context_complex_circularity.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
