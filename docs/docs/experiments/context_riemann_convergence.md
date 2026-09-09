<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/1-context/">1 · Context</a>
<span class="sep">/</span>
<span class="here">context_riemann_convergence</span>
</nav>

# context_riemann_convergence

Fixed point, Riemannian descent and projected Euclidean descent on Tyler's cost

**Tags:** `context`  `riemann`  `optimisation`  `robust`

## Run

```sh
uv run python 1-context/riemann_convergence/main.py
```

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--n_features 10</code><br>
  <code>--n_samples 100</code><br>
  <code>--dof 3.0</code><br>
  <code>--condition 100.0</code><br>
  <code>--iter_max 150</code><br>
  <code>--tol 1e-12</code><br>
  <code>--floor 1e-14</code><br>
  <code>--axis_width 0.45\textwidth</code><br>
  <code>--axis_height 4.6cm</code><br>
  <code>--backend numpy</code><br>
  <code>--seed 42</code><br>
  <span class="mn-date">590ac0f · 2026-08-19</span>
</span>


<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/1-context/riemann_convergence/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">240 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">1-context/riemann_convergence/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Three ways of minimising the same criterion</span>
<span class="c1">#</span>
<span class="c1"># Tyler&#39;s cost is minimised from the same starting point by three algorithms:</span>
<span class="c1"># the fixed-point iteration, the Riemannian gradient descent with a</span>
<span class="c1"># backtracking line search, and a Euclidean gradient descent that has to</span>
<span class="c1"># project its iterates back onto the cone. The left panel counts iterations,</span>
<span class="c1"># the right one seconds.</span>
<span class="c1">#</span>
<span class="c1"># The first two curves are nearly on top of each other, which is the point of</span>
<span class="c1"># the chapter: the fixed point *is* a Riemannian gradient step of unit length,</span>
<span class="c1"># so it cannot do better than the descent it turns out to be an instance of.</span>
<span class="c1"># The Euclidean method minimises the same function and reaches the same</span>
<span class="c1"># minimiser, only much more slowly: its gradient ignores the geometry that</span>
<span class="c1"># the congruence Sigma . Sigma restores.</span>
<span class="c1">#</span>
<span class="c1"># The three minimisers come from hdrlib.core.estimation and are written out</span>
<span class="c1"># there — gradient, step rule and retraction — rather than delegated to a</span>
<span class="c1"># manifold optimisation library.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.estimation</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">minimize_tyler_fixed_point</span><span class="p">,</span>
    <span class="n">minimize_tyler_riemannian</span><span class="p">,</span>
    <span class="n">minimize_tyler_euclidean</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.elliptical</span><span class="w"> </span><span class="kn">import</span> <span class="n">StudentTDistribution</span><span class="p">,</span> <span class="n">sample_elliptical</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.manifolds</span><span class="w"> </span><span class="kn">import</span> <span class="n">HermitianPositiveDefinite</span>


<span class="k">def</span><span class="w"> </span><span class="nf">scatter_matrix</span><span class="p">(</span><span class="n">n_features</span><span class="p">,</span> <span class="n">condition</span><span class="p">,</span> <span class="n">rng</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Ill-conditioned scatter matrix of unit determinant.&quot;&quot;&quot;</span>
    <span class="n">eigenvalues</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">logspace</span><span class="p">(</span>
        <span class="o">-</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">condition</span><span class="p">),</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">condition</span><span class="p">),</span> <span class="n">n_features</span>
    <span class="p">)</span>
    <span class="n">rotation</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">qr</span><span class="p">(</span><span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">n_features</span><span class="p">,</span> <span class="n">n_features</span><span class="p">)))[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">scatter</span> <span class="o">=</span> <span class="n">rotation</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">eigenvalues</span><span class="p">)</span> <span class="o">@</span> <span class="n">rotation</span><span class="o">.</span><span class="n">T</span>
    <span class="k">return</span> <span class="n">scatter</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">det</span><span class="p">(</span><span class="n">scatter</span><span class="p">)</span> <span class="o">**</span> <span class="p">(</span><span class="mf">1.0</span> <span class="o">/</span> <span class="n">n_features</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Fixed point, Riemannian descent and projected Euclidean descent on &quot;</span>
        <span class="s2">&quot;Tyler&#39;s cost.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Dimension of the observations.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of observations used by the three algorithms.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--dof&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">3.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Degrees of freedom of the Student data.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--condition&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">100.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Condition number of the true scatter matrix. The larger it is, &quot;</span>
             <span class="s2">&quot;the further the identity — the common starting point — sits &quot;</span>
             <span class="s2">&quot;from the solution.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--iter_max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">150</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Maximum number of iterations granted to each algorithm.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-12</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Stopping tolerance on the Riemannian gradient norm. Deliberately &quot;</span>
             <span class="s2">&quot;unreachable, so that every algorithm spends its whole budget and &quot;</span>
             <span class="s2">&quot;the curves can be compared over their full length.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--floor&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-14</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Smallest optimality gap shown; below it the cost is dominated by &quot;</span>
             <span class="s2">&quot;rounding rather than by the algorithm.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/riemann_convergence&quot;</span><span class="p">,</span>
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
    <span class="n">distribution</span> <span class="o">=</span> <span class="n">StudentTDistribution</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">sample_elliptical</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span>
        <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">d</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
        <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">scatter</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
        <span class="n">distribution</span><span class="p">,</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="c1"># Same budget, same starting point — the identity — for the three.</span>
    <span class="n">solutions</span><span class="p">,</span> <span class="n">histories</span> <span class="o">=</span> <span class="p">{},</span> <span class="p">{}</span>
    <span class="n">runs</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;point fixe&quot;</span><span class="p">:</span> <span class="n">minimize_tyler_fixed_point</span><span class="p">,</span>
        <span class="s2">&quot;gradient riemannien&quot;</span><span class="p">:</span> <span class="n">minimize_tyler_riemannian</span><span class="p">,</span>
        <span class="s2">&quot;gradient euclidien projeté&quot;</span><span class="p">:</span> <span class="n">minimize_tyler_euclidean</span><span class="p">,</span>
    <span class="p">}</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">minimize</span> <span class="ow">in</span> <span class="n">runs</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">solutions</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">histories</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="n">minimize</span><span class="p">(</span>
            <span class="n">data</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">tol</span><span class="p">,</span> <span class="n">backend_name</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span>
        <span class="p">)</span>

    <span class="c1"># The reference value is the smallest cost any of the three reached: the</span>
    <span class="c1"># criterion has a single minimum, so this is the common target and what</span>
    <span class="c1"># makes the three gaps comparable.</span>
    <span class="n">optimum</span> <span class="o">=</span> <span class="nb">min</span><span class="p">(</span><span class="nb">min</span><span class="p">(</span><span class="n">history</span><span class="p">[</span><span class="s2">&quot;cost&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">history</span> <span class="ow">in</span> <span class="n">histories</span><span class="o">.</span><span class="n">values</span><span class="p">())</span>
    <span class="c1"># Anything below this is the noise of double precision on the cost, not</span>
    <span class="c1"># convergence, so the curves are clipped there rather than plunging to</span>
    <span class="c1"># whatever rounding happened to produce.</span>
    <span class="n">floor</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">floor</span>

    <span class="n">colors</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;point fixe&quot;</span><span class="p">:</span> <span class="s2">&quot;C1&quot;</span><span class="p">,</span>
        <span class="s2">&quot;gradient riemannien&quot;</span><span class="p">:</span> <span class="s2">&quot;C2&quot;</span><span class="p">,</span>
        <span class="s2">&quot;gradient euclidien projeté&quot;</span><span class="p">:</span> <span class="s2">&quot;C3&quot;</span><span class="p">,</span>
    <span class="p">}</span>
    <span class="n">markers</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;point fixe&quot;</span><span class="p">:</span> <span class="s2">&quot;o&quot;</span><span class="p">,</span>
        <span class="s2">&quot;gradient riemannien&quot;</span><span class="p">:</span> <span class="s2">&quot;s&quot;</span><span class="p">,</span>
        <span class="s2">&quot;gradient euclidien projeté&quot;</span><span class="p">:</span> <span class="s2">&quot;^&quot;</span><span class="p">,</span>
    <span class="p">}</span>

    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">),</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">history</span> <span class="ow">in</span> <span class="n">histories</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">gap</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">maximum</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">history</span><span class="p">[</span><span class="s2">&quot;cost&quot;</span><span class="p">])</span> <span class="o">-</span> <span class="n">optimum</span><span class="p">,</span> <span class="n">floor</span><span class="p">)</span>
        <span class="c1"># The label is attached to the left panel only: matplot2tikz collects</span>
        <span class="c1"># the labelled curves of every axis into the single exported legend,</span>
        <span class="c1"># so labelling both panels lists each algorithm twice.</span>
        <span class="k">for</span> <span class="n">column</span><span class="p">,</span> <span class="n">abscissa</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span>
            <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">gap</span><span class="p">)),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">history</span><span class="p">[</span><span class="s2">&quot;time&quot;</span><span class="p">])]</span>
        <span class="p">):</span>
            <span class="n">axes</span><span class="p">[</span><span class="n">column</span><span class="p">]</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
                <span class="n">abscissa</span><span class="p">,</span> <span class="n">gap</span><span class="p">,</span>
                <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.4</span><span class="p">,</span>
                <span class="n">marker</span><span class="o">=</span><span class="n">markers</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">markevery</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
                <span class="n">label</span><span class="o">=</span><span class="n">name</span> <span class="k">if</span> <span class="n">column</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
            <span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;itération&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$L - L^{\star}$&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;temps (s)&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">ax</span> <span class="ow">in</span> <span class="n">axes</span><span class="p">:</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylim</span><span class="p">(</span><span class="n">bottom</span><span class="o">=</span><span class="mf">0.5</span> <span class="o">*</span> <span class="n">floor</span><span class="p">)</span>
    <span class="c1"># Legend below the panels: exported at this size, an inner one covers the</span>
    <span class="c1"># tick labels of the very axis it sits in.</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;upper left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.45</span><span class="p">),</span> <span class="n">ncol</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
        <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;d = </span><span class="si">{</span><span class="n">d</span><span class="si">}</span><span class="s2">, N = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="si">}</span><span class="s2">, Student nu = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;condition </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">history</span> <span class="ow">in</span> <span class="n">histories</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">distance</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span>
            <span class="n">manifold</span><span class="o">.</span><span class="n">dist</span><span class="p">(</span>
                <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">solutions</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
                <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">solutions</span><span class="p">[</span><span class="s2">&quot;point fixe&quot;</span><span class="p">],</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">),</span>
            <span class="p">)</span>
        <span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;  </span><span class="si">{</span><span class="n">name</span><span class="si">:</span><span class="s2">27</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">history</span><span class="p">[</span><span class="s1">&#39;cost&#39;</span><span class="p">])</span><span class="w"> </span><span class="o">-</span><span class="w"> </span><span class="mi">1</span><span class="si">:</span><span class="s2">4d</span><span class="si">}</span><span class="s2"> iterations   &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;final gap </span><span class="si">{</span><span class="n">history</span><span class="p">[</span><span class="s1">&#39;cost&#39;</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="w"> </span><span class="o">-</span><span class="w"> </span><span class="n">optimum</span><span class="si">:</span><span class="s2">.2e</span><span class="si">}</span><span class="s2">   &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;gradient </span><span class="si">{</span><span class="n">history</span><span class="p">[</span><span class="s1">&#39;gradient_norm&#39;</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">:</span><span class="s2">.2e</span><span class="si">}</span><span class="s2">   &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">history</span><span class="p">[</span><span class="s1">&#39;time&#39;</span><span class="p">][</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2"> s   &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;distance to the fixed point </span><span class="si">{</span><span class="n">distance</span><span class="si">:</span><span class="s2">.2e</span><span class="si">}</span><span class="s2">&quot;</span>
        <span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_features</span><span class="o">=</span><span class="n">d</span><span class="p">,</span> <span class="n">n_samples</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span> <span class="n">dof</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">dof</span><span class="p">,</span>
        <span class="n">condition</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">condition</span><span class="p">,</span> <span class="n">iter_max</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">iter_max</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">tol</span><span class="p">,</span>
        <span class="n">scatter</span><span class="o">=</span><span class="n">scatter</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">to_numpy</span><span class="p">(</span><span class="n">data</span><span class="p">),</span> <span class="n">optimum</span><span class="o">=</span><span class="n">optimum</span><span class="p">,</span>
        <span class="o">**</span><span class="p">{</span>
            <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">key</span><span class="si">}</span><span class="s2">_</span><span class="si">{</span><span class="n">name</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;é&#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;e&#39;</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">values</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">history</span> <span class="ow">in</span> <span class="n">histories</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
            <span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">values</span> <span class="ow">in</span> <span class="n">history</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
            <span class="k">if</span> <span class="n">key</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">&quot;cost&quot;</span><span class="p">,</span> <span class="s2">&quot;gradient_norm&quot;</span><span class="p">,</span> <span class="s2">&quot;time&quot;</span><span class="p">)</span>
        <span class="p">},</span>
        <span class="o">**</span><span class="p">{</span>
            <span class="sa">f</span><span class="s2">&quot;solution_</span><span class="si">{</span><span class="n">name</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39; &#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;_&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;é&#39;</span><span class="p">,</span><span class="w"> </span><span class="s1">&#39;e&#39;</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span>
            <span class="n">to_numpy</span><span class="p">(</span><span class="n">solution</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">solution</span> <span class="ow">in</span> <span class="n">solutions</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
        <span class="p">},</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;convergence.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved convergence curves in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_features</span><span class="param-type">int</span><span class="param-default">default <b>10</b></span>
</div>
<p class="param-help">Dimension of the observations.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>100</b></span>
</div>
<p class="param-help">Number of observations used by the three algorithms.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--dof</span><span class="param-type">float</span><span class="param-default">default <b>3.0</b></span>
</div>
<p class="param-help">Degrees of freedom of the Student data.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--condition</span><span class="param-type">float</span><span class="param-default">default <b>100.0</b></span>
</div>
<p class="param-help">Condition number of the true scatter matrix. The larger it is, the further the identity — the common starting point — sits from the solution.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--iter_max</span><span class="param-type">int</span><span class="param-default">default <b>150</b></span>
</div>
<p class="param-help">Maximum number of iterations granted to each algorithm.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-12</b></span>
</div>
<p class="param-help">Stopping tolerance on the Riemannian gradient norm. Deliberately unreachable, so that every algorithm spends its whole budget and the curves can be compared over their full length.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--floor</span><span class="param-type">float</span><span class="param-default">default <b>1e-14</b></span>
</div>
<p class="param-help">Smallest optimality gap shown; below it the cost is dominated by rounding rather than by the algorithm.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/riemann_convergence</b></span>
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

<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_riemann_convergence.json" data-title="context_riemann_convergence"></div>
</div>

## Config

`1-context/experiments/context_riemann_convergence.yaml`

<a class="back-link" href="../../chapters/1-context/">← All experiments in 1 · Context</a>
