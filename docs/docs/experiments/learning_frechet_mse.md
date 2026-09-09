<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/3-learning/">3 · Learning</a>
<span class="sep">/</span>
<span class="here">learning_frechet_mse</span>
</nav>

# learning_frechet_mse

MSE of the Fréchet mean of a set of covariances, against the number of samples and against the number of matrices — SCM, Ledoit-Wolf, OAS, non-linear shrinkage and the RMT correction

**Tags:** `learning`  `random-matrix-theory`  `frechet-mean`  `monte-carlo`

## Run

```sh
uv run python 3-learning/frechet_mse/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/3-learning/frechet_mse/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">335 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">3-learning/frechet_mse/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># How well is the Fréchet mean of a set of covariances estimated?</span>
<span class="c1">#</span>
<span class="c1"># The mean of ch:learning is the one every nearest-centroid classifier and</span>
<span class="c1"># every Riemannian k-means computes. In practice the true covariances are not</span>
<span class="c1"># available, only their sample estimates, and the regime is the one of</span>
<span class="c1"># subsec:learning-rmt: p and N comparable. The plain Fréchet mean of the SCMs</span>
<span class="c1"># is then biased, and the question is by how much, and what the correction</span>
<span class="c1"># buys.</span>
<span class="c1">#</span>
<span class="c1"># Two panels, and they answer two different questions:</span>
<span class="c1">#</span>
<span class="c1">#   * against the number of samples N: the gap closes as N grows, which is the</span>
<span class="c1">#     signature of a bias of regime and not of a variance;</span>
<span class="c1">#   * against the number of matrices K: the gap *widens*. Averaging more</span>
<span class="c1">#     matrices reduces the variance of the estimate but not its bias, which is</span>
<span class="c1">#     common to every SCM; past some K the bias is all that is left and it is</span>
<span class="c1">#     the only thing separating the methods. This is the panel that carries</span>
<span class="c1">#     the argument, and it is the one the intuition gets wrong.</span>
<span class="c1">#</span>
<span class="c1"># Five estimators, as in the paper: the plain Fréchet mean of the SCMs, of the</span>
<span class="c1"># linearly shrunk covariances (Ledoit-Wolf, OAS), of the non-linearly shrunk</span>
<span class="c1"># ones, and the corrected mean. Note where the correction is applied: the</span>
<span class="c1"># shrinkage methods regularise each covariance *before* averaging, the RMT</span>
<span class="c1"># method corrects the *distance* the average minimises. It is not the same</span>
<span class="c1"># gesture, and the figures are what separates them.</span>
<span class="c1">#</span>
<span class="c1"># Backend-free through hdrlib.learning.rmt, whose port of the reference</span>
<span class="c1"># implementation is checked term by term against the published code by</span>
<span class="c1"># validate_against_paper.py. Needs float64 — see rmt.require_double.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">functools</span><span class="w"> </span><span class="kn">import</span> <span class="n">partial</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">multiprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Pool</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">matplotlib.ticker</span><span class="w"> </span><span class="kn">import</span> <span class="n">NullFormatter</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span><span class="p">,</span> <span class="n">to_numpy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="n">add_mc_base_args</span><span class="p">,</span> <span class="n">init_logging</span><span class="p">,</span> <span class="n">make_mc_parser</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.learning</span><span class="w"> </span><span class="kn">import</span> <span class="n">rmt</span>


<span class="n">METHODS</span> <span class="o">=</span> <span class="p">(</span><span class="s2">&quot;SCM&quot;</span><span class="p">,</span> <span class="s2">&quot;LW&quot;</span><span class="p">,</span> <span class="s2">&quot;OAS&quot;</span><span class="p">,</span> <span class="s2">&quot;LW-NL&quot;</span><span class="p">,</span> <span class="s2">&quot;RMT&quot;</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">random_spd</span><span class="p">(</span><span class="n">rng</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">condition_number</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;SPD matrix with a prescribed condition number.</span>

<span class="sd">    Random orthogonal basis, eigenvalues uniform between the two extremes,</span>
<span class="sd">    which are pinned so that the condition number is exactly the one asked</span>
<span class="sd">    for. Same construction as the reference implementation.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">basis</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">qr</span><span class="p">(</span><span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">n_features</span><span class="p">,</span> <span class="n">n_features</span><span class="p">)))[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">low</span><span class="p">,</span> <span class="n">high</span> <span class="o">=</span> <span class="mi">1</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">condition_number</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="n">condition_number</span><span class="p">)</span>
    <span class="n">eigenvalues</span> <span class="o">=</span> <span class="n">rng</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="n">low</span><span class="p">,</span> <span class="n">high</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="n">n_features</span><span class="p">)</span>
    <span class="n">eigenvalues</span><span class="p">[</span><span class="o">-</span><span class="mi">2</span><span class="p">],</span> <span class="n">eigenvalues</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="n">low</span><span class="p">,</span> <span class="n">high</span>
    <span class="k">return</span> <span class="n">basis</span> <span class="o">@</span> <span class="n">np</span><span class="o">.</span><span class="n">diag</span><span class="p">(</span><span class="n">eigenvalues</span><span class="p">)</span> <span class="o">@</span> <span class="n">basis</span><span class="o">.</span><span class="n">T</span>


<span class="k">def</span><span class="w"> </span><span class="nf">covariances_around</span><span class="p">(</span><span class="n">rng</span><span class="p">,</span> <span class="n">centre</span><span class="p">,</span> <span class="n">n_matrices</span><span class="p">,</span> <span class="n">scale</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Covariances whose Fréchet mean is exactly ``centre``.</span>

<span class="sd">    Tangent vectors at the centre are drawn, then *centred* before being</span>
<span class="sd">    pushed onto the manifold: their arithmetic mean is zero, so the centre is</span>
<span class="sd">    the exact Fréchet mean of the resulting set rather than its limit for</span>
<span class="sd">    large ``n_matrices``. Without that subtraction the MSE would measure the</span>
<span class="sd">    sampling of the cloud on top of the estimation error, and the two would be</span>
<span class="sd">    impossible to tell apart at small ``n_matrices`` — which is precisely the</span>
<span class="sd">    regime the second panel is about.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">n_features</span> <span class="o">=</span> <span class="n">centre</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">tangents</span> <span class="o">=</span> <span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">n_matrices</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">n_features</span><span class="p">))</span> <span class="o">*</span> <span class="n">scale</span>
    <span class="n">tangents</span> <span class="o">=</span> <span class="p">(</span><span class="n">tangents</span> <span class="o">+</span> <span class="n">tangents</span><span class="o">.</span><span class="n">swapaxes</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">))</span> <span class="o">/</span> <span class="mi">2</span>
    <span class="n">tangents</span> <span class="o">-=</span> <span class="n">tangents</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="n">values</span><span class="p">,</span> <span class="n">vectors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigh</span><span class="p">(</span><span class="n">tangents</span><span class="p">)</span>
    <span class="n">factor</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">cholesky</span><span class="p">(</span><span class="n">centre</span><span class="p">)</span>
    <span class="n">exponential</span> <span class="o">=</span> <span class="n">vectors</span> <span class="o">@</span> <span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">values</span><span class="p">)[</span><span class="o">...</span><span class="p">,</span> <span class="kc">None</span><span class="p">]</span> <span class="o">*</span> <span class="n">vectors</span><span class="o">.</span><span class="n">swapaxes</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">)</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="n">factor</span> <span class="o">@</span> <span class="n">exponential</span> <span class="o">@</span> <span class="n">factor</span><span class="o">.</span><span class="n">T</span>


<span class="k">def</span><span class="w"> </span><span class="nf">gaussian_data</span><span class="p">(</span><span class="n">rng</span><span class="p">,</span> <span class="n">covariances</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Centred Gaussian samples, one block per covariance.&quot;&quot;&quot;</span>
    <span class="n">n_matrices</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">covariances</span><span class="o">.</span><span class="n">shape</span>
    <span class="n">factors</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">cholesky</span><span class="p">(</span><span class="n">covariances</span><span class="p">)</span>
    <span class="n">white</span> <span class="o">=</span> <span class="n">rng</span><span class="o">.</span><span class="n">standard_normal</span><span class="p">((</span><span class="n">n_matrices</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">,</span> <span class="n">n_features</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">white</span> <span class="o">@</span> <span class="n">factors</span><span class="o">.</span><span class="n">swapaxes</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">squared_distance</span><span class="p">(</span><span class="n">reference</span><span class="p">,</span> <span class="n">estimate</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Squared affine-invariant distance — the error the figures report.&quot;&quot;&quot;</span>
    <span class="n">inverse</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">inv</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">cholesky</span><span class="p">(</span><span class="n">reference</span><span class="p">))</span>
    <span class="n">logarithms</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span>
        <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigvalsh</span><span class="p">(</span><span class="n">inverse</span> <span class="o">@</span> <span class="n">estimate</span> <span class="o">@</span> <span class="n">inverse</span><span class="o">.</span><span class="n">swapaxes</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="o">-</span><span class="mi">2</span><span class="p">))</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">logarithms</span> <span class="o">@</span> <span class="n">logarithms</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">estimate_all</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">backend</span><span class="p">,</span> <span class="n">max_iterations</span><span class="p">,</span> <span class="n">tol</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;The five estimates of the Fréchet mean, from one set of data blocks.&quot;&quot;&quot;</span>
    <span class="n">device_data</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">estimates</span> <span class="o">=</span> <span class="p">{}</span>

    <span class="c1"># The four two-step methods: regularise each covariance, then average.</span>
    <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">estimator</span> <span class="ow">in</span> <span class="p">(</span>
        <span class="p">(</span><span class="s2">&quot;SCM&quot;</span><span class="p">,</span> <span class="n">rmt</span><span class="o">.</span><span class="n">scm</span><span class="p">),</span>
        <span class="p">(</span><span class="s2">&quot;LW&quot;</span><span class="p">,</span> <span class="n">rmt</span><span class="o">.</span><span class="n">ledoit_wolf_linear</span><span class="p">),</span>
        <span class="p">(</span><span class="s2">&quot;OAS&quot;</span><span class="p">,</span> <span class="n">rmt</span><span class="o">.</span><span class="n">oas</span><span class="p">),</span>
        <span class="p">(</span><span class="s2">&quot;LW-NL&quot;</span><span class="p">,</span> <span class="k">lambda</span> <span class="n">d</span><span class="p">,</span> <span class="n">b</span><span class="p">:</span> <span class="n">rmt</span><span class="o">.</span><span class="n">analytical_shrinkage</span><span class="p">(</span><span class="n">d</span><span class="p">,</span> <span class="n">b</span><span class="p">,</span> <span class="n">shrink</span><span class="o">=</span><span class="mi">0</span><span class="p">)),</span>
    <span class="p">):</span>
        <span class="n">covariances</span> <span class="o">=</span> <span class="n">estimator</span><span class="p">(</span><span class="n">device_data</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
        <span class="n">mean</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">rmt</span><span class="o">.</span><span class="n">frechet_mean_cholesky</span><span class="p">(</span>
            <span class="n">covariances</span><span class="p">,</span> <span class="n">max_iterations</span><span class="o">=</span><span class="n">max_iterations</span><span class="p">,</span> <span class="n">backend</span><span class="o">=</span><span class="n">backend</span>
        <span class="p">)</span>
        <span class="n">estimates</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">mean</span><span class="p">)</span>

    <span class="c1"># The one-step method: correct the distance the average minimises.</span>
    <span class="n">mean</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">rmt</span><span class="o">.</span><span class="n">rmt_frechet_mean</span><span class="p">(</span>
        <span class="n">device_data</span><span class="p">,</span> <span class="n">max_iterations</span><span class="o">=</span><span class="n">max_iterations</span><span class="p">,</span> <span class="n">tol</span><span class="o">=</span><span class="n">tol</span><span class="p">,</span> <span class="n">backend</span><span class="o">=</span><span class="n">backend</span>
    <span class="p">)</span>
    <span class="n">estimates</span><span class="p">[</span><span class="s2">&quot;RMT&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">to_numpy</span><span class="p">(</span><span class="n">mean</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">estimates</span>


<span class="k">def</span><span class="w"> </span><span class="nf">one_trial</span><span class="p">(</span><span class="n">job</span><span class="p">,</span> <span class="n">rng_seed</span><span class="p">,</span> <span class="n">centre</span><span class="p">,</span> <span class="n">n_axis</span><span class="p">,</span> <span class="n">args</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;One Monte-Carlo trial: draw a cloud, estimate it five ways, score them.</span>

<span class="sd">    Seeded per ``(axis index, trial)`` so that a run is reproducible and a</span>
<span class="sd">    single point can be re-drawn without replaying the whole sweep — which</span>
<span class="sd">    matters here, because the expensive points are at the ends of the axes.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">index</span><span class="p">,</span> <span class="n">trial</span><span class="p">,</span> <span class="n">n_matrices</span><span class="p">,</span> <span class="n">n_samples</span> <span class="o">=</span> <span class="n">job</span>
    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">([</span><span class="n">rng_seed</span><span class="p">,</span> <span class="n">index</span><span class="p">,</span> <span class="n">trial</span><span class="p">,</span> <span class="n">n_axis</span><span class="p">])</span>
    <span class="n">covariances</span> <span class="o">=</span> <span class="n">covariances_around</span><span class="p">(</span><span class="n">rng</span><span class="p">,</span> <span class="n">centre</span><span class="p">,</span> <span class="n">n_matrices</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">scale</span><span class="p">)</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">gaussian_data</span><span class="p">(</span><span class="n">rng</span><span class="p">,</span> <span class="n">covariances</span><span class="p">,</span> <span class="n">n_samples</span><span class="p">)</span>
    <span class="n">estimates</span> <span class="o">=</span> <span class="n">estimate_all</span><span class="p">(</span>
        <span class="n">data</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_iterations_max</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">tol</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="n">index</span><span class="p">,</span> <span class="n">trial</span><span class="p">,</span> <span class="p">{</span>
        <span class="n">name</span><span class="p">:</span> <span class="n">squared_distance</span><span class="p">(</span><span class="n">centre</span><span class="p">,</span> <span class="n">estimate</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">estimate</span> <span class="ow">in</span> <span class="n">estimates</span><span class="o">.</span><span class="n">items</span><span class="p">()</span>
    <span class="p">}</span>


<span class="k">def</span><span class="w"> </span><span class="nf">sweep</span><span class="p">(</span><span class="n">rng_seed</span><span class="p">,</span> <span class="n">centre</span><span class="p">,</span> <span class="n">axis_name</span><span class="p">,</span> <span class="n">axis_values</span><span class="p">,</span> <span class="n">fixed</span><span class="p">,</span> <span class="n">args</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;One panel: errors of every method over a Monte-Carlo, per axis value.</span>

<span class="sd">    The trials of every axis value are independent, so the whole panel is one</span>
<span class="sd">    flat job list. On the numpy backend it is handed to a process pool; the</span>
<span class="sd">    other backends already batch internally and are run in this process.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">errors</span> <span class="o">=</span> <span class="p">{</span><span class="n">name</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">axis_values</span><span class="p">),</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">))</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">METHODS</span><span class="p">}</span>
    <span class="n">jobs</span> <span class="o">=</span> <span class="p">[</span>
        <span class="p">(</span>
            <span class="n">index</span><span class="p">,</span>
            <span class="n">trial</span><span class="p">,</span>
            <span class="n">value</span> <span class="k">if</span> <span class="n">axis_name</span> <span class="o">==</span> <span class="s2">&quot;n_matrices&quot;</span> <span class="k">else</span> <span class="n">fixed</span><span class="p">,</span>
            <span class="n">value</span> <span class="k">if</span> <span class="n">axis_name</span> <span class="o">==</span> <span class="s2">&quot;n_samples&quot;</span> <span class="k">else</span> <span class="n">fixed</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="n">value</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">axis_values</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">trial</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">)</span>
    <span class="p">]</span>
    <span class="n">worker</span> <span class="o">=</span> <span class="n">partial</span><span class="p">(</span>
        <span class="n">one_trial</span><span class="p">,</span> <span class="n">rng_seed</span><span class="o">=</span><span class="n">rng_seed</span><span class="p">,</span> <span class="n">centre</span><span class="o">=</span><span class="n">centre</span><span class="p">,</span>
        <span class="n">n_axis</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">axis_values</span><span class="p">),</span> <span class="n">args</span><span class="o">=</span><span class="n">args</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">n_workers</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_workers</span> <span class="ow">or</span> <span class="n">os</span><span class="o">.</span><span class="n">cpu_count</span><span class="p">()</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span> <span class="o">==</span> <span class="s2">&quot;numpy&quot;</span> <span class="ow">and</span> <span class="n">n_workers</span> <span class="o">&gt;</span> <span class="mi">1</span><span class="p">:</span>
        <span class="k">with</span> <span class="n">Pool</span><span class="p">(</span><span class="n">n_workers</span><span class="p">)</span> <span class="k">as</span> <span class="n">pool</span><span class="p">:</span>
            <span class="n">results</span> <span class="o">=</span> <span class="n">pool</span><span class="o">.</span><span class="n">map</span><span class="p">(</span><span class="n">worker</span><span class="p">,</span> <span class="n">jobs</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">results</span> <span class="o">=</span> <span class="p">[</span><span class="n">worker</span><span class="p">(</span><span class="n">job</span><span class="p">)</span> <span class="k">for</span> <span class="n">job</span> <span class="ow">in</span> <span class="n">jobs</span><span class="p">]</span>

    <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="n">trial</span><span class="p">,</span> <span class="n">scores</span> <span class="ow">in</span> <span class="n">results</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">score</span> <span class="ow">in</span> <span class="n">scores</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
            <span class="n">errors</span><span class="p">[</span><span class="n">name</span><span class="p">][</span><span class="n">index</span><span class="p">,</span> <span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="n">score</span>

    <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="n">value</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">axis_values</span><span class="p">):</span>
        <span class="nb">print</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;  </span><span class="si">{</span><span class="n">axis_name</span><span class="si">}</span><span class="s2"> = </span><span class="si">{</span><span class="n">value</span><span class="si">:</span><span class="s2">5d</span><span class="si">}</span><span class="s2">: &quot;</span>
            <span class="o">+</span> <span class="s2">&quot;  &quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="mi">10</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">errors</span><span class="p">[</span><span class="n">name</span><span class="p">][</span><span class="n">index</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">())</span><span class="si">:</span><span class="s2">7.2f</span><span class="si">}</span><span class="s2"> dB&quot;</span>
                <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">METHODS</span>
            <span class="p">)</span>
        <span class="p">)</span>
    <span class="k">return</span> <span class="n">errors</span>


<span class="k">def</span><span class="w"> </span><span class="nf">draw</span><span class="p">(</span><span class="n">ax</span><span class="p">,</span> <span class="n">axis_values</span><span class="p">,</span> <span class="n">errors</span><span class="p">,</span> <span class="n">xlabel</span><span class="p">,</span> <span class="n">first</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;One panel: mean curve per method, with a 5/95 interpercentile band.&quot;&quot;&quot;</span>
    <span class="n">colors</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;SCM&quot;</span><span class="p">:</span> <span class="s2">&quot;C3&quot;</span><span class="p">,</span> <span class="s2">&quot;LW&quot;</span><span class="p">:</span> <span class="s2">&quot;C1&quot;</span><span class="p">,</span> <span class="s2">&quot;OAS&quot;</span><span class="p">:</span> <span class="s2">&quot;C4&quot;</span><span class="p">,</span> <span class="s2">&quot;LW-NL&quot;</span><span class="p">:</span> <span class="s2">&quot;C0&quot;</span><span class="p">,</span> <span class="s2">&quot;RMT&quot;</span><span class="p">:</span> <span class="s2">&quot;C2&quot;</span><span class="p">}</span>
    <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">METHODS</span><span class="p">:</span>
        <span class="n">decibels</span> <span class="o">=</span> <span class="mi">10</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">errors</span><span class="p">[</span><span class="n">name</span><span class="p">])</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">axis_values</span><span class="p">,</span> <span class="n">decibels</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">),</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span>
            <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.4</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">name</span> <span class="k">if</span> <span class="n">first</span> <span class="k">else</span> <span class="kc">None</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">fill_between</span><span class="p">(</span>
            <span class="n">axis_values</span><span class="p">,</span>
            <span class="n">np</span><span class="o">.</span><span class="n">percentile</span><span class="p">(</span><span class="n">decibels</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">),</span>
            <span class="n">np</span><span class="o">.</span><span class="n">percentile</span><span class="p">(</span><span class="n">decibels</span><span class="p">,</span> <span class="mi">95</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">),</span>
            <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">[</span><span class="n">name</span><span class="p">],</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.15</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="c1"># Minor tick labels off. Two reasons: on a range this narrow matplotlib</span>
    <span class="c1"># labels a dozen minor decades, which is unreadable at export size; and</span>
    <span class="c1"># matplot2tikz mis-exports them — it emits the label list without the</span>
    <span class="c1"># `minor xticklabels={` key that opens it, which is a syntax error in the</span>
    <span class="c1"># generated .tex and stops the dissertation build outright.</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">xaxis</span><span class="o">.</span><span class="n">set_minor_formatter</span><span class="p">(</span><span class="n">NullFormatter</span><span class="p">())</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="n">xlabel</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">first</span><span class="p">:</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;eqm (dB)&quot;</span><span class="p">)</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span>
        <span class="s2">&quot;Mean squared error of the Fréchet mean of a set of covariances, &quot;</span>
        <span class="s2">&quot;against the number of samples and against the number of matrices.&quot;</span>
    <span class="p">)</span>
    <span class="n">add_mc_base_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Dimension of the covariances. The paper uses 64.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">65</span><span class="p">,</span> <span class="mi">68</span><span class="p">,</span> <span class="mi">80</span><span class="p">,</span> <span class="mi">100</span><span class="p">,</span> <span class="mi">150</span><span class="p">,</span> <span class="mi">200</span><span class="p">,</span> <span class="mi">300</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Sample sizes of the first panel. The smallest is barely above &quot;</span>
             <span class="s2">&quot;the dimension, which is where the correction matters most.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_matrices_fixed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of matrices held fixed in the first panel.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_matrices&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">20</span><span class="p">,</span> <span class="mi">30</span><span class="p">,</span> <span class="mi">40</span><span class="p">,</span> <span class="mi">60</span><span class="p">,</span> <span class="mi">80</span><span class="p">,</span> <span class="mi">100</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Numbers of matrices of the second panel.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_samples_fixed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">128</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Sample size held fixed in the second panel.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--condition_number&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">100.0</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Condition number of the true mean.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--scale&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Spread of the covariances around their mean, in the tangent &quot;</span>
             <span class="s2">&quot;space. Large values quickly cost numerical stability.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_iterations_max&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Iteration budget of the Riemannian descents.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--tol&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-6</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Stopping tolerance on the relative change of the iterate.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_width&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.44</span><span class="se">\\</span><span class="s2">textwidth&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_height&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;5.2cm&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Height of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>
    <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">export_path</span>

    <span class="n">init_logging</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>
    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">rng</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">default_rng</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>
    <span class="n">centre</span> <span class="o">=</span> <span class="n">random_spd</span><span class="p">(</span><span class="n">rng</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">condition_number</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;d = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_features</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2"> trials, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;backend = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;panel 1 — against the number of samples &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;(K = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_matrices_fixed</span><span class="si">}</span><span class="s2">):&quot;</span><span class="p">)</span>
    <span class="n">errors_samples</span> <span class="o">=</span> <span class="n">sweep</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">centre</span><span class="p">,</span> <span class="s2">&quot;n_samples&quot;</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_matrices_fixed</span><span class="p">,</span> <span class="n">args</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;panel 2 — against the number of matrices &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;(N = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples_fixed</span><span class="si">}</span><span class="s2">):&quot;</span><span class="p">)</span>
    <span class="n">errors_matrices</span> <span class="o">=</span> <span class="n">sweep</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">seed</span> <span class="o">+</span> <span class="mi">1</span><span class="p">,</span> <span class="n">centre</span><span class="p">,</span> <span class="s2">&quot;n_matrices&quot;</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_matrices</span><span class="p">,</span>
        <span class="n">args</span><span class="o">.</span><span class="n">n_samples_fixed</span><span class="p">,</span> <span class="n">args</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">6.4</span><span class="p">,</span> <span class="mf">3.2</span><span class="p">))</span>
    <span class="n">draw</span><span class="p">(</span><span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">,</span> <span class="n">errors_samples</span><span class="p">,</span>
         <span class="sa">r</span><span class="s2">&quot;nombre d&#39;échantillons $N$&quot;</span><span class="p">,</span> <span class="n">first</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">draw</span><span class="p">(</span><span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">args</span><span class="o">.</span><span class="n">n_matrices</span><span class="p">,</span> <span class="n">errors_matrices</span><span class="p">,</span>
         <span class="sa">r</span><span class="s2">&quot;nombre de matrices $K$&quot;</span><span class="p">,</span> <span class="n">first</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="c1"># Legend below the panels: at export size an inner one covers the curves.</span>
    <span class="c1"># Attached to an axis, since matplot2tikz drops figure legends.</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span>
        <span class="n">loc</span><span class="o">=</span><span class="s2">&quot;upper left&quot;</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">0.0</span><span class="p">,</span> <span class="o">-</span><span class="mf">0.30</span><span class="p">),</span> <span class="n">ncol</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">frameon</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
            <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
            <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_features</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span>
            <span class="n">n_trials</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">condition_number</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">condition_number</span><span class="p">,</span>
            <span class="n">scale</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">scale</span><span class="p">,</span> <span class="n">centre</span><span class="o">=</span><span class="n">centre</span><span class="p">,</span>
            <span class="n">n_samples</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples</span><span class="p">),</span>
            <span class="n">n_matrices</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">asarray</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">n_matrices</span><span class="p">),</span>
            <span class="n">n_matrices_fixed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_matrices_fixed</span><span class="p">,</span>
            <span class="n">n_samples_fixed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples_fixed</span><span class="p">,</span>
            <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;samples_</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">errors_samples</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">METHODS</span><span class="p">},</span>
            <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;matrices_</span><span class="si">{</span><span class="n">name</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">errors_matrices</span><span class="p">[</span><span class="n">name</span><span class="p">]</span> <span class="k">for</span> <span class="n">name</span> <span class="ow">in</span> <span class="n">METHODS</span><span class="p">},</span>
        <span class="p">)</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;frechet_mse.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">save_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved MSE figure in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_features</span><span class="param-type">int</span><span class="param-default">default <b>64</b></span>
</div>
<p class="param-help">Dimension of the covariances. The paper uses 64.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples</span><span class="param-type">int</span><span class="param-default">default <b>[65, 68, 80, 100, 150, 200, 300]</b></span>
</div>
<p class="param-help">Sample sizes of the first panel. The smallest is barely above the dimension, which is where the correction matters most.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_matrices_fixed</span><span class="param-type">int</span><span class="param-default">default <b>10</b></span>
</div>
<p class="param-help">Number of matrices held fixed in the first panel.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_matrices</span><span class="param-type">int</span><span class="param-default">default <b>[3, 5, 20, 30, 40, 60, 80, 100]</b></span>
</div>
<p class="param-help">Numbers of matrices of the second panel.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_samples_fixed</span><span class="param-type">int</span><span class="param-default">default <b>128</b></span>
</div>
<p class="param-help">Sample size held fixed in the second panel.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--condition_number</span><span class="param-type">float</span><span class="param-default">default <b>100.0</b></span>
</div>
<p class="param-help">Condition number of the true mean.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--scale</span><span class="param-type">float</span><span class="param-default">default <b>0.1</b></span>
</div>
<p class="param-help">Spread of the covariances around their mean, in the tangent space. Large values quickly cost numerical stability.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_iterations_max</span><span class="param-type">int</span><span class="param-default">default <b>100</b></span>
</div>
<p class="param-help">Iteration budget of the Riemannian descents.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--tol</span><span class="param-type">float</span><span class="param-default">default <b>1e-06</b></span>
</div>
<p class="param-help">Stopping tolerance on the relative change of the iterate.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_width</span><span class="param-type">str</span><span class="param-default">default <b>0.44\textwidth</b></span>
</div>
<p class="param-help">Width of a single panel in the exported PGFPlots figure.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_height</span><span class="param-type">str</span><span class="param-default">default <b>5.2cm</b></span>
</div>
<p class="param-help">Height of a single panel in the exported PGFPlots figure.</p>
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

## Results

<span class="marginnote">
  <span class="mn-label">Run</span>
  <code>--n-trials 100</code><br>
  <code>--seed 42</code><br>
  <code>--backend numpy</code><br>
  <code>--n_features 64</code><br>
  <code>--n_samples [65, 68, 80, 100, 150, 200, 300]</code><br>
  <code>--n_matrices_fixed 10</code><br>
  <code>--n_matrices [3, 5, 20, 30, 40, 60, 80, 100]</code><br>
  <code>--n_samples_fixed 128</code><br>
  <code>--condition_number 100.0</code><br>
  <code>--scale 0.1</code><br>
  <code>--n_iterations_max 100</code><br>
  <code>--tol 1e-06</code><br>
  <code>--axis_width 0.44\textwidth</code><br>
  <code>--axis_height 5.2cm</code><br>
  <span class="mn-date">6149f77 · 2026-08-27</span>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/learning_frechet_mse.json" data-title="learning_frechet_mse"></div>
</div>

## Config

`3-learning/experiments/learning_frechet_mse.yaml`

<a class="back-link" href="../../chapters/3-learning/">← All experiments in 3 · Learning</a>
