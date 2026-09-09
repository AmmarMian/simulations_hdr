<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/3-learning/">3 · Learning</a>
<span class="sep">/</span>
<span class="here">learning_hyperspectral_metrics</span>
</nav>

# learning_hyperspectral_metrics

Euclidean, log-Euclidean and affine-invariant K-means on a hyperspectral scene — what the geometry alone buys, before any correction, entirely on the device

**Tags:** `learning`  `clustering`  `hyperspectral`  `gpu`

## Run

```sh
uv run python 3-learning/hyperspectral-metrics/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/3-learning/hyperspectral-metrics/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">333 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">3-learning/hyperspectral-metrics/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># What does the geometry alone buy? Euclidean against Riemannian K-means.</span>
<span class="c1">#</span>
<span class="c1"># The sibling experiment (3-learning/hyperspectral-rmt) asks whether the RMT</span>
<span class="c1"># correction survives downstream, and answers it by holding the optimiser fixed</span>
<span class="c1"># and varying the estimator. This one steps back and asks the prior question,</span>
<span class="c1"># the one pyRiemann&#39;s image-radar example poses: before any correction, how much</span>
<span class="c1"># of the gain is the *metric*? Same scene, same windows, same covariances, same</span>
<span class="c1"># alternation — only the geometry in which the centroids are means changes.</span>
<span class="c1">#</span>
<span class="c1"># Three geometries, in increasing order of what they respect and of what they</span>
<span class="c1"># cost:</span>
<span class="c1">#</span>
<span class="c1">#   euclid     the cone treated as a flat vector space. The centroid is the</span>
<span class="c1">#              arithmetic mean of the covariances, closed-form. It is the</span>
<span class="c1">#              baseline that a comparison needs and, on its own, the reason the</span>
<span class="c1">#              literature bothered with the other two.</span>
<span class="c1">#   logeuclid  flat, but on the matrix logarithms. Respects the positivity of</span>
<span class="c1">#              the eigenvalues; not affine-invariant. Still closed-form, and it</span>
<span class="c1">#              pays for its eigendecompositions once for the whole run.</span>
<span class="c1">#   riemann    the affine-invariant metric. The centroid is a Karcher mean, so</span>
<span class="c1">#              the only one of the three that iterates. It is what pyRiemann</span>
<span class="c1">#              calls ``riemann`` and what the RMT correction of the sibling</span>
<span class="c1">#              experiment corrects.</span>
<span class="c1">#</span>
<span class="c1"># Everything runs on the device — hdrlib.learning.clustering.spd_kmeans keeps the</span>
<span class="c1"># covariances, the centroids and the labels there, and reads back only the two</span>
<span class="c1"># scalars that decide control flow. The windows are dropped before the loop</span>
<span class="c1"># starts, since none of these three metrics needs the samples a covariance came</span>
<span class="c1"># from; that is the whole reason this experiment fits on a GPU and the corrected</span>
<span class="c1"># one does not.</span>
<span class="c1">#</span>
<span class="c1"># float64 throughout, and enforced rather than assumed: all three metrics end in</span>
<span class="c1"># the eigenvalues of a 5x5 covariance built from 25 samples, two of them take the</span>
<span class="c1"># logarithm, and in single precision the smallest eigenvalue&#39;s sign is not</span>
<span class="c1"># reliable — so the answer would be wrong rather than merely imprecise.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">platform</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">time</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">empty_cache</span><span class="p">,</span>
    <span class="n">get_data_on_device</span><span class="p">,</span>
    <span class="n">peak_memory_bytes</span><span class="p">,</span>
    <span class="n">reset_peak_memory</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.learning.clustering</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">SPD_METRICS</span><span class="p">,</span>
    <span class="n">clustering_accuracy</span><span class="p">,</span>
    <span class="n">match_labels</span><span class="p">,</span>
    <span class="n">reference_mean_iou</span><span class="p">,</span>
    <span class="n">mean_iou</span><span class="p">,</span>
    <span class="n">require_double</span><span class="p">,</span>
    <span class="n">spd_kmeans</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.learning.hyperspectral</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">crop_labels</span><span class="p">,</span>
    <span class="n">download_scene</span><span class="p">,</span>
    <span class="n">pca_image</span><span class="p">,</span>
    <span class="n">read_scene</span><span class="p">,</span>
    <span class="n">remove_global_mean</span><span class="p">,</span>
    <span class="n">sliding_window_vectorize</span><span class="p">,</span>
    <span class="n">unvectorize_labels</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.mc</span><span class="w"> </span><span class="kn">import</span> <span class="n">Progress</span><span class="p">,</span> <span class="n">add_mc_base_args</span><span class="p">,</span> <span class="n">make_mc_parser</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.learning.rmt</span><span class="w"> </span><span class="kn">import</span> <span class="n">scm</span>

<span class="c1"># Same colour per metric everywhere: the figure, the docs export, the tables.</span>
<span class="n">COLOURS</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;euclid&quot;</span><span class="p">:</span> <span class="s2">&quot;#c0504d&quot;</span><span class="p">,</span> <span class="s2">&quot;logeuclid&quot;</span><span class="p">:</span> <span class="s2">&quot;#dea11f&quot;</span><span class="p">,</span> <span class="s2">&quot;riemann&quot;</span><span class="p">:</span> <span class="s2">&quot;#59bfa3&quot;</span><span class="p">}</span>
<span class="n">LABELS</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">&quot;euclid&quot;</span><span class="p">:</span> <span class="s2">&quot;euclidien&quot;</span><span class="p">,</span>
    <span class="s2">&quot;logeuclid&quot;</span><span class="p">:</span> <span class="s2">&quot;log-euclidien&quot;</span><span class="p">,</span>
    <span class="s2">&quot;riemann&quot;</span><span class="p">:</span> <span class="s2">&quot;riemannien&quot;</span><span class="p">,</span>
<span class="p">}</span>


<span class="k">def</span><span class="w"> </span><span class="nf">describe_device</span><span class="p">(</span><span class="n">backend</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Name the hardware the timings were measured on.</span>

<span class="sd">    Without this the seconds in ``scores.json`` are unreadable a year later, and</span>
<span class="sd">    for this experiment they are unreadable in a specific way: float64 runs at</span>
<span class="sd">    half the float32 rate on a datacentre card and at a sixty-fourth of it on a</span>
<span class="sd">    workstation one. The affine-invariant metric is the one that notices — it is</span>
<span class="sd">    eigendecomposition-bound where the two flat metrics are matmul-bound — so the</span>
<span class="sd">    *ranking by time* is a property of the card as much as of the method.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">information</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">&quot;backend&quot;</span><span class="p">:</span> <span class="n">backend</span><span class="p">,</span>
        <span class="s2">&quot;device&quot;</span><span class="p">:</span> <span class="s2">&quot;cpu&quot;</span><span class="p">,</span>
        <span class="s2">&quot;host&quot;</span><span class="p">:</span> <span class="n">platform</span><span class="o">.</span><span class="n">node</span><span class="p">(),</span>
        <span class="s2">&quot;platform&quot;</span><span class="p">:</span> <span class="n">platform</span><span class="o">.</span><span class="n">platform</span><span class="p">(),</span>
    <span class="p">}</span>
    <span class="k">if</span> <span class="n">backend</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">&quot;torch&quot;</span><span class="p">):</span>
        <span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

        <span class="n">information</span><span class="p">[</span><span class="s2">&quot;torch&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">__version__</span>
        <span class="k">if</span> <span class="n">backend</span> <span class="o">==</span> <span class="s2">&quot;torch-cuda&quot;</span> <span class="ow">and</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_available</span><span class="p">():</span>
            <span class="n">properties</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">get_device_properties</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
            <span class="n">information</span><span class="p">[</span><span class="s2">&quot;device&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">properties</span><span class="o">.</span><span class="n">name</span>
            <span class="n">information</span><span class="p">[</span><span class="s2">&quot;vram_gb&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="nb">round</span><span class="p">(</span><span class="n">properties</span><span class="o">.</span><span class="n">total_memory</span> <span class="o">/</span> <span class="mi">1024</span><span class="o">**</span><span class="mi">3</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
            <span class="n">information</span><span class="p">[</span><span class="s2">&quot;capability&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">properties</span><span class="o">.</span><span class="n">major</span><span class="si">}</span><span class="s2">.</span><span class="si">{</span><span class="n">properties</span><span class="o">.</span><span class="n">minor</span><span class="si">}</span><span class="s2">&quot;</span>
    <span class="k">return</span> <span class="n">information</span>


<span class="k">def</span><span class="w"> </span><span class="nf">prepare</span><span class="p">(</span><span class="n">scene</span><span class="p">,</span> <span class="n">data_path</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">window_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Scene -&gt; one covariance per pixel, on the device, plus its truth.</span>

<span class="sd">    The cube crosses to the device once, here, and nothing crosses back until</span>
<span class="sd">    the labels are scored. ``scipy.io.loadmat`` hands back a numpy array whatever</span>
<span class="sd">    the backend is, so without the explicit move every step below would be a</span>
<span class="sd">    backend call on a host array.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">cube</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">n_classes</span> <span class="o">=</span> <span class="n">read_scene</span><span class="p">(</span><span class="n">scene</span><span class="p">,</span> <span class="n">data_path</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">n_features</span> <span class="o">&gt;=</span> <span class="n">cube</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]:</span>
        <span class="c1"># pca_image returns the cube untouched in this case, and the windows</span>
        <span class="c1"># would then be (n_pixels, window², n_bands) — 4.4 GB on Salinas at 204</span>
        <span class="c1"># bands, against 108 MB at five components. The reduction is not an</span>
        <span class="c1"># optimisation here, it is what makes the windowing representable at all.</span>
        <span class="k">raise</span> <span class="ne">ValueError</span><span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;n_features=</span><span class="si">{</span><span class="n">n_features</span><span class="si">}</span><span class="s2"> does not reduce a </span><span class="si">{</span><span class="n">cube</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">-band &quot;</span>
            <span class="s2">&quot;cube; the windows would not fit. Choose n_features well below the &quot;</span>
            <span class="s2">&quot;number of bands (five is the configuration of the published table).&quot;</span>
        <span class="p">)</span>
    <span class="n">cube</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">cube</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">require_double</span><span class="p">(</span><span class="n">cube</span><span class="p">,</span> <span class="s2">&quot;the hyperspectral cube&quot;</span><span class="p">)</span>

    <span class="n">centred</span> <span class="o">=</span> <span class="n">remove_global_mean</span><span class="p">(</span><span class="n">cube</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="c1"># Global scale normalisation. The affine-invariant distance is unchanged by</span>
    <span class="c1"># a common positive factor, so this is statistically free; it is not free</span>
    <span class="c1"># numerically, since raw radiance puts the eigenvalues around 1e8 and the</span>
    <span class="c1"># eigensolvers lose most of their precision before they start. The two flat</span>
    <span class="c1"># metrics are *not* scale-invariant, so for them this fixes the units the</span>
    <span class="c1"># comparison is made in — one more reason to do it before the split.</span>
    <span class="n">centred</span> <span class="o">=</span> <span class="n">centred</span> <span class="o">/</span> <span class="n">centred</span><span class="o">.</span><span class="n">std</span><span class="p">()</span>
    <span class="n">reduced</span> <span class="o">=</span> <span class="n">pca_image</span><span class="p">(</span><span class="n">centred</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">windows</span> <span class="o">=</span> <span class="n">sliding_window_vectorize</span><span class="p">(</span><span class="n">reduced</span><span class="p">,</span> <span class="n">window_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">truth</span> <span class="o">=</span> <span class="n">crop_labels</span><span class="p">(</span><span class="n">labels</span><span class="p">,</span> <span class="n">window_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">)</span>

    <span class="c1"># Formed once, here, rather than inside the K-means: none of these three</span>
    <span class="c1"># metrics needs the samples the covariance came from, which is exactly what</span>
    <span class="c1"># lets the windows be dropped before the loop starts. On Salinas that is</span>
    <span class="c1"># 108 MB returned against 22 MB kept.</span>
    <span class="n">covariances</span> <span class="o">=</span> <span class="n">scm</span><span class="p">(</span><span class="n">windows</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="k">del</span> <span class="n">windows</span><span class="p">,</span> <span class="n">reduced</span><span class="p">,</span> <span class="n">centred</span><span class="p">,</span> <span class="n">cube</span>
    <span class="n">empty_cache</span><span class="p">(</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">covariances</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">labels</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="n">n_classes</span>


<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span>
        <span class="s2">&quot;Euclidean, log-Euclidean and affine-invariant K-means on a &quot;</span>
        <span class="s2">&quot;hyperspectral scene, entirely on the device.&quot;</span>
    <span class="p">)</span>
    <span class="n">add_mc_base_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--scene&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;salinas&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Scene to segment: indianpines or salinas. Salinas is the one &quot;</span>
             <span class="s2">&quot;pyRiemann&#39;s example uses.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--data_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;data/hyperspectral&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Where the .mat files live; downloaded there if missing.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Principal components kept. Five is pyRiemann&#39;s setting and the &quot;</span>
             <span class="s2">&quot;configuration of the published table.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--window_size&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Side of the square neighbourhood, odd. With n_features=5 this &quot;</span>
             <span class="s2">&quot;gives 25 samples for 5 variables, c = 0.2.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--stride&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Step between two windows. One segments every pixel; a larger &quot;</span>
             <span class="s2">&quot;value trades resolution for time, and the caption must say so.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_init&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Restarts of the K-means; the one of least inertia is kept. Every &quot;</span>
             <span class="s2">&quot;metric gets the same starting partitions.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--max_iter&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">100</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Assignment/re-estimation rounds per restart.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--mean_iterations&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Iteration budget of one Karcher mean, warm-started on the &quot;</span>
             <span class="s2">&quot;previous centroids. Ignored by the two flat metrics.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--max_batch&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">16000</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Largest batch handed to the eigensolver at once. Only the &quot;</span>
             <span class="s2">&quot;affine-invariant metric is bounded by it: cuSOLVER refuses a &quot;</span>
             <span class="s2">&quot;batch of n_clusters x n_pixels matrices outright on a scene the &quot;</span>
             <span class="s2">&quot;size of Salinas. Lowering it costs kernel launches, not results.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--metrics&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="nb">list</span><span class="p">(</span><span class="n">SPD_METRICS</span><span class="p">),</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Subset of the three geometries to run.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--figure_width&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.23</span><span class="se">\\</span><span class="s2">textwidth&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of one map in the exported figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>
    <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">export_path</span>

    <span class="c1"># Not init_logging: its GPU disclaimer is written for the Monte-Carlo</span>
    <span class="c1"># experiments, whose batched path is sequential over T checkpoints. Nothing</span>
    <span class="c1"># here is a Monte-Carlo trial and the device path is the fast one, so the</span>
    <span class="c1"># warning would be actively misleading.</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>
    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">device</span> <span class="o">=</span> <span class="n">describe_device</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;backend </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="si">}</span><span class="s2"> on </span><span class="si">{</span><span class="n">device</span><span class="p">[</span><span class="s1">&#39;device&#39;</span><span class="p">]</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">reset_peak_memory</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>

    <span class="n">download_scene</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">data_path</span><span class="p">)</span>
    <span class="n">covariances</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">image_shape</span><span class="p">,</span> <span class="n">n_classes</span> <span class="o">=</span> <span class="n">prepare</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">data_path</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="p">,</span>
        <span class="n">args</span><span class="o">.</span><span class="n">stride</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">concentration</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="o">**</span><span class="mi">2</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">covariances</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s2"> covariances &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">covariances</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">x</span><span class="si">{</span><span class="n">covariances</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;c = </span><span class="si">{</span><span class="n">concentration</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">n_classes</span><span class="si">}</span><span class="s2"> classes&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">progress</span> <span class="o">=</span> <span class="n">Progress</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">metrics</span><span class="p">),</span>
        <span class="n">description</span><span class="o">=</span><span class="s2">&quot;Geometries&quot;</span><span class="p">,</span> <span class="n">unit</span><span class="o">=</span><span class="s2">&quot;metrics&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">maps</span><span class="p">,</span> <span class="n">scores</span> <span class="o">=</span> <span class="p">{},</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">metric</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">metrics</span><span class="p">:</span>
        <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
        <span class="n">labels</span><span class="p">,</span> <span class="n">inertia</span><span class="p">,</span> <span class="n">histories</span> <span class="o">=</span> <span class="n">spd_kmeans</span><span class="p">(</span>
            <span class="n">covariances</span><span class="p">,</span> <span class="n">n_classes</span><span class="p">,</span> <span class="n">metric</span><span class="o">=</span><span class="n">metric</span><span class="p">,</span> <span class="n">n_init</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_init</span><span class="p">,</span>
            <span class="n">max_iter</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">max_iter</span><span class="p">,</span> <span class="n">mean_iterations</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">mean_iterations</span><span class="p">,</span>
            <span class="n">max_batch</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">max_batch</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">backend</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
            <span class="n">verbose</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">segmented</span> <span class="o">=</span> <span class="n">unvectorize_labels</span><span class="p">(</span>
            <span class="n">labels</span><span class="p">,</span> <span class="o">*</span><span class="n">image_shape</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">stride</span>
        <span class="p">)</span>
        <span class="n">matched</span> <span class="o">=</span> <span class="n">match_labels</span><span class="p">(</span><span class="n">segmented</span><span class="p">,</span> <span class="n">truth</span><span class="p">)</span>
        <span class="n">accuracy</span> <span class="o">=</span> <span class="n">clustering_accuracy</span><span class="p">(</span><span class="n">matched</span><span class="p">,</span> <span class="n">truth</span><span class="p">)</span>
        <span class="n">ious</span><span class="p">,</span> <span class="n">miou</span> <span class="o">=</span> <span class="n">mean_iou</span><span class="p">(</span><span class="n">matched</span><span class="p">,</span> <span class="n">truth</span><span class="p">)</span>
        <span class="n">miou_reference</span> <span class="o">=</span> <span class="n">reference_mean_iou</span><span class="p">(</span><span class="n">matched</span><span class="p">,</span> <span class="n">truth</span><span class="p">)</span>
        <span class="n">elapsed</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">start</span>

        <span class="n">maps</span><span class="p">[</span><span class="n">metric</span><span class="p">]</span> <span class="o">=</span> <span class="n">matched</span>
        <span class="n">scores</span><span class="p">[</span><span class="n">metric</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">&quot;accuracy&quot;</span><span class="p">:</span> <span class="n">accuracy</span><span class="p">,</span> <span class="s2">&quot;mIoU&quot;</span><span class="p">:</span> <span class="n">miou</span><span class="p">,</span>
            <span class="s2">&quot;mIoU_reference&quot;</span><span class="p">:</span> <span class="n">miou_reference</span><span class="p">,</span>
            <span class="c1"># Comparable between restarts of one metric, never between metrics:</span>
            <span class="c1"># the three measure lengths in different geometries. The ranking of</span>
            <span class="c1"># the metrics is the accuracy and the mIoU, which are on the truth.</span>
            <span class="s2">&quot;inertia&quot;</span><span class="p">:</span> <span class="n">inertia</span><span class="p">,</span> <span class="s2">&quot;seconds&quot;</span><span class="p">:</span> <span class="n">elapsed</span><span class="p">,</span>
            <span class="s2">&quot;restarts&quot;</span><span class="p">:</span> <span class="n">histories</span><span class="p">,</span>
            <span class="s2">&quot;worst_moved&quot;</span><span class="p">:</span> <span class="nb">max</span><span class="p">(</span><span class="n">h</span><span class="p">[</span><span class="s2">&quot;moved&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">h</span> <span class="ow">in</span> <span class="n">histories</span><span class="p">),</span>
        <span class="p">}</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">LABELS</span><span class="p">[</span><span class="n">metric</span><span class="p">]</span><span class="si">:</span><span class="s2">15s</span><span class="si">}</span><span class="s2"> acc=</span><span class="si">{</span><span class="n">accuracy</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">  mIoU=</span><span class="si">{</span><span class="n">miou</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">  &quot;</span>
              <span class="sa">f</span><span class="s2">&quot;(</span><span class="si">{</span><span class="n">elapsed</span><span class="si">:</span><span class="s2">.0f</span><span class="si">}</span><span class="s2">s, worst restart left &quot;</span>
              <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">scores</span><span class="p">[</span><span class="n">metric</span><span class="p">][</span><span class="s1">&#39;worst_moved&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.2%</span><span class="si">}</span><span class="s2"> moving)&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="n">progress</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>

    <span class="n">peak</span> <span class="o">=</span> <span class="n">peak_memory_bytes</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">peak</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
        <span class="n">device</span><span class="p">[</span><span class="s2">&quot;peak_vram_gb&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="nb">round</span><span class="p">(</span><span class="n">peak</span> <span class="o">/</span> <span class="mi">1024</span><span class="o">**</span><span class="mi">3</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;peak device memory </span><span class="si">{</span><span class="n">device</span><span class="p">[</span><span class="s1">&#39;peak_vram_gb&#39;</span><span class="p">]</span><span class="si">}</span><span class="s2"> GB&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="c1"># ── the figure: ground truth, then one map per geometry ───────────────</span>
    <span class="n">panels</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;vérité terrain&quot;</span><span class="p">]</span> <span class="o">+</span> <span class="nb">list</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">metrics</span><span class="p">)</span>
    <span class="n">figure</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">panels</span><span class="p">),</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">2.0</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">panels</span><span class="p">),</span> <span class="mf">2.4</span><span class="p">))</span>
    <span class="n">axes</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">atleast_1d</span><span class="p">(</span><span class="n">axes</span><span class="p">)</span>
    <span class="c1"># A discrete colormap: these are class labels, not a continuous field, so a</span>
    <span class="c1"># perceptual gradient would suggest an order between crops that has none.</span>
    <span class="n">colormap</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">get_cmap</span><span class="p">(</span><span class="s2">&quot;tab20&quot;</span><span class="p">,</span> <span class="n">n_classes</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">axis</span><span class="p">,</span> <span class="n">panel</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">panels</span><span class="p">):</span>
        <span class="n">image</span> <span class="o">=</span> <span class="n">truth</span> <span class="k">if</span> <span class="n">panel</span> <span class="o">==</span> <span class="s2">&quot;vérité terrain&quot;</span> <span class="k">else</span> <span class="n">maps</span><span class="p">[</span><span class="n">panel</span><span class="p">]</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">colormap</span><span class="p">,</span> <span class="n">vmin</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">vmax</span><span class="o">=</span><span class="n">n_classes</span><span class="p">,</span>
                    <span class="n">interpolation</span><span class="o">=</span><span class="s2">&quot;nearest&quot;</span><span class="p">)</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_xticks</span><span class="p">([])</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_yticks</span><span class="p">([])</span>
        <span class="k">if</span> <span class="n">panel</span> <span class="o">==</span> <span class="s2">&quot;vérité terrain&quot;</span><span class="p">:</span>
            <span class="n">axis</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="n">panel</span><span class="p">,</span> <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">axis</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">LABELS</span><span class="p">[</span><span class="n">panel</span><span class="p">]</span><span class="si">}</span><span class="se">\\</span><span class="s2">n</span><span class="si">{</span><span class="n">scores</span><span class="p">[</span><span class="n">panel</span><span class="p">][</span><span class="s1">&#39;accuracy&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2"> / &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">scores</span><span class="p">[</span><span class="n">panel</span><span class="p">][</span><span class="s1">&#39;mIoU&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span>
                <span class="n">fontsize</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
            <span class="p">)</span>
    <span class="n">figure</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
            <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
            <span class="n">scene</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="p">,</span> <span class="n">seed</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">,</span> <span class="n">n_features</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span>
            <span class="n">window_size</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="p">,</span> <span class="n">stride</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">stride</span><span class="p">,</span>
            <span class="n">n_init</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_init</span><span class="p">,</span> <span class="n">max_iter</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">max_iter</span><span class="p">,</span>
            <span class="n">mean_iterations</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">mean_iterations</span><span class="p">,</span> <span class="n">max_batch</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">max_batch</span><span class="p">,</span>
            <span class="n">n_classes</span><span class="o">=</span><span class="n">n_classes</span><span class="p">,</span> <span class="n">concentration</span><span class="o">=</span><span class="n">concentration</span><span class="p">,</span>
            <span class="n">metrics</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">metrics</span><span class="p">),</span> <span class="n">truth</span><span class="o">=</span><span class="n">truth</span><span class="p">,</span>
            <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;map_</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">maps</span><span class="p">[</span><span class="n">m</span><span class="p">]</span> <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">metrics</span><span class="p">},</span>
        <span class="p">)</span>
        <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;scores.json&quot;</span><span class="p">),</span> <span class="s2">&quot;w&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">handle</span><span class="p">:</span>
            <span class="n">json</span><span class="o">.</span><span class="n">dump</span><span class="p">({</span><span class="s2">&quot;device&quot;</span><span class="p">:</span> <span class="n">device</span><span class="p">,</span> <span class="s2">&quot;scores&quot;</span><span class="p">:</span> <span class="n">scores</span><span class="p">},</span> <span class="n">handle</span><span class="p">,</span> <span class="n">indent</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
        <span class="c1"># Saved as an image rather than PGFPlots: these are label maps, and a</span>
        <span class="c1"># PGFPlots export of a few hundred thousand coloured cells would be</span>
        <span class="c1"># unusable both to compile and to open.</span>
        <span class="n">save_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;segmentation.pdf&quot;</span><span class="p">)</span>
        <span class="n">figure</span><span class="o">.</span><span class="n">savefig</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">bbox_inches</span><span class="o">=</span><span class="s2">&quot;tight&quot;</span><span class="p">,</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">300</span><span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">save_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;Saved segmentation maps in </span><span class="si">{</span><span class="n">save_path</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">progress</span><span class="o">.</span><span class="n">done</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">main</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--scene</span><span class="param-type">str</span><span class="param-default">default <b>salinas</b></span>
</div>
<p class="param-help">Scene to segment: indianpines or salinas. Salinas is the one pyRiemann&#x27;s example uses.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--data_path</span><span class="param-type">str</span><span class="param-default">default <b>data/hyperspectral</b></span>
</div>
<p class="param-help">Where the .mat files live; downloaded there if missing.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_features</span><span class="param-type">int</span><span class="param-default">default <b>5</b></span>
</div>
<p class="param-help">Principal components kept. Five is pyRiemann&#x27;s setting and the configuration of the published table.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--window_size</span><span class="param-type">int</span><span class="param-default">default <b>5</b></span>
</div>
<p class="param-help">Side of the square neighbourhood, odd. With n_features=5 this gives 25 samples for 5 variables, c = 0.2.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--stride</span><span class="param-type">int</span><span class="param-default">default <b>1</b></span>
</div>
<p class="param-help">Step between two windows. One segments every pixel; a larger value trades resolution for time, and the caption must say so.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_init</span><span class="param-type">int</span><span class="param-default">default <b>10</b></span>
</div>
<p class="param-help">Restarts of the K-means; the one of least inertia is kept. Every metric gets the same starting partitions.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--max_iter</span><span class="param-type">int</span><span class="param-default">default <b>100</b></span>
</div>
<p class="param-help">Assignment/re-estimation rounds per restart.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--mean_iterations</span><span class="param-type">int</span><span class="param-default">default <b>10</b></span>
</div>
<p class="param-help">Iteration budget of one Karcher mean, warm-started on the previous centroids. Ignored by the two flat metrics.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--max_batch</span><span class="param-type">int</span><span class="param-default">default <b>16000</b></span>
</div>
<p class="param-help">Largest batch handed to the eigensolver at once. Only the affine-invariant metric is bounded by it: cuSOLVER refuses a batch of n_clusters x n_pixels matrices outright on a scene the size of Salinas. Lowering it costs kernel launches, not results.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--metrics</span><span class="param-type">str</span>
</div>
<p class="param-help">Subset of the three geometries to run.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--figure_width</span><span class="param-type">str</span><span class="param-default">default <b>0.23\textwidth</b></span>
</div>
<p class="param-help">Width of one map in the exported figure.</p>
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

## Config

`3-learning/experiments/learning_hyperspectral_metrics.yaml`

<a class="back-link" href="../../chapters/3-learning/">← All experiments in 3 · Learning</a>
