<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/3-learning/">3 · Learning</a>
<span class="sep">/</span>
<span class="here">learning_hyperspectral_rmt</span>
</nav>

# learning_hyperspectral_rmt

Riemannian K-means segmentation of a hyperspectral scene — SCM, Ledoit-Wolf, non-linear shrinkage and the RMT correction, judged on the ground truth

**Tags:** `learning`  `random-matrix-theory`  `clustering`  `hyperspectral`

## Run

```sh
uv run python 3-learning/hyperspectral-rmt/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/3-learning/hyperspectral-rmt/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">298 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">3-learning/hyperspectral-rmt/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Does the correction survive downstream? Segmenting a hyperspectral scene.</span>
<span class="c1">#</span>
<span class="c1"># The eqm figure of the same chapter measures an *internal* criterion: how far</span>
<span class="c1"># the estimated Fréchet mean is from the true one. But the thesis of</span>
<span class="c1"># ch:learning is that the criterion has left the model — so the estimate has to</span>
<span class="c1"># be judged on the task, not on itself. That is what this experiment does.</span>
<span class="c1">#</span>
<span class="c1"># The pipeline is the standard one for covariance-based segmentation:</span>
<span class="c1">#</span>
<span class="c1">#   scene -&gt; remove the global mean -&gt; PCA to n_features bands</span>
<span class="c1">#         -&gt; sliding window -&gt; one covariance per pixel</span>
<span class="c1">#         -&gt; Riemannian K-means -&gt; compare to the ground truth</span>
<span class="c1">#</span>
<span class="c1"># and it is where the dimensional regime becomes concrete. A 5x5 window on 5</span>
<span class="c1"># principal components gives 25 samples for 5 variables: c = 0.2, and every</span>
<span class="c1"># pixel&#39;s covariance is estimated from a handful of neighbours. Exactly the</span>
<span class="c1"># regime of subsec:learning-rmt, and exactly the regime the second panel of the</span>
<span class="c1"># eqm figure describes — many matrices, each badly estimated.</span>
<span class="c1">#</span>
<span class="c1"># Four methods, differing only in what the centroids minimise: the plain</span>
<span class="c1"># Fréchet mean of the SCMs, of the linearly shrunk covariances, of the</span>
<span class="c1"># non-linearly shrunk ones, and the corrected mean. Two scores, because they</span>
<span class="c1"># disagree informatively: accuracy is dominated by the large classes, mIoU is</span>
<span class="c1"># not.</span>
<span class="c1">#</span>
<span class="c1"># Backend-free, float64 (see hdrlib.core.rmt.require_double), and no</span>
<span class="c1"># scikit-learn: the pipeline must be able to run on a GPU backend.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">logging</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">time</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.clustering</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">clustering_accuracy</span><span class="p">,</span>
    <span class="n">match_labels</span><span class="p">,</span>
    <span class="n">mean_iou</span><span class="p">,</span>
    <span class="n">riemannian_kmeans</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.backend</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_data_on_device</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.hyperspectral</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
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


<span class="n">METHODS</span> <span class="o">=</span> <span class="p">(</span><span class="s2">&quot;SCM&quot;</span><span class="p">,</span> <span class="s2">&quot;LW&quot;</span><span class="p">,</span> <span class="s2">&quot;LW-NL&quot;</span><span class="p">,</span> <span class="s2">&quot;RMT&quot;</span><span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">prepare</span><span class="p">(</span><span class="n">scene</span><span class="p">,</span> <span class="n">data_path</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">window_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">,</span> <span class="n">backend</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Scene -&gt; one block of neighbouring samples per pixel, plus its truth.&quot;&quot;&quot;</span>
    <span class="n">cube</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">n_classes</span> <span class="o">=</span> <span class="n">read_scene</span><span class="p">(</span><span class="n">scene</span><span class="p">,</span> <span class="n">data_path</span><span class="p">)</span>
    <span class="c1"># scipy hands back a numpy array whatever the backend is; every step below</span>
    <span class="c1"># calls into the backend module, so the cube has to cross to the device</span>
    <span class="c1"># here. Without this, --backend torch-cuda dies in remove_global_mean on</span>
    <span class="c1"># torch.mean(&lt;numpy.ndarray&gt;). The labels stay on the host: they are only</span>
    <span class="c1"># ever indexed and scored there.</span>
    <span class="n">cube</span> <span class="o">=</span> <span class="n">get_data_on_device</span><span class="p">(</span><span class="n">cube</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">centred</span> <span class="o">=</span> <span class="n">remove_global_mean</span><span class="p">(</span><span class="n">cube</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="c1"># Global scale normalisation. The affine-invariant distance is unchanged by</span>
    <span class="c1"># a common positive factor, so this is statistically free; it is not free</span>
    <span class="c1"># numerically, since raw radiance puts the eigenvalues around 1e8 and the</span>
    <span class="c1"># descents lose most of their precision before they start.</span>
    <span class="n">centred</span> <span class="o">=</span> <span class="n">centred</span> <span class="o">/</span> <span class="n">centred</span><span class="o">.</span><span class="n">std</span><span class="p">()</span>
    <span class="n">reduced</span> <span class="o">=</span> <span class="n">pca_image</span><span class="p">(</span><span class="n">centred</span><span class="p">,</span> <span class="n">n_features</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">windows</span> <span class="o">=</span> <span class="n">sliding_window_vectorize</span><span class="p">(</span><span class="n">reduced</span><span class="p">,</span> <span class="n">window_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">,</span> <span class="n">backend</span><span class="p">)</span>
    <span class="n">truth</span> <span class="o">=</span> <span class="n">crop_labels</span><span class="p">(</span><span class="n">labels</span><span class="p">,</span> <span class="n">window_size</span><span class="p">,</span> <span class="n">stride</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">windows</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">labels</span><span class="o">.</span><span class="n">shape</span><span class="p">,</span> <span class="n">n_classes</span>


<span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">make_mc_parser</span><span class="p">(</span>
        <span class="s2">&quot;Riemannian K-means segmentation of a hyperspectral scene, with and &quot;</span>
        <span class="s2">&quot;without the random-matrix-theory correction.&quot;</span>
    <span class="p">)</span>
    <span class="n">add_mc_base_args</span><span class="p">(</span><span class="n">parser</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--scene&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;salinas&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Scene to segment: indianpines or salinas.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--data_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;data/hyperspectral&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Where the .mat files live; downloaded there if missing.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_features&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Principal components kept. Five is the configuration of the &quot;</span>
             <span class="s2">&quot;published table, and a handful of directions represent these &quot;</span>
             <span class="s2">&quot;scenes well.&quot;</span><span class="p">,</span>
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
        <span class="s2">&quot;--n_init&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Restarts of the K-means; the one of least inertia is kept.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--max_iter&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">30</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Assignment/re-estimation rounds per restart.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--mean_iterations&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Iteration budget of one Fréchet mean.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--seeds&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Repeat the whole comparison on several starting partitions and &quot;</span>
             <span class="s2">&quot;report the mean and spread across them; defaults to the single &quot;</span>
             <span class="s2">&quot;--seed. Accepts either form: &#39;--seeds 42 123 456&#39; or &quot;</span>
             <span class="s2">&quot;&#39;--seeds 42,123,456&#39;. The comma form is the one to use through &quot;</span>
             <span class="s2">&quot;qanat, which passes only the first token of a multi-value &quot;</span>
             <span class="s2">&quot;argument and turns the rest into positionals.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--methods&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="nb">list</span><span class="p">(</span><span class="n">METHODS</span><span class="p">),</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Subset of the four methods to run.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--figure_width&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.23</span><span class="se">\\</span><span class="s2">textwidth&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of one map in the exported figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>
    <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">export_path</span>

    <span class="c1"># Not init_logging: its GPU disclaimer is written for the Monte-Carlo</span>
    <span class="c1"># experiments, whose batched path walks T checkpoints sequentially. Nothing</span>
    <span class="c1"># here is a Monte-Carlo trial, so the warning would be misleading.</span>
    <span class="n">logging</span><span class="o">.</span><span class="n">basicConfig</span><span class="p">(</span><span class="n">level</span><span class="o">=</span><span class="n">logging</span><span class="o">.</span><span class="n">INFO</span><span class="p">,</span> <span class="nb">format</span><span class="o">=</span><span class="s2">&quot;</span><span class="si">%(levelname)s</span><span class="s2"> </span><span class="si">%(message)s</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>
    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">download_scene</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">data_path</span><span class="p">)</span>
    <span class="n">windows</span><span class="p">,</span> <span class="n">truth</span><span class="p">,</span> <span class="n">image_shape</span><span class="p">,</span> <span class="n">n_classes</span> <span class="o">=</span> <span class="n">prepare</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">data_path</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="p">,</span>
        <span class="n">args</span><span class="o">.</span><span class="n">stride</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">concentration</span> <span class="o">=</span> <span class="n">args</span><span class="o">.</span><span class="n">n_features</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="o">**</span><span class="mi">2</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">windows</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="si">}</span><span class="s2"> fenêtres de &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">windows</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="si">}</span><span class="s2"> échantillons en dimension </span><span class="si">{</span><span class="n">windows</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;c = </span><span class="si">{</span><span class="n">concentration</span><span class="si">:</span><span class="s2">.2f</span><span class="si">}</span><span class="s2">, </span><span class="si">{</span><span class="n">n_classes</span><span class="si">}</span><span class="s2"> classes&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">seeds</span> <span class="o">=</span> <span class="p">(</span>
        <span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">value</span><span class="p">)</span> <span class="k">for</span> <span class="n">token</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">seeds</span> <span class="k">for</span> <span class="n">value</span> <span class="ow">in</span> <span class="n">token</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">&quot;,&quot;</span><span class="p">)</span> <span class="k">if</span> <span class="n">value</span><span class="p">]</span>
        <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">seeds</span> <span class="k">else</span> <span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">]</span>
    <span class="p">)</span>
    <span class="c1"># One step per estimator per seed, which is the coarsest unit that still</span>
    <span class="c1"># moves often enough to be worth watching: the corrected method alone takes</span>
    <span class="c1"># a quarter of an hour on Salinas.</span>
    <span class="n">progress</span> <span class="o">=</span> <span class="n">Progress</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">seeds</span><span class="p">)</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">methods</span><span class="p">),</span>
        <span class="n">description</span><span class="o">=</span><span class="s2">&quot;Estimators x seeds&quot;</span><span class="p">,</span> <span class="n">unit</span><span class="o">=</span><span class="s2">&quot;fits&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">maps</span><span class="p">,</span> <span class="n">scores</span><span class="p">,</span> <span class="n">per_seed</span> <span class="o">=</span> <span class="p">{},</span> <span class="p">{},</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">seed</span> <span class="ow">in</span> <span class="n">seeds</span><span class="p">:</span>
      <span class="k">for</span> <span class="n">method</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">methods</span><span class="p">:</span>
        <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span>
        <span class="n">labels</span><span class="p">,</span> <span class="n">inertia</span><span class="p">,</span> <span class="n">histories</span> <span class="o">=</span> <span class="n">riemannian_kmeans</span><span class="p">(</span>
            <span class="n">windows</span><span class="p">,</span> <span class="n">n_classes</span><span class="p">,</span> <span class="n">method</span><span class="o">=</span><span class="n">method</span><span class="p">,</span> <span class="n">n_init</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_init</span><span class="p">,</span>
            <span class="n">max_iter</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">max_iter</span><span class="p">,</span> <span class="n">mean_iterations</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">mean_iterations</span><span class="p">,</span>
            <span class="n">seed</span><span class="o">=</span><span class="n">seed</span><span class="p">,</span> <span class="n">backend</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">backend</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">segmented</span> <span class="o">=</span> <span class="n">unvectorize_labels</span><span class="p">(</span>
            <span class="n">labels</span><span class="p">,</span> <span class="o">*</span><span class="n">image_shape</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">window_size</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">stride</span>
        <span class="p">)</span>
        <span class="n">matched</span> <span class="o">=</span> <span class="n">match_labels</span><span class="p">(</span><span class="n">segmented</span><span class="p">,</span> <span class="n">truth</span><span class="p">)</span>
        <span class="n">accuracy</span> <span class="o">=</span> <span class="n">clustering_accuracy</span><span class="p">(</span><span class="n">matched</span><span class="p">,</span> <span class="n">truth</span><span class="p">)</span>
        <span class="n">ious</span><span class="p">,</span> <span class="n">miou</span> <span class="o">=</span> <span class="n">mean_iou</span><span class="p">(</span><span class="n">matched</span><span class="p">,</span> <span class="n">truth</span><span class="p">)</span>
        <span class="n">elapsed</span> <span class="o">=</span> <span class="n">time</span><span class="o">.</span><span class="n">perf_counter</span><span class="p">()</span> <span class="o">-</span> <span class="n">start</span>

        <span class="n">per_seed</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
            <span class="s2">&quot;seed&quot;</span><span class="p">:</span> <span class="n">seed</span><span class="p">,</span> <span class="s2">&quot;method&quot;</span><span class="p">:</span> <span class="n">method</span><span class="p">,</span>
            <span class="s2">&quot;accuracy&quot;</span><span class="p">:</span> <span class="n">accuracy</span><span class="p">,</span> <span class="s2">&quot;mIoU&quot;</span><span class="p">:</span> <span class="n">miou</span><span class="p">,</span>
            <span class="s2">&quot;inertia&quot;</span><span class="p">:</span> <span class="n">inertia</span><span class="p">,</span> <span class="s2">&quot;seconds&quot;</span><span class="p">:</span> <span class="n">elapsed</span><span class="p">,</span>
            <span class="s2">&quot;worst_moved&quot;</span><span class="p">:</span> <span class="nb">max</span><span class="p">(</span><span class="n">h</span><span class="p">[</span><span class="s2">&quot;moved&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">h</span> <span class="ow">in</span> <span class="n">histories</span><span class="p">),</span>
        <span class="p">})</span>
        <span class="c1"># The maps and the figure show the first seed; the table below is what</span>
        <span class="c1"># carries the comparison when there is more than one.</span>
        <span class="k">if</span> <span class="n">seed</span> <span class="o">==</span> <span class="n">seeds</span><span class="p">[</span><span class="mi">0</span><span class="p">]:</span>
            <span class="n">maps</span><span class="p">[</span><span class="n">method</span><span class="p">]</span> <span class="o">=</span> <span class="n">matched</span>
            <span class="n">scores</span><span class="p">[</span><span class="n">method</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span>
                <span class="s2">&quot;accuracy&quot;</span><span class="p">:</span> <span class="n">accuracy</span><span class="p">,</span> <span class="s2">&quot;mIoU&quot;</span><span class="p">:</span> <span class="n">miou</span><span class="p">,</span>
                <span class="s2">&quot;inertia&quot;</span><span class="p">:</span> <span class="n">inertia</span><span class="p">,</span> <span class="s2">&quot;seconds&quot;</span><span class="p">:</span> <span class="n">elapsed</span><span class="p">,</span>
                <span class="s2">&quot;restarts&quot;</span><span class="p">:</span> <span class="n">histories</span><span class="p">,</span>
                <span class="s2">&quot;worst_moved&quot;</span><span class="p">:</span> <span class="nb">max</span><span class="p">(</span><span class="n">h</span><span class="p">[</span><span class="s2">&quot;moved&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">h</span> <span class="ow">in</span> <span class="n">histories</span><span class="p">),</span>
            <span class="p">}</span>
        <span class="c1"># Written after every method so a run killed part way still leaves a</span>
        <span class="c1"># readable table of what it did finish.</span>
        <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;seeds.json&quot;</span><span class="p">),</span> <span class="s2">&quot;w&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">handle</span><span class="p">:</span>
            <span class="n">json</span><span class="o">.</span><span class="n">dump</span><span class="p">(</span><span class="n">per_seed</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">indent</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
        <span class="n">progress</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">method</span><span class="si">:</span><span class="s2">6s</span><span class="si">}</span><span class="s2"> acc=</span><span class="si">{</span><span class="n">accuracy</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">  mIoU=</span><span class="si">{</span><span class="n">miou</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">  &quot;</span>
              <span class="sa">f</span><span class="s2">&quot;(</span><span class="si">{</span><span class="n">elapsed</span><span class="si">:</span><span class="s2">.0f</span><span class="si">}</span><span class="s2">s, worst restart left &quot;</span>
              <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">scores</span><span class="p">[</span><span class="n">method</span><span class="p">][</span><span class="s1">&#39;worst_moved&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.2%</span><span class="si">}</span><span class="s2"> moving)&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>


    <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">seeds</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">1</span><span class="p">:</span>
        <span class="c1"># Mean and spread over the seeds, and how often each method actually</span>
        <span class="c1"># came first: a mean can hide the difference between a method that wins</span>
        <span class="c1"># narrowly every time and one that wins once by a lot.</span>
        <span class="n">summary</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="se">\n</span><span class="si">{</span><span class="s1">&#39;method&#39;</span><span class="si">:</span><span class="s2">7s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;accuracy&#39;</span><span class="si">:</span><span class="s2">&gt;18s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;mIoU&#39;</span><span class="si">:</span><span class="s2">&gt;18s</span><span class="si">}</span><span class="s2">   wins&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">method</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">methods</span><span class="p">:</span>
            <span class="n">rows</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">per_seed</span> <span class="k">if</span> <span class="n">r</span><span class="p">[</span><span class="s2">&quot;method&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">method</span><span class="p">]</span>
            <span class="n">acc</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span><span class="p">[</span><span class="s2">&quot;accuracy&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span><span class="p">]</span>
            <span class="n">iou</span> <span class="o">=</span> <span class="p">[</span><span class="n">r</span><span class="p">[</span><span class="s2">&quot;mIoU&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">rows</span><span class="p">]</span>
            <span class="n">wins</span> <span class="o">=</span> <span class="nb">sum</span><span class="p">(</span>
                <span class="nb">max</span><span class="p">((</span><span class="n">r</span> <span class="k">for</span> <span class="n">r</span> <span class="ow">in</span> <span class="n">per_seed</span> <span class="k">if</span> <span class="n">r</span><span class="p">[</span><span class="s2">&quot;seed&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">s</span><span class="p">),</span>
                    <span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">r</span><span class="p">:</span> <span class="n">r</span><span class="p">[</span><span class="s2">&quot;accuracy&quot;</span><span class="p">])[</span><span class="s2">&quot;method&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">method</span>
                <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="n">seeds</span>
            <span class="p">)</span>
            <span class="n">summary</span><span class="p">[</span><span class="n">method</span><span class="p">]</span> <span class="o">=</span> <span class="p">{</span>
                <span class="s2">&quot;accuracy_mean&quot;</span><span class="p">:</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">acc</span><span class="p">)),</span>
                <span class="s2">&quot;accuracy_std&quot;</span><span class="p">:</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">acc</span><span class="p">,</span> <span class="n">ddof</span><span class="o">=</span><span class="mi">1</span><span class="p">))</span> <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">acc</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">1</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">,</span>
                <span class="s2">&quot;mIoU_mean&quot;</span><span class="p">:</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">iou</span><span class="p">)),</span>
                <span class="s2">&quot;mIoU_std&quot;</span><span class="p">:</span> <span class="nb">float</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">iou</span><span class="p">,</span> <span class="n">ddof</span><span class="o">=</span><span class="mi">1</span><span class="p">))</span> <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">iou</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">1</span> <span class="k">else</span> <span class="mf">0.0</span><span class="p">,</span>
                <span class="s2">&quot;wins&quot;</span><span class="p">:</span> <span class="n">wins</span><span class="p">,</span> <span class="s2">&quot;n_seeds&quot;</span><span class="p">:</span> <span class="nb">len</span><span class="p">(</span><span class="n">rows</span><span class="p">),</span>
            <span class="p">}</span>
            <span class="n">entry</span> <span class="o">=</span> <span class="n">summary</span><span class="p">[</span><span class="n">method</span><span class="p">]</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">method</span><span class="si">:</span><span class="s2">7s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">entry</span><span class="p">[</span><span class="s1">&#39;accuracy_mean&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">8.4f</span><span class="si">}</span><span class="s2"> ± &quot;</span>
                  <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">entry</span><span class="p">[</span><span class="s1">&#39;accuracy_std&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">entry</span><span class="p">[</span><span class="s1">&#39;mIoU_mean&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">8.4f</span><span class="si">}</span><span class="s2"> ± &quot;</span>
                  <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">entry</span><span class="p">[</span><span class="s1">&#39;mIoU_std&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2">   </span><span class="si">{</span><span class="n">wins</span><span class="si">}</span><span class="s2">/</span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">seeds</span><span class="p">)</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;summary.json&quot;</span><span class="p">),</span> <span class="s2">&quot;w&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">handle</span><span class="p">:</span>
            <span class="n">json</span><span class="o">.</span><span class="n">dump</span><span class="p">({</span><span class="s2">&quot;scene&quot;</span><span class="p">:</span> <span class="n">args</span><span class="o">.</span><span class="n">scene</span><span class="p">,</span> <span class="s2">&quot;seeds&quot;</span><span class="p">:</span> <span class="n">seeds</span><span class="p">,</span>
                       <span class="s2">&quot;per_seed&quot;</span><span class="p">:</span> <span class="n">per_seed</span><span class="p">,</span> <span class="s2">&quot;summary&quot;</span><span class="p">:</span> <span class="n">summary</span><span class="p">},</span>
                      <span class="n">handle</span><span class="p">,</span> <span class="n">indent</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>

    <span class="c1"># ── the figure: ground truth, then one map per method ────────────────</span>
    <span class="n">panels</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;vérité terrain&quot;</span><span class="p">]</span> <span class="o">+</span> <span class="nb">list</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">methods</span><span class="p">)</span>
    <span class="n">figure</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span>
        <span class="mi">1</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">panels</span><span class="p">),</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">2.0</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">panels</span><span class="p">),</span> <span class="mf">2.4</span><span class="p">)</span>
    <span class="p">)</span>
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
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">panel</span><span class="si">}</span><span class="se">\\</span><span class="s2">n</span><span class="si">{</span><span class="n">scores</span><span class="p">[</span><span class="n">panel</span><span class="p">][</span><span class="s1">&#39;accuracy&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2"> / &quot;</span>
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
            <span class="n">n_classes</span><span class="o">=</span><span class="n">n_classes</span><span class="p">,</span> <span class="n">concentration</span><span class="o">=</span><span class="n">concentration</span><span class="p">,</span>
            <span class="n">truth</span><span class="o">=</span><span class="n">truth</span><span class="p">,</span> <span class="o">**</span><span class="p">{</span><span class="sa">f</span><span class="s2">&quot;map_</span><span class="si">{</span><span class="n">m</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">:</span> <span class="n">maps</span><span class="p">[</span><span class="n">m</span><span class="p">]</span> <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">methods</span><span class="p">},</span>
        <span class="p">)</span>
        <span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;scores.json&quot;</span><span class="p">),</span> <span class="s2">&quot;w&quot;</span><span class="p">)</span> <span class="k">as</span> <span class="n">handle</span><span class="p">:</span>
            <span class="n">json</span><span class="o">.</span><span class="n">dump</span><span class="p">(</span><span class="n">scores</span><span class="p">,</span> <span class="n">handle</span><span class="p">,</span> <span class="n">indent</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
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
<p class="param-help">Scene to segment: indianpines or salinas.</p>
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
<p class="param-help">Principal components kept. Five is the configuration of the published table, and a handful of directions represent these scenes well.</p>
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
<span class="param-flag">--n_init</span><span class="param-type">int</span><span class="param-default">default <b>5</b></span>
</div>
<p class="param-help">Restarts of the K-means; the one of least inertia is kept.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--max_iter</span><span class="param-type">int</span><span class="param-default">default <b>30</b></span>
</div>
<p class="param-help">Assignment/re-estimation rounds per restart.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--mean_iterations</span><span class="param-type">int</span><span class="param-default">default <b>50</b></span>
</div>
<p class="param-help">Iteration budget of one Fréchet mean.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seeds</span><span class="param-type">str</span>
</div>
<p class="param-help">Repeat the whole comparison on several starting partitions and report the mean and spread across them; defaults to the single --seed. Accepts either form: &#x27;--seeds 42 123 456&#x27; or &#x27;--seeds 42,123,456&#x27;. The comma form is the one to use through qanat, which passes only the first token of a multi-value argument and turns the rest into positionals.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--methods</span><span class="param-type">str</span>
</div>
<p class="param-help">Subset of the four methods to run.</p>
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

`3-learning/experiments/learning_hyperspectral_rmt.yaml`

<a class="back-link" href="../../chapters/3-learning/">← All experiments in 3 · Learning</a>
