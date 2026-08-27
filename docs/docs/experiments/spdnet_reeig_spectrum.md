<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/4-deeplearning/">4 · Deep Learning</a>
<span class="sep">/</span>
<span class="here">spdnet_reeig_spectrum</span>
</nav>

# spdnet_reeig_spectrum

What the ReEig threshold does to the spectrum of a CovPool matrix, and the 1/eps bound it puts on the Loewner factor of the backward pass

**Tags:** `deeplearning`  `spdnet`  `reeig`  `covariance`

## Run

```sh
uv run python 4-deeplearning/reeig_spectrum/main.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/4-deeplearning/reeig_spectrum/main.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">325 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">4-deeplearning/reeig_spectrum/main.py</p>
<div class="highlight"><pre><span></span><span class="c1"># What ReEig does to the spectrum, and what it buys the backward pass.</span>
<span class="c1">#</span>
<span class="c1"># The CovPool layer of eq:spdnet-covpool is an empirical covariance estimated</span>
<span class="c1"># from n_pixels positions for n_filters channels. In the setting of</span>
<span class="c1"># rem:spdnet-covpool-regime that ratio is about three (n_filters = 8 x 32 = 256</span>
<span class="c1"># against n_pixels = 38 x 20 = 760), which is a *low* sampling regime: the</span>
<span class="c1"># smallest eigenvalues of such a matrix are far below those of the covariance</span>
<span class="c1"># it estimates, and it is those that the rest of the network has to survive.</span>
<span class="c1">#</span>
<span class="c1"># Two things are measured against the sampling ratio, both on the same</span>
<span class="c1"># simulated matrices:</span>
<span class="c1">#</span>
<span class="c1">#   (a) the spectrum itself, with the ReEig threshold drawn across it, which</span>
<span class="c1">#       says how much of the spectrum the layer actually rectifies;</span>
<span class="c1">#   (b) the largest entry of the Loewner matrix of prop:spdnet-diffm for the</span>
<span class="c1">#       LogEig that follows, which is the factor by which backpropagating</span>
<span class="c1">#       through the spectral layers multiplies the incoming error.</span>
<span class="c1">#</span>
<span class="c1"># The point of the second panel: that factor is 1/lambda_min, so it diverges as</span>
<span class="c1"># the sampling ratio drops, and a ReEig layer of threshold eps caps it at</span>
<span class="c1"># exactly 1/eps. This is the sense in which ReEig is a spectral regularisation</span>
<span class="c1"># of the same family as the shrinkage of the second part of the dissertation</span>
<span class="c1"># (rem:spdnet-reeig-retrecissement) — it pays a bias on the small eigenvalues</span>
<span class="c1"># to buy a bound on the gradient.</span>
<span class="c1">#</span>
<span class="c1"># Simulated data only; the same two measurements are run on the datasets of the</span>
<span class="c1"># chapter by real_data.py.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">common</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">decaying_covariance</span><span class="p">,</span>
    <span class="n">resolve_device</span><span class="p">,</span>
    <span class="n">sample_covpool</span><span class="p">,</span>
    <span class="n">spectral_summary</span><span class="p">,</span>
<span class="p">)</span>


<span class="k">def</span><span class="w"> </span><span class="nf">sweep</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;Measure the summary of common.py at every (decay, ratio) pair.</span>

<span class="sd">    Two axes rather than one, because the two of them drive the instability</span>
<span class="sd">    independently: the sampling ratio sets how far the empirical spectrum falls</span>
<span class="sd">    below the true one, and the decay sets where the true one already was. The</span>
<span class="sd">    condition numbers reported on the real datasets of the chapter (from</span>
<span class="sd">    9.1e5 on HDM05 to 1.2e7 on Rices90) are far beyond what a mild decay</span>
<span class="sd">    reaches at any ratio, so fixing the decay would answer a question the</span>
<span class="sd">    chapter is not asking.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="n">generator</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">Generator</span><span class="p">(</span><span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">)</span>
    <span class="n">generator</span><span class="o">.</span><span class="n">manual_seed</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seed</span><span class="p">)</span>

    <span class="n">covariances_by_decay</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="n">records</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">decay</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">decays</span><span class="p">:</span>
        <span class="n">true_covariance</span> <span class="o">=</span> <span class="n">decaying_covariance</span><span class="p">(</span>
            <span class="n">args</span><span class="o">.</span><span class="n">n_filters</span><span class="p">,</span> <span class="n">decay</span><span class="p">,</span> <span class="n">device</span><span class="o">=</span><span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">dtype</span>
        <span class="p">)</span>
        <span class="n">covariances_by_decay</span><span class="p">[</span><span class="n">decay</span><span class="p">]</span> <span class="o">=</span> <span class="n">true_covariance</span>
        <span class="k">for</span> <span class="n">ratio</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">ratios</span><span class="p">:</span>
            <span class="n">n_pixels</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="nb">int</span><span class="p">(</span><span class="nb">round</span><span class="p">(</span><span class="n">ratio</span> <span class="o">*</span> <span class="n">args</span><span class="o">.</span><span class="n">n_filters</span><span class="p">)),</span> <span class="mi">2</span><span class="p">)</span>
            <span class="n">covariances</span> <span class="o">=</span> <span class="n">sample_covpool</span><span class="p">(</span>
                <span class="n">true_covariance</span><span class="p">,</span> <span class="n">n_pixels</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="p">,</span> <span class="n">generator</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span>
            <span class="p">)</span>
            <span class="n">summary</span> <span class="o">=</span> <span class="n">spectral_summary</span><span class="p">(</span><span class="n">covariances</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">eps</span><span class="p">)</span>
            <span class="n">summary</span><span class="p">[</span><span class="s2">&quot;decay&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">decay</span>
            <span class="n">summary</span><span class="p">[</span><span class="s2">&quot;ratio&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">ratio</span>
            <span class="n">summary</span><span class="p">[</span><span class="s2">&quot;n_pixels&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">n_pixels</span>
            <span class="n">records</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">summary</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">covariances_by_decay</span><span class="p">,</span> <span class="n">records</span>


<span class="k">def</span><span class="w"> </span><span class="nf">median</span><span class="p">(</span><span class="n">values</span><span class="p">):</span>
    <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">torch</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">values</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">())</span>


<span class="k">def</span><span class="w"> </span><span class="nf">band</span><span class="p">(</span><span class="n">values</span><span class="p">,</span> <span class="n">low</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span> <span class="n">high</span><span class="o">=</span><span class="mf">0.95</span><span class="p">):</span>
    <span class="n">quantiles</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">quantile</span><span class="p">(</span>
        <span class="n">values</span><span class="o">.</span><span class="n">cpu</span><span class="p">(),</span> <span class="n">torch</span><span class="o">.</span><span class="n">tensor</span><span class="p">([</span><span class="n">low</span><span class="p">,</span> <span class="n">high</span><span class="p">],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">values</span><span class="o">.</span><span class="n">dtype</span><span class="p">)</span>
    <span class="p">)</span>
    <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">quantiles</span><span class="p">[</span><span class="mi">0</span><span class="p">]),</span> <span class="nb">float</span><span class="p">(</span><span class="n">quantiles</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Effect of the ReEig threshold on the spectrum of a CovPool matrix, &quot;</span>
        <span class="s2">&quot;and on the Loewner factor of the backward pass.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_filters&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of channels, i.e. the size of the SPD matrix. Default is &quot;</span>
             <span class="s2">&quot;the 8 x 32 = 256 of the SRCNet architecture.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--ratios&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span>
        <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mf">0.75</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">,</span> <span class="mf">1.5</span><span class="p">,</span> <span class="mf">2.0</span><span class="p">,</span> <span class="mf">3.0</span><span class="p">,</span> <span class="mf">5.0</span><span class="p">,</span> <span class="mf">8.0</span><span class="p">,</span> <span class="mf">12.0</span><span class="p">,</span> <span class="mf">20.0</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Sampling ratios n_pixels / n_filters to sweep. The architecture &quot;</span>
             <span class="s2">&quot;of the chapter sits at about 3.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--spectra_at&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mf">1.0</span><span class="p">,</span> <span class="mf">3.0</span><span class="p">,</span> <span class="mf">20.0</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Ratios whose full spectrum is drawn on the left panel. Kept to &quot;</span>
             <span class="s2">&quot;three so that the panel stays readable at 355 pt.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--decays&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mf">1e2</span><span class="p">,</span> <span class="mf">1e4</span><span class="p">,</span> <span class="mf">1e6</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Ratios between the largest and the smallest eigenvalue of the &quot;</span>
             <span class="s2">&quot;true covariance, which decays geometrically in between. The &quot;</span>
             <span class="s2">&quot;chapter&#39;s real data sit at the top of this range: the reported &quot;</span>
             <span class="s2">&quot;condition numbers run from 9.1e5 to 1.2e7.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--spectra_decay&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e4</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Which of the decays the left panel draws the spectra for.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--eps&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">1e-4</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;ReEig rectification threshold, the default of the layer.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--n_trials&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Number of CovPool matrices drawn at each ratio.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/reeig_spectrum&quot;</span><span class="p">,</span>
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
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Width of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--axis_height&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;4.6cm&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Height of a single panel in the exported PGFPlots figure.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--device&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;cpu&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Compute device: cpu or cuda. MPS is refused, see common.py.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Base seed.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>

    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">device</span> <span class="o">=</span> <span class="n">resolve_device</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">device</span><span class="p">)</span>
    <span class="n">dtype</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">float64</span>

    <span class="n">covariances_by_decay</span><span class="p">,</span> <span class="n">records</span> <span class="o">=</span> <span class="n">sweep</span><span class="p">(</span><span class="n">args</span><span class="p">,</span> <span class="n">device</span><span class="p">,</span> <span class="n">dtype</span><span class="p">)</span>

    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">))</span>

    <span class="c1"># ---- Panel (a): the spectra --------------------------------------------</span>
    <span class="c1"># One decay only, at several sampling ratios: the panel is about the gap</span>
    <span class="c1"># the sampling opens between the true spectrum and the estimated one, and</span>
    <span class="c1"># about where the ReEig threshold falls in that gap.</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">index</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">args</span><span class="o">.</span><span class="n">n_filters</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
    <span class="n">true_spectrum</span> <span class="o">=</span> <span class="p">(</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eigvalsh</span><span class="p">(</span><span class="n">covariances_by_decay</span><span class="p">[</span><span class="n">args</span><span class="o">.</span><span class="n">spectra_decay</span><span class="p">])</span>
        <span class="o">.</span><span class="n">flip</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span>
    <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
        <span class="n">index</span><span class="p">,</span> <span class="n">true_spectrum</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;k&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;:&quot;</span><span class="p">,</span>
        <span class="n">label</span><span class="o">=</span><span class="s2">&quot;covariance vraie&quot;</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="k">for</span> <span class="n">position</span><span class="p">,</span> <span class="n">ratio</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">spectra_at</span><span class="p">):</span>
        <span class="n">drawn</span> <span class="o">=</span> <span class="p">[</span>
            <span class="n">record</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span>
            <span class="k">if</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;ratio&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">ratio</span> <span class="ow">and</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;decay&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">args</span><span class="o">.</span><span class="n">spectra_decay</span>
        <span class="p">]</span>
        <span class="k">if</span> <span class="ow">not</span> <span class="n">drawn</span><span class="p">:</span>
            <span class="k">continue</span>
        <span class="n">spectra</span> <span class="o">=</span> <span class="n">drawn</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">&quot;eigenvalues&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">flip</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">()</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">index</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">spectra</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">),</span> <span class="n">color</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">position</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="sa">rf</span><span class="s2">&quot;$N_</span><span class="se">{{</span><span class="s2">pix</span><span class="se">}}</span><span class="s2">/N_</span><span class="se">{{</span><span class="s2">filtre</span><span class="se">}}</span><span class="s2"> = </span><span class="si">{</span><span class="n">ratio</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">axhline</span><span class="p">(</span>
        <span class="n">args</span><span class="o">.</span><span class="n">eps</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C3&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
        <span class="n">label</span><span class="o">=</span><span class="sa">rf</span><span class="s2">&quot;seuil $\varepsilon = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">eps</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">$&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;rang de la valeur propre&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;valeur propre&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="sa">rf</span><span class="s2">&quot;décroissance $10^</span><span class="se">{{</span><span class="si">{</span><span class="nb">int</span><span class="p">(</span><span class="nb">round</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">spectra_decay</span><span class="p">)))</span><span class="si">}</span><span class="se">}}</span><span class="s2">$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>

    <span class="c1"># ---- Panel (b): what the backward pass pays ----------------------------</span>
    <span class="c1"># One curve per decay without ReEig, all of them flattened onto the same</span>
    <span class="c1"># 1/eps ceiling once the layer is applied. Drawing the rectified curves as</span>
    <span class="c1"># dashed rather than as a second family of colours keeps the panel readable</span>
    <span class="c1"># at 355 pt.</span>
    <span class="n">ax</span> <span class="o">=</span> <span class="n">axes</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">all_ratios</span> <span class="o">=</span> <span class="nb">sorted</span><span class="p">({</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;ratio&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">})</span>

    <span class="c1"># Below a ratio of one the CovPool matrix is rank deficient — the centring</span>
    <span class="c1"># of eq:spdnet-covpool caps its rank at n_pixels - 1 — so lambda_min is zero</span>
    <span class="c1"># up to rounding, LogEig is undefined and the Loewner factor has no value to</span>
    <span class="c1"># plot. Those ratios are shaded rather than drawn, because &quot;no finite value&quot;</span>
    <span class="c1"># is the finding, not a large value.</span>
    <span class="n">singular</span> <span class="o">=</span> <span class="p">[</span>
        <span class="n">record</span><span class="p">[</span><span class="s2">&quot;ratio&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span> <span class="k">if</span> <span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;lambda_min&quot;</span><span class="p">])</span> <span class="o">&lt;=</span> <span class="mi">0</span>
    <span class="p">]</span>
    <span class="k">if</span> <span class="n">singular</span><span class="p">:</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">axvspan</span><span class="p">(</span>
            <span class="nb">min</span><span class="p">(</span><span class="n">all_ratios</span><span class="p">),</span> <span class="nb">max</span><span class="p">(</span><span class="n">singular</span><span class="p">)</span> <span class="o">*</span> <span class="mf">1.05</span><span class="p">,</span>
            <span class="n">color</span><span class="o">=</span><span class="s2">&quot;0.85&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span>
        <span class="p">)</span>

    <span class="k">for</span> <span class="n">position</span><span class="p">,</span> <span class="n">decay</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">decays</span><span class="p">):</span>
        <span class="n">regular</span> <span class="o">=</span> <span class="p">[</span>
            <span class="n">record</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span>
            <span class="k">if</span> <span class="n">record</span><span class="p">[</span><span class="s2">&quot;decay&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="n">decay</span> <span class="ow">and</span> <span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;lambda_min&quot;</span><span class="p">])</span> <span class="o">&gt;</span> <span class="mi">0</span>
        <span class="p">]</span>
        <span class="n">exponent</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">round</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">log10</span><span class="p">(</span><span class="n">decay</span><span class="p">)))</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="p">[</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;ratio&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">regular</span><span class="p">],</span>
            <span class="p">[</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;loewner_max&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">regular</span><span class="p">],</span>
            <span class="n">color</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">position</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.4</span><span class="p">,</span> <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="sa">rf</span><span class="s2">&quot;$10^</span><span class="se">{{</span><span class="si">{</span><span class="n">exponent</span><span class="si">}</span><span class="se">}}</span><span class="s2">$, sans \textsc</span><span class="se">{{</span><span class="s2">reeig</span><span class="se">}}</span><span class="s2">&quot;</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">ax</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="p">[</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;ratio&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">regular</span><span class="p">],</span>
            <span class="p">[</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;loewner_max_reeig&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">regular</span><span class="p">],</span>
            <span class="n">color</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">position</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.2</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
            <span class="n">label</span><span class="o">=</span><span class="sa">rf</span><span class="s2">&quot;$10^</span><span class="se">{{</span><span class="si">{</span><span class="n">exponent</span><span class="si">}</span><span class="se">}}</span><span class="s2">$, avec \textsc</span><span class="se">{{</span><span class="s2">reeig</span><span class="se">}}</span><span class="s2">&quot;</span><span class="p">,</span>
        <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">axhline</span><span class="p">(</span>
        <span class="mf">1.0</span> <span class="o">/</span> <span class="n">args</span><span class="o">.</span><span class="n">eps</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s2">&quot;C3&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span> <span class="n">zorder</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span>
        <span class="n">label</span><span class="o">=</span><span class="sa">r</span><span class="s2">&quot;borne $1/\varepsilon$&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$N_</span><span class="si">{pix}</span><span class="s2">/N_</span><span class="si">{filtre}</span><span class="s2">$&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;$\max_</span><span class="si">{ij}</span><span class="s2">|\mathbf</span><span class="si">{G}</span><span class="s2">_</span><span class="si">{ij}</span><span class="s2">|$ de \textsc</span><span class="si">{logeig}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">ax</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>

    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="c1"># ---- A short digest on stdout, which is what qanat keeps ---------------</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;n_filters = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_filters</span><span class="si">}</span><span class="s2">, eps = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">eps</span><span class="si">:</span><span class="s2">g</span><span class="si">}</span><span class="s2">, &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;n_trials = </span><span class="si">{</span><span class="n">args</span><span class="o">.</span><span class="n">n_trials</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;decay&#39;</span><span class="si">:</span><span class="s2">&gt;9s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;ratio&#39;</span><span class="si">:</span><span class="s2">&gt;7s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;n_pix&#39;</span><span class="si">:</span><span class="s2">&gt;7s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;lambda_min&#39;</span><span class="si">:</span><span class="s2">&gt;12s</span><span class="si">}</span><span class="s2"> &quot;</span>
          <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;cond&#39;</span><span class="si">:</span><span class="s2">&gt;11s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;</span><span class="si">% e</span><span class="s1">cretees&#39;</span><span class="si">:</span><span class="s2">&gt;11s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;Loewner&#39;</span><span class="si">:</span><span class="s2">&gt;11s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;+ReEig&#39;</span><span class="si">:</span><span class="s2">&gt;11s</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">:</span>
        <span class="n">lambda_min</span> <span class="o">=</span> <span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;lambda_min&quot;</span><span class="p">])</span>
        <span class="n">prefix</span> <span class="o">=</span> <span class="p">(</span>
            <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;decay&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">9.0e</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;ratio&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">7.2f</span><span class="si">}</span><span class="s2"> &quot;</span>
            <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;n_pixels&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">7d</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">lambda_min</span><span class="si">:</span><span class="s2">12.3e</span><span class="si">}</span><span class="s2"> &quot;</span>
        <span class="p">)</span>
        <span class="c1"># A non-positive lambda_min means the matrix is singular; the condition</span>
        <span class="c1"># number and the Loewner factor are then meaningless rather than large,</span>
        <span class="c1"># and printing a number there would invite reading one.</span>
        <span class="k">if</span> <span class="n">lambda_min</span> <span class="o">&lt;=</span> <span class="mi">0</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span>
                <span class="n">prefix</span> <span class="o">+</span> <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;singuliere&#39;</span><span class="si">:</span><span class="s2">&gt;11s</span><span class="si">}</span><span class="s2"> &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="mi">100</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;fraction_clamped&#39;</span><span class="p">])</span><span class="si">:</span><span class="s2">10.1f</span><span class="si">}</span><span class="s2">% &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;non defini&#39;</span><span class="si">:</span><span class="s2">&gt;11s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;loewner_max_reeig&#39;</span><span class="p">])</span><span class="si">:</span><span class="s2">11.3e</span><span class="si">}</span><span class="s2">&quot;</span>
            <span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span>
                <span class="n">prefix</span> <span class="o">+</span> <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;condition&#39;</span><span class="p">])</span><span class="si">:</span><span class="s2">11.3e</span><span class="si">}</span><span class="s2"> &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="mi">100</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;fraction_clamped&#39;</span><span class="p">])</span><span class="si">:</span><span class="s2">10.1f</span><span class="si">}</span><span class="s2">% &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;loewner_max&#39;</span><span class="p">])</span><span class="si">:</span><span class="s2">11.3e</span><span class="si">}</span><span class="s2"> &quot;</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s1">&#39;loewner_max_reeig&#39;</span><span class="p">])</span><span class="si">:</span><span class="s2">11.3e</span><span class="si">}</span><span class="s2">&quot;</span>
            <span class="p">)</span>

    <span class="c1"># Raw measurements next to the figure, so that a qanat action can redraw</span>
    <span class="c1"># without re-running the sweep. Only the medians and the band are kept: the</span>
    <span class="c1"># per-trial spectra are a few hundred megabytes at the default settings and</span>
    <span class="c1"># nothing downstream reads them.</span>
    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">decays</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;decay&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">ratios</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;ratio&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">n_pixels</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;n_pixels&quot;</span><span class="p">]</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">lambda_min</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;lambda_min&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">condition</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;condition&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">fraction_clamped</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span>
            <span class="p">[</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;fraction_clamped&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]</span>
        <span class="p">),</span>
        <span class="n">loewner_max</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;loewner_max&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]),</span>
        <span class="n">loewner_max_reeig</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span>
            <span class="p">[</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;loewner_max_reeig&quot;</span><span class="p">])</span> <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span><span class="p">]</span>
        <span class="p">),</span>
        <span class="n">spectra_median</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">stack</span><span class="p">(</span>
            <span class="p">[</span>
                <span class="n">np</span><span class="o">.</span><span class="n">median</span><span class="p">(</span><span class="n">record</span><span class="p">[</span><span class="s2">&quot;eigenvalues&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">flip</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">cpu</span><span class="p">()</span><span class="o">.</span><span class="n">numpy</span><span class="p">(),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
                <span class="k">for</span> <span class="n">record</span> <span class="ow">in</span> <span class="n">records</span>
            <span class="p">]</span>
        <span class="p">),</span>
        <span class="n">eps</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">eps</span><span class="p">,</span>
        <span class="n">n_filters</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_filters</span><span class="p">,</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">figure_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;reeig_spectrum.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">figure_path</span><span class="p">,</span>
            <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span>
            <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span><span class="p">,</span>
        <span class="p">)</span>
        <span class="n">write_prov_sidecar</span><span class="p">(</span><span class="n">figure_path</span><span class="p">,</span> <span class="n">args</span><span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</details>

## Parameters

<div class="params">
<div class="param">
<div class="param-head">
<span class="param-flag">--n_filters</span><span class="param-type">int</span><span class="param-default">default <b>256</b></span>
</div>
<p class="param-help">Number of channels, i.e. the size of the SPD matrix. Default is the 8 x 32 = 256 of the SRCNet architecture.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--ratios</span><span class="param-type">float</span><span class="param-default">default <b>[0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]</b></span>
</div>
<p class="param-help">Sampling ratios n_pixels / n_filters to sweep. The architecture of the chapter sits at about 3.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--spectra_at</span><span class="param-type">float</span><span class="param-default">default <b>[1.0, 3.0, 20.0]</b></span>
</div>
<p class="param-help">Ratios whose full spectrum is drawn on the left panel. Kept to three so that the panel stays readable at 355 pt.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--decays</span><span class="param-type">float</span><span class="param-default">default <b>[100.0, 10000.0, 1000000.0]</b></span>
</div>
<p class="param-help">Ratios between the largest and the smallest eigenvalue of the true covariance, which decays geometrically in between. The chapter&#x27;s real data sit at the top of this range: the reported condition numbers run from 9.1e5 to 1.2e7.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--spectra_decay</span><span class="param-type">float</span><span class="param-default">default <b>10000.0</b></span>
</div>
<p class="param-help">Which of the decays the left panel draws the spectra for.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--eps</span><span class="param-type">float</span><span class="param-default">default <b>0.0001</b></span>
</div>
<p class="param-help">ReEig rectification threshold, the default of the layer.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n_trials</span><span class="param-type">int</span><span class="param-default">default <b>200</b></span>
</div>
<p class="param-help">Number of CovPool matrices drawn at each ratio.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/reeig_spectrum</b></span>
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
<p class="param-help">Width of a single panel in the exported PGFPlots figure.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_height</span><span class="param-type">str</span><span class="param-default">default <b>4.6cm</b></span>
</div>
<p class="param-help">Height of a single panel in the exported PGFPlots figure.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--device</span><span class="param-type">str</span><span class="param-default">default <b>cpu</b></span>
</div>
<p class="param-help">Compute device: cpu or cuda. MPS is refused, see common.py.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">Base seed.</p>
</div>
</div>

## Config

`4-deeplearning/experiments/spdnet_reeig_spectrum.yaml`

<a class="back-link" href="../../chapters/4-deeplearning/">← All experiments in 4 · Deep Learning</a>
