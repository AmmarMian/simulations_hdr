<nav class="crumbs" aria-label="Breadcrumb">
<a href="../../experiments-overview/">Experiments</a>
<span class="sep">/</span>
<a href="../../chapters/4-deeplearning/">4 · Deep Learning</a>
<span class="sep">/</span>
<span class="here">spdnet_wishart_model</span>
</nav>

# spdnet_wishart_model

Does the batch-norm mean matched to the Wishart model win, and by how much, as the degrees of freedom grow

**Tags:** `deeplearning`  `spdnet`  `batchnorm`  `wishart`

## Run

```sh
uv run python 4-deeplearning/wishart_model/df_sweep.py
```

<div class="src-bar">
<a class="src-btn" href="https://github.com/AmmarMian/simulations_hdr/blob/main/4-deeplearning/wishart_model/df_sweep.py" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg><span>View on GitHub</span></a>
</div>
<details class="src-view">
<summary><span class="src-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg><span>Source code</span><span class="param-alias">187 lines</span><svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg></span></summary>
<div class="src-body">
<p class="src-path">4-deeplearning/wishart_model/df_sweep.py</p>
<div class="highlight"><pre><span></span><span class="c1"># Does the mean matched to the model win, and by how much?</span>
<span class="c1">#</span>
<span class="c1"># prop:spdnet-moyennes-frechet identifies each mean of the batch-norm layer with</span>
<span class="c1"># the Fréchet mean of a geometry or a divergence: arithmetic with the left</span>
<span class="c1"># Kullback-Leibler (Wishart), harmonic with the right one (inverse-Wishart), GAH</span>
<span class="c1"># with the symmetrised one. The chapter needs that read as a modelling statement</span>
<span class="c1"># rather than a numerical curiosity, and the only support it currently plans for</span>
<span class="c1"># is the F1 tables on three real datasets — an indirect argument, on data whose</span>
<span class="c1"># law is unknown.</span>
<span class="c1">#</span>
<span class="c1"># The wishart-inverse grid of eusipco_2026 tests it head on: draw the data from a</span>
<span class="c1"># Wishart, then from an inverse-Wishart, and see which mean wins. This adds the</span>
<span class="c1"># one axis that grid does not sweep — the degrees of freedom. As df grows the</span>
<span class="c1"># Wishart concentrates around its scale matrix and the choice of mean should</span>
<span class="c1"># matter less, so the mechanism should appear as a gradient rather than as two</span>
<span class="c1"># points. That is what makes it a *statement about the model* and not about one</span>
<span class="c1"># particular setting.</span>
<span class="c1">#</span>
<span class="c1"># Everything is reused from eusipco_2026: run_single_experiment does the data</span>
<span class="c1"># generation, the training and the evaluation. This file only sweeps and draws.</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">argparse</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">os</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">sys</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">collections</span><span class="w"> </span><span class="kn">import</span> <span class="n">defaultdict</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.exporter</span><span class="w"> </span><span class="kn">import</span> <span class="n">save_tikz</span><span class="p">,</span> <span class="n">write_prov_sidecar</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">hdrlib.core.plot_style</span><span class="w"> </span><span class="kn">import</span> <span class="n">apply_style</span>

<span class="k">try</span><span class="p">:</span>
    <span class="kn">from</span><span class="w"> </span><span class="nn">eusipco_2026.simulation.runner</span><span class="w"> </span><span class="kn">import</span> <span class="n">run_single_experiment</span>
    <span class="kn">from</span><span class="w"> </span><span class="nn">spdnet_datasets.synthetic</span><span class="w"> </span><span class="kn">import</span> <span class="n">ExperimentConfig</span>
<span class="k">except</span> <span class="ne">ImportError</span><span class="p">:</span>  <span class="c1"># pragma: no cover - environment guidance, not logic</span>
    <span class="n">sys</span><span class="o">.</span><span class="n">exit</span><span class="p">(</span>
        <span class="s2">&quot;eusipco_2026 and spdnet-datasets are needed and are not dependencies &quot;</span>
        <span class="s2">&quot;of this repository. See README.md in this directory.&quot;</span>
    <span class="p">)</span>

<span class="c1"># Wishart draws sit at &#39;large&#39;, inverse-Wishart draws at &#39;small&#39;; the generator</span>
<span class="c1"># of spdnet-datasets overloads discriminant_position to select the law.</span>
<span class="n">MODELS</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;large&quot;</span><span class="p">:</span> <span class="s2">&quot;Wishart&quot;</span><span class="p">,</span> <span class="s2">&quot;small&quot;</span><span class="p">:</span> <span class="s2">&quot;Wishart inverse&quot;</span><span class="p">}</span>

<span class="n">MEANS</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">&quot;arithmetic&quot;</span><span class="p">:</span> <span class="s2">&quot;arithmétique&quot;</span><span class="p">,</span>
    <span class="s2">&quot;harmonic&quot;</span><span class="p">:</span> <span class="s2">&quot;harmonique&quot;</span><span class="p">,</span>
    <span class="s2">&quot;geometric_arithmetic_harmonic&quot;</span><span class="p">:</span> <span class="sa">r</span><span class="s2">&quot;\textsc</span><span class="si">{gah}</span><span class="s2">&quot;</span><span class="p">,</span>
    <span class="s2">&quot;adaptive_geometric_arithmetic_harmonic&quot;</span><span class="p">:</span> <span class="sa">r</span><span class="s2">&quot;\textsc</span><span class="si">{armagnac}</span><span class="s2">&quot;</span><span class="p">,</span>
    <span class="s2">&quot;affine_invariant&quot;</span><span class="p">:</span> <span class="s2">&quot;géométrique&quot;</span><span class="p">,</span>
<span class="p">}</span>


<span class="k">def</span><span class="w"> </span><span class="nf">sweep</span><span class="p">(</span><span class="n">args</span><span class="p">):</span>
<span class="w">    </span><span class="sd">&quot;&quot;&quot;One experiment per (df, model, mean, seed), all through eusipco_2026.&quot;&quot;&quot;</span>
    <span class="n">accuracies</span> <span class="o">=</span> <span class="n">defaultdict</span><span class="p">(</span><span class="nb">list</span><span class="p">)</span>
    <span class="n">total</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">df</span><span class="p">)</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">MODELS</span><span class="p">)</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">)</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">seeds</span><span class="p">)</span>
    <span class="n">done</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="k">for</span> <span class="n">df</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">df</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">position</span> <span class="ow">in</span> <span class="n">MODELS</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">mean</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">:</span>
                <span class="k">for</span> <span class="n">seed</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">seeds</span><span class="p">:</span>
                    <span class="n">config</span> <span class="o">=</span> <span class="n">ExperimentConfig</span><span class="p">(</span>
                        <span class="n">generation_mode</span><span class="o">=</span><span class="s2">&quot;wishart&quot;</span><span class="p">,</span>
                        <span class="n">structure</span><span class="o">=</span><span class="s2">&quot;full&quot;</span><span class="p">,</span>
                        <span class="n">eigenvalue_mode</span><span class="o">=</span><span class="s2">&quot;random&quot;</span><span class="p">,</span>
                        <span class="n">matrix_size</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">matrix_size</span><span class="p">,</span>
                        <span class="n">n_classes</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_classes</span><span class="p">,</span>
                        <span class="n">n_samples_per_class</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">n_samples_per_class</span><span class="p">,</span>
                        <span class="n">conditioning</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">conditioning</span><span class="p">,</span>
                        <span class="n">n_discriminant</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span>
                        <span class="n">discriminant_position</span><span class="o">=</span><span class="n">position</span><span class="p">,</span>
                        <span class="n">class_separation_ratio</span><span class="o">=</span><span class="mf">0.0</span><span class="p">,</span>
                        <span class="n">max_value</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
                        <span class="n">df</span><span class="o">=</span><span class="n">df</span><span class="p">,</span>
                        <span class="n">batchnorm_method</span><span class="o">=</span><span class="n">mean</span><span class="p">,</span>
                        <span class="n">seed</span><span class="o">=</span><span class="n">seed</span><span class="p">,</span>
                    <span class="p">)</span>
                    <span class="n">result</span> <span class="o">=</span> <span class="n">run_single_experiment</span><span class="p">(</span><span class="n">config</span><span class="p">,</span> <span class="n">force_cpu</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">cpu</span><span class="p">)</span>
                    <span class="n">accuracies</span><span class="p">[(</span><span class="n">df</span><span class="p">,</span> <span class="n">position</span><span class="p">,</span> <span class="n">mean</span><span class="p">)]</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">result</span><span class="p">[</span><span class="s2">&quot;test_acc&quot;</span><span class="p">])</span>
                    <span class="n">done</span> <span class="o">+=</span> <span class="mi">1</span>
                    <span class="nb">print</span><span class="p">(</span>
                        <span class="sa">f</span><span class="s2">&quot;  [</span><span class="si">{</span><span class="n">done</span><span class="si">:</span><span class="s2">4d</span><span class="si">}</span><span class="s2">/</span><span class="si">{</span><span class="n">total</span><span class="si">}</span><span class="s2">] df=</span><span class="si">{</span><span class="n">df</span><span class="si">:</span><span class="s2">4d</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">MODELS</span><span class="p">[</span><span class="n">position</span><span class="p">]</span><span class="si">:</span><span class="s2">16s</span><span class="si">}</span><span class="s2"> &quot;</span>
                        <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">mean</span><span class="si">:</span><span class="s2">40s</span><span class="si">}</span><span class="s2"> seed=</span><span class="si">{</span><span class="n">seed</span><span class="si">:</span><span class="s2">5d</span><span class="si">}</span><span class="s2"> acc=</span><span class="si">{</span><span class="n">result</span><span class="p">[</span><span class="s1">&#39;test_acc&#39;</span><span class="p">]</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span>
                        <span class="n">flush</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                    <span class="p">)</span>
    <span class="k">return</span> <span class="n">accuracies</span>


<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">&quot;__main__&quot;</span><span class="p">:</span>
    <span class="n">parser</span> <span class="o">=</span> <span class="n">argparse</span><span class="o">.</span><span class="n">ArgumentParser</span><span class="p">(</span>
        <span class="s2">&quot;Accuracy of each batch-norm mean against the degrees of freedom of the &quot;</span>
        <span class="s2">&quot;Wishart / inverse-Wishart model that generated the data.&quot;</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--df&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">64</span><span class="p">,</span> <span class="mi">96</span><span class="p">,</span> <span class="mi">160</span><span class="p">,</span> <span class="mi">320</span><span class="p">,</span> <span class="mi">640</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Degrees of freedom. Must exceed matrix_size - 1. As df grows the &quot;</span>
             <span class="s2">&quot;law concentrates and the choice of mean should matter less.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--means&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="nb">sorted</span><span class="p">(</span><span class="n">MEANS</span><span class="p">),</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Batch-norm means to compare.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--matrix-size&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">64</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--n-classes&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">3</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--n-samples-per-class&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">120</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--conditioning&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">float</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mf">100.0</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--seeds&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">nargs</span><span class="o">=</span><span class="s2">&quot;+&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="p">[</span><span class="mi">42</span><span class="p">,</span> <span class="mi">123</span><span class="p">,</span> <span class="mi">456</span><span class="p">,</span> <span class="mi">789</span><span class="p">,</span> <span class="mi">1011</span><span class="p">],</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Seeds, as in the wishart-inverse grid of eusipco_2026.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--cpu&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--storage_path&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;outputs/wishart_model&quot;</span><span class="p">,</span>
        <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Output directory for LaTeX exports.&quot;</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--show-interactive&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="s2">&quot;store_true&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span>
        <span class="s2">&quot;--export&quot;</span><span class="p">,</span> <span class="n">action</span><span class="o">=</span><span class="n">argparse</span><span class="o">.</span><span class="n">BooleanOptionalAction</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="kc">True</span>
    <span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--axis_width&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;0.45</span><span class="se">\\</span><span class="s2">textwidth&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--axis_height&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">str</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="s2">&quot;4.6cm&quot;</span><span class="p">)</span>
    <span class="n">parser</span><span class="o">.</span><span class="n">add_argument</span><span class="p">(</span><span class="s2">&quot;--seed&quot;</span><span class="p">,</span> <span class="nb">type</span><span class="o">=</span><span class="nb">int</span><span class="p">,</span> <span class="n">default</span><span class="o">=</span><span class="mi">42</span><span class="p">,</span> <span class="n">help</span><span class="o">=</span><span class="s2">&quot;Unused; for provenance.&quot;</span><span class="p">)</span>
    <span class="n">args</span> <span class="o">=</span> <span class="n">parser</span><span class="o">.</span><span class="n">parse_args</span><span class="p">()</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">show_interactive</span><span class="p">:</span>
        <span class="n">apply_style</span><span class="p">()</span>
    <span class="n">os</span><span class="o">.</span><span class="n">makedirs</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="n">exist_ok</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

    <span class="n">accuracies</span> <span class="o">=</span> <span class="n">sweep</span><span class="p">(</span><span class="n">args</span><span class="p">)</span>

    <span class="c1"># ---- One panel per model, the means as curves against df ---------------</span>
    <span class="n">fig</span><span class="p">,</span> <span class="n">axes</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mf">3.4</span> <span class="o">*</span> <span class="mi">2</span><span class="p">,</span> <span class="mf">3.4</span><span class="p">),</span> <span class="n">sharey</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">axis</span><span class="p">,</span> <span class="p">(</span><span class="n">position</span><span class="p">,</span> <span class="n">title</span><span class="p">)</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">axes</span><span class="p">,</span> <span class="n">MODELS</span><span class="o">.</span><span class="n">items</span><span class="p">()):</span>
        <span class="k">for</span> <span class="n">index</span><span class="p">,</span> <span class="n">mean</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">):</span>
            <span class="n">medians</span> <span class="o">=</span> <span class="p">[</span>
                <span class="mi">100</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">accuracies</span><span class="p">[(</span><span class="n">df</span><span class="p">,</span> <span class="n">position</span><span class="p">,</span> <span class="n">mean</span><span class="p">)])</span> <span class="k">for</span> <span class="n">df</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">df</span>
            <span class="p">]</span>
            <span class="n">spread</span> <span class="o">=</span> <span class="p">[</span>
                <span class="mi">100</span> <span class="o">*</span> <span class="n">np</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">accuracies</span><span class="p">[(</span><span class="n">df</span><span class="p">,</span> <span class="n">position</span><span class="p">,</span> <span class="n">mean</span><span class="p">)])</span> <span class="k">for</span> <span class="n">df</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">df</span>
            <span class="p">]</span>
            <span class="n">axis</span><span class="o">.</span><span class="n">errorbar</span><span class="p">(</span>
                <span class="n">args</span><span class="o">.</span><span class="n">df</span><span class="p">,</span> <span class="n">medians</span><span class="p">,</span> <span class="n">yerr</span><span class="o">=</span><span class="n">spread</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="sa">f</span><span class="s2">&quot;C</span><span class="si">{</span><span class="n">index</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">linewidth</span><span class="o">=</span><span class="mf">1.3</span><span class="p">,</span>
                <span class="n">marker</span><span class="o">=</span><span class="s2">&quot;o&quot;</span><span class="p">,</span> <span class="n">markersize</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">capsize</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">MEANS</span><span class="p">[</span><span class="n">mean</span><span class="p">],</span>
            <span class="p">)</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_xscale</span><span class="p">(</span><span class="s2">&quot;log&quot;</span><span class="p">)</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;degrés de liberté&quot;</span><span class="p">)</span>
        <span class="n">axis</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;données </span><span class="si">{</span><span class="n">title</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="sa">r</span><span class="s2">&quot;précision de test (\%)&quot;</span><span class="p">)</span>
    <span class="n">axes</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
    <span class="n">fig</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>

    <span class="c1"># ---- Digest ------------------------------------------------------------</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="n">header</span> <span class="o">=</span> <span class="s2">&quot;  &quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">MEANS</span><span class="p">[</span><span class="n">m</span><span class="p">][:</span><span class="mi">12</span><span class="p">]</span><span class="si">:</span><span class="s2">&gt;13s</span><span class="si">}</span><span class="s2">&quot;</span> <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="s1">&#39;df&#39;</span><span class="si">:</span><span class="s2">&gt;6s</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="s1">&#39;modèle&#39;</span><span class="si">:</span><span class="s2">&gt;16s</span><span class="si">}</span><span class="s2">  </span><span class="si">{</span><span class="n">header</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="k">for</span> <span class="n">df</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">df</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">position</span><span class="p">,</span> <span class="n">title</span> <span class="ow">in</span> <span class="n">MODELS</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
            <span class="n">cells</span> <span class="o">=</span> <span class="s2">&quot;  &quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="mi">100</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">accuracies</span><span class="p">[(</span><span class="n">df</span><span class="p">,</span><span class="w"> </span><span class="n">position</span><span class="p">,</span><span class="w"> </span><span class="n">m</span><span class="p">)])</span><span class="si">:</span><span class="s2">12.1f</span><span class="si">}</span><span class="s2">%&quot;</span>
                <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">means</span>
            <span class="p">)</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">&quot;</span><span class="si">{</span><span class="n">df</span><span class="si">:</span><span class="s2">6d</span><span class="si">}</span><span class="s2"> </span><span class="si">{</span><span class="n">title</span><span class="si">:</span><span class="s2">&gt;16s</span><span class="si">}</span><span class="s2">  </span><span class="si">{</span><span class="n">cells</span><span class="si">}</span><span class="s2">&quot;</span><span class="p">)</span>

    <span class="n">np</span><span class="o">.</span><span class="n">savez</span><span class="p">(</span>
        <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;results.npz&quot;</span><span class="p">),</span>
        <span class="n">df</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">df</span><span class="p">),</span>
        <span class="n">means</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">),</span>
        <span class="n">models</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="nb">list</span><span class="p">(</span><span class="n">MODELS</span><span class="p">)),</span>
        <span class="n">accuracies</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span>
            <span class="p">[</span>
                <span class="p">[[</span><span class="n">accuracies</span><span class="p">[(</span><span class="n">df</span><span class="p">,</span> <span class="n">p</span><span class="p">,</span> <span class="n">m</span><span class="p">)]</span> <span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">means</span><span class="p">]</span> <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">MODELS</span><span class="p">]</span>
                <span class="k">for</span> <span class="n">df</span> <span class="ow">in</span> <span class="n">args</span><span class="o">.</span><span class="n">df</span>
            <span class="p">]</span>
        <span class="p">),</span>
    <span class="p">)</span>

    <span class="k">if</span> <span class="n">args</span><span class="o">.</span><span class="n">export</span><span class="p">:</span>
        <span class="n">figure_path</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">args</span><span class="o">.</span><span class="n">storage_path</span><span class="p">,</span> <span class="s2">&quot;wishart_model.tex&quot;</span><span class="p">)</span>
        <span class="n">save_tikz</span><span class="p">(</span>
            <span class="n">figure_path</span><span class="p">,</span> <span class="n">axis_width</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_width</span><span class="p">,</span> <span class="n">axis_height</span><span class="o">=</span><span class="n">args</span><span class="o">.</span><span class="n">axis_height</span>
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
<span class="param-flag">--df</span><span class="param-type">int</span><span class="param-default">default <b>[64, 96, 160, 320, 640]</b></span>
</div>
<p class="param-help">Degrees of freedom. Must exceed matrix_size - 1. As df grows the law concentrates and the choice of mean should matter less.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--means</span><span class="param-type">str</span>
</div>
<p class="param-help">Batch-norm means to compare.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--matrix-size</span><span class="param-type">int</span><span class="param-default">default <b>64</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-classes</span><span class="param-type">int</span><span class="param-default">default <b>3</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--n-samples-per-class</span><span class="param-type">int</span><span class="param-default">default <b>120</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--conditioning</span><span class="param-type">float</span><span class="param-default">default <b>100.0</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seeds</span><span class="param-type">int</span><span class="param-default">default <b>[42, 123, 456, 789, 1011]</b></span>
</div>
<p class="param-help">Seeds, as in the wishart-inverse grid of eusipco_2026.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--cpu</span><span class="param-type">flag</span><span class="param-default">default <b>True</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--storage_path</span><span class="param-type">str</span><span class="param-default">default <b>outputs/wishart_model</b></span>
</div>
<p class="param-help">Output directory for LaTeX exports.</p>
</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--show-interactive</span><span class="param-type">flag</span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--export</span><span class="param-default">default <b>True</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_width</span><span class="param-type">str</span><span class="param-default">default <b>0.45\textwidth</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--axis_height</span><span class="param-type">str</span><span class="param-default">default <b>4.6cm</b></span>
</div>

</div>
<div class="param">
<div class="param-head">
<span class="param-flag">--seed</span><span class="param-type">int</span><span class="param-default">default <b>42</b></span>
</div>
<p class="param-help">Unused; for provenance.</p>
</div>
</div>

## Config

`4-deeplearning/experiments/spdnet_wishart_model.yaml`

<a class="back-link" href="../../chapters/4-deeplearning/">← All experiments in 4 · Deep Learning</a>
