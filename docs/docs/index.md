# HDR Simulations


<div style="display: flex; gap: 2rem; align-items: flex-start; flex-wrap: wrap;">

  <!-- LEFT: IMAGE -->
  <div style="flex: 0 0 400px;">
    <figure>
    <img id="theme-img" src="./assets/flow_light.png" style="width: 100%;">
    <figcaption>A flow particle simulation on the sphere</figcaption>
    </figure>
  </div>

<script>
function updateImg() {
  const img = document.getElementById('theme-img');
  const isDark = document.documentElement.getAttribute('data-paper') === 'dark';
  img.src = isDark ? './assets/flow_dark.png' : './assets/flow_light.png';
}
updateImg();
new MutationObserver(updateImg).observe(document.documentElement, { attributes: true, attributeFilter: ['data-paper'] });
</script>

  <!-- RIGHT: TEXT -->
  <div style="flex: 1; min-width: 300px;">

    <p>
      This page groups documentation that accompany my dissertation for the diploma of Habilitation à Diriger des recherches (HDR) called:
      <br><br>
      <b>Matrices de covariances : des statistiques multivariées à l'apprentissage profond</b>,
      <br><br>
      for which the PDF is available <a href="#">here</a>
      <label for="sn-1" class="sidenote-number"></label>
      <input type="checkbox" id="sn-1" class="margin-toggle"/>
      <span class="sidenote">
        Only available in French language. The documentation being useful to a broader community outside of France has been done in English.
      </span>
    </p>

  </div>

</div>


Given that the numerous results presented depend on numerical experimenting, this sidecar allows, for any interested reader, to be able to reproduce, play and experiment each of them<label for="sn-2" class="sidenote-number"></label><input type="checkbox" id="sn-2" class="margin-toggle"/><span class="sidenote">To the exception of experiments on real Sonar and GPR data, not having been granted permission to share the datasets.</span>. The aim is to have a diversity of useful code for working on covariance matrices. This is done with several concerns in mind :

* **reproducibility:** the ability to obtain same results and conclusions on any given computer;
* **scalability:** being able to run on a small laptop or take advantage of CPU/GPU parallelisation when available<label for="sn-3" class="sidenote-number"></label><input type="checkbox" id="sn-3" class="margin-toggle"/><span class="sidenote">We omit the case of HPC parallelisation which introduce a specialized layer that obfuscate the algorithms used.</span>;


## Installation

Each chapter declares its own extras. Install only what you need:

```sh
# base (numpy, torch-cpu, scipy)
uv sync

# chapter 2 — optional compute backends
uv sync --extra cupy        # NVIDIA CUDA
uv sync --extra jax         # JAX CPU
uv sync --extra jax-cuda    # JAX CUDA
uv sync --extra jax-metal   # Apple Silicon
```

## Running an experiment

Experiments are managed by [qanat](https://github.com/ammarmian/qanat). Each YAML file in a chapter's `experiments/` directory defines one experiment.

```sh
# list all experiments for chapter 2
qanat experiment list

# run a specific experiment
qanat experiment run sar_mc_kron_h1

# plot results from the last run
qanat action run sar_mc_kron_h1 plot
```

Results land in `results/<experiment_name>/run_<N>/` alongside a self-contained `_plot.py` that can be run standalone.
