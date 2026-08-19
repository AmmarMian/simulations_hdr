# context_riemann_frobenius_rao

Estimation error of the SCM and of Tyler's estimator, in Frobenius norm and in Rao distance

**Tags:** `context`  `riemann`  `robust`  `monte-carlo`

## Run

```sh
uv run python 1-context/riemann_frobenius_rao/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_features` | int | `7` | Dimension of the observations. |
| `--n_samples` | int | `[9, 10, 12, 14, 18, 25, 40, 70, 120, 200]` | Sample sizes at which the error is evaluated. The first ones are just above the dimension, which is where the two metrics disagree. |
| `--n_trials` | int | `500` | Number of MC-trials per sample size. |
| `--dof` | float | `3.0` | Degrees of freedom of the Student data. Heavy enough for the SCM to suffer, light enough for its covariance to exist. |
| `--condition` | float | `50.0` | Condition number of the true scatter matrix. |
| `--iter_max` | int | `300` | Fixed-point iterations for Tyler. |
| `--storage_path` | str | `outputs/riemann_frobenius_rao` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figure (.tex) (default: True). |
| `--axis_width` | str | `0.45\textwidth` | Width of a single panel in the exported PGFPlots figure. Set here rather than patched into the .tex afterwards, so that a re-sync into the dissertation does not undo it. |
| `--axis_height` | str | `4.6cm` | Height of a single panel in the exported PGFPlots figure. |
| `--backend` | str | `numpy` | Compute backend (numpy, torch-cpu, torch-mps, ...). |
| `--seed` | int | `42` | random seed generation base seed |

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
