# context_complex_circularity

Same covariance, four pseudo-covariances — what circularity buys and what it hides

**Tags:** `context`  `complex`  `circularity`  `illustration`

## Run

```sh
uv run python 1-context/complex_circularity/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rho` | float | `[0.0, 0.5, 0.5, 0.9]` | Moduli \|C\|/Gamma of the pseudo-covariance, one panel each. Must lie in [0,1]: 0 is circular, 1 is a degenerate (real) variable. |
| `--phase` | float | `[0.0, 0.0, 0.3333, 0.3333]` | Arguments of the pseudo-covariance, in units of pi, one per rho. |
| `--gamma` | float | `1.0` | The covariance, shared by every panel. It is what the panels hold fixed, so that only the pseudo-covariance distinguishes them. |
| `--n_samples` | int | `600` | Observations drawn per panel. |
| `--probability` | float | `0.9` | Probability mass enclosed by the drawn concentration curves. |
| `--storage_path` | str | `outputs/complex_circularity` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figure (.tex) (default: True). |
| `--seed` | int | `42` | random seed generation base seed |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-08-18</span><br>
  <code>--rho</code> <span class='mn-default'>[0.0, 0.5, 0.5, 0.9]</span><br>
  <code>--phase</code> <span class='mn-default'>[0.0, 0.0, 0.3333, 0.3333]</span><br>
  <code>--gamma</code> <span class='mn-default'>1.0</span><br>
  <code>--n_samples</code> <span class='mn-default'>600</span><br>
  <code>--probability</code> <span class='mn-default'>0.9</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_complex_circularity.json" data-title="context_complex_circularity"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Gamma = 1, shared by every panel; N = 600
  |C|/Gamma=0    arg C=0      pi  Gamma_hat=0.971  C_hat=0.089 exp(j-0.586pi)
  |C|/Gamma=0.5  arg C=0      pi  Gamma_hat=0.998  C_hat=0.510 exp(j0.000pi)
  |C|/Gamma=0.5  arg C=0.3333 pi  Gamma_hat=0.953  C_hat=0.444 exp(j0.376pi)
  |C|/Gamma=0.9  arg C=0.3333 pi  Gamma_hat=0.995  C_hat=0.889 exp(j0.337pi)
Saved circularity panels in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_complex_circularity/run_17/circularity.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_complex_circularity.yaml`
