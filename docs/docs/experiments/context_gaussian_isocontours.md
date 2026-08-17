# context_gaussian_isocontours

Isodensity contours and samples of the bivariate Gaussian for three covariance regimes

**Tags:** `context`  `gaussian`  `distributions`  `illustration`

## Run

```sh
uv run python 1-context/probability_densities/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_samples` | int | `50` | Number of samples drawn per regime. Drawn as hollow markers so the isodensity contours stay readable underneath. |
| `--rho` | float | `0.8` | Correlation coefficient of the correlated regime. |
| `--condition` | float | `50.0` | Condition number of the ill-conditioned regime. |
| `--storage_path` | str | `outputs/gaussian_isocontours` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figure (.tex) (default: True). |
| `--seed` | int | `42` | random seed generation base seed |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-08-17</span><br>
  <code>--n_samples</code> <span class='mn-default'>50</span><br>
  <code>--rho</code> <span class='mn-default'>0.8</span><br>
  <code>--condition</code> <span class='mn-default'>50.0</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_gaussian_isocontours.json" data-title="context_gaussian_isocontours"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Saved isocontours in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_gaussian_isocontours/run_12/gaussian_isocontours.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_gaussian_isocontours.yaml`
