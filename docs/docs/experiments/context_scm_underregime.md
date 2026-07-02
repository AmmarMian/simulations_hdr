# context_scm_underregime

SCM estimator behavior in the under-regime (d > N) — mean, covariance, condition number errors

**Tags:** `context`  `scm`  `under-regime`  `monte-carlo`

## Run

```sh
uv run python 1-context/scm_underregime/main.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--N` | int | `30` | Number of observations. |
| `--n_trials` | int | `50` | Number of MC-trials. |
| `--storage_path` | str | `outputs/error_estimation_scm_underregime` | Output directory for LaTeX exports (injected by qanat, or set manually). |
| `--show-interactive` | — | — | Show plots interactively with matplotlib. |
| `--export` | — | `True` | Save TikZ/PGFPlots figures (.tex) (default: True). |
| `--seed` | int | `42` | random seed generation base seed |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <span class="mn-date">Generated: 2026-07-02</span><br>
  <code>--N</code> <span class='mn-default'>30</span><br>
  <code>--n_trials</code> <span class='mn-default'>10000</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
</span>
<div class="exp-result-card">
<div class="plotly-wrap" data-src="../../assets/data/context_scm_underregime.json" data-title="context_scm_underregime"></div>
<details class="exp-log">
<summary>stdout</summary>
<div class="exp-log-text">Launching simulation
Working... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Done.
Saved mean error in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_scm_underregime/run_2/mean.tex
Saved cov error in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_scm_underregime/run_2/cov.tex
Saved cov error in /Users/ammarmian/Research/HDR/simulations_hdr/results/context_scm_underregime/run_2/cond.tex
</div>
</details>
</div>

## Config

`1-context/experiments/context_scm_underregime.yaml`
