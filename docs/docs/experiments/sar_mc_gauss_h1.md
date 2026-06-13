# sar_mc_gauss_h1

MC power curve — OnlineGaussianGLRT vs GaussianGLRT under H1 (change detection)

**Tags:** `detection`  `gaussian`  `H1`  `monte-carlo`  `power-curve`

## Run

```sh
uv run python 2-detection/sar_experiments/mc_simulations/mc_gaussian_h1.py
```

## Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n-features` | int | `8` | Feature dimension p; n_samples is fixed to 2*p+1 (default 8). |
| `--n-trials` | int | `10000` | Number of Monte-Carlo trials (default 10000). |
| `--T-max` | int | `1000` | Maximum number of time steps (default 1000). |
| `--T-min` | int | `5` | Minimum number of time steps (default 5). |
| `--n-T` | int | `30` | Number of T values in log scale (default 30). |
| `--seed` | int | `42` | RNG seed for data generation (default 42). |
| `--sigma-seed` | int | `0` | Seed for Sigma_true generation, independent from --seed (default 0). |
| `--backend` | str | `numpy` | Compute backend. numpy → multiprocessing.Pool (one worker per trial); all others → trials stacked in leading batch dimension (default numpy). |
| `--n-workers` | int | — | Pool workers for numpy backend (default: os.cpu_count()). |
| `--export` | — | `True` | Save .npz results + provenance sidecar + plot script (default: True). |
| `--storage-path` / `--storage_path` / `--export-path` | str | `./exports` | Directory for exported results; --storage-path is the qanat alias (default: ./exports). |
| `--show-interactive` | — | — | Display figures interactively at the end of the simulation. |

## Results

<span class="marginnote">
  <span class="mn-label">Parameters</span>
  <code>--n-features</code> <span class='mn-default'>8</span><br>
  <code>--n-trials</code> <span class='mn-default'>100</span><br>
  <code>--T-max</code> <span class='mn-default'>1000</span><br>
  <code>--T-min</code> <span class='mn-default'>5</span><br>
  <code>--n-T</code> <span class='mn-default'>30</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
  <code>--sigma-seed</code> <span class='mn-default'>0</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--n-workers</code><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
</span>
<div class="plotly-wrap" data-src="../../assets/data/sar_mc_gauss_h1.json" data-title="sar_mc_gauss_h1 — power curve"></div>

<span class="marginnote">
  <span class="mn-label">Run · n100</span>
  <code>--n-features</code> <span class='mn-default'>8</span><br>
  <code>--n-trials</code> <span class='mn-default'>100</span><br>
  <code>--T-max</code> <span class='mn-default'>200</span><br>
  <code>--T-min</code> <span class='mn-default'>5</span><br>
  <code>--n-T</code> <span class='mn-default'>30</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
  <code>--sigma-seed</code> <span class='mn-default'>0</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--n-workers</code><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
</span>
<div class="plotly-wrap" data-src="../../assets/data/sar_mc_gauss_h1.n100.json" data-title="sar_mc_gauss_h1 — n100"></div>

<span class="marginnote">
  <span class="mn-label">Run · n1000</span>
  <code>--n-features</code> <span class='mn-default'>8</span><br>
  <code>--n-trials</code> <span class='mn-default'>100</span><br>
  <code>--T-max</code> <span class='mn-default'>1000</span><br>
  <code>--T-min</code> <span class='mn-default'>5</span><br>
  <code>--n-T</code> <span class='mn-default'>30</span><br>
  <code>--seed</code> <span class='mn-default'>42</span><br>
  <code>--sigma-seed</code> <span class='mn-default'>0</span><br>
  <code>--backend</code> <span class='mn-default'>numpy</span><br>
  <code>--n-workers</code><br>
  <code>--export</code> <span class='mn-default'>True</span><br>
  <code>--show-interactive</code> <span class='mn-default'>False</span><br>
</span>
<div class="plotly-wrap" data-src="../../assets/data/sar_mc_gauss_h1.n1000.json" data-title="sar_mc_gauss_h1 — n1000"></div>

## Config

`2-detection/experiments/sar_mc_gauss_h1.yaml`
