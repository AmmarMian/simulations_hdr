# Chapter 2 · Detection

Deciding whether something changed between two radar or sonar acquisitions of
the same scene, when the only description of the background is a covariance
matrix estimated from a handful of neighbouring pixels.

The experiments come in two kinds: Monte-Carlo studies that measure detection
power and false-alarm behaviour on simulated clutter, and runs of the same
detectors over real SAR images.

## Data

**Simulated** — the Monte-Carlo experiments need nothing; they generate their
own clutter under Gaussian, compound-Gaussian and Kronecker-structured models.

**Real SAR** — five UAVSAR scenes, published on Zenodo and fetched by a script
in the repository. They need one reformatting pass before use:

```sh
cd 2-detection
bash data/download_sar.sh
uv run sar_experiments/compute_detection_real_data/prepare_data.py data/SAR/Scene1.npy
```

**Real sonar** — not distributable, so the sonar experiments here are the
simulated ones only.

## Caveats

**The Monte-Carlo experiments are the long ones.** The sonar angle-grid sweep
computes a detection probability at every pair of arrival angles on a 51×51 grid
with a thousand trials in each cell, and takes hours; the SNR sweep and the ROC
curves are comparable. They parallelise over CPU cores — use `--n-workers`, and
`--n-trials` to shorten a trial run. `qanat experiment status` reports how far
along they are.

**A validation script exists but needs a reference file that is not in the
repository.** `sonar_experiments/validation/compare_matlab_reference.py`
compares the sonar detectors against a reference implementation, using `.mat`
files that carry the data, the exact covariance and the expected statistic for
every detector. Without those files the script cannot run; the detectors
themselves are covered by the ordinary test suite.

## Experiments

<!-- experiments-start -->
<div class="exp-chapter">
<div class="exp-group">
<h3 class="exp-group-heading">SAR · Benchmarks</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_bench_memory</div>

</div>
<div class="exp-desc">Memory benchmark for offline Gaussian and DCG GLRT detectors (CPU memray + GPU torch)</div>
<div class="exp-tags"><span class="exp-tag">benchmark</span><span class="exp-tag">memory</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>bash 2-detection/sar_experiments/benchmarks/memory_benchmark.sh</code></div>
<a class="exp-details-link" href="../../experiments/sar_bench_memory/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_bench_online_memory</div>

</div>
<div class="exp-desc">Memory benchmark for online Gaussian, DCG and Kronecker detectors (CPU memray + GPU torch)</div>
<div class="exp-tags"><span class="exp-tag">benchmark</span><span class="exp-tag">memory</span><span class="exp-tag">online</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>bash 2-detection/sar_experiments/benchmarks/memory_benchmark_online.sh</code></div>
<a class="exp-details-link" href="../../experiments/sar_bench_online_memory/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_bench_online_time</div>

</div>
<div class="exp-desc">Time benchmark for online Gaussian, DCG and Kronecker detectors (CPU + GPU)</div>
<div class="exp-tags"><span class="exp-tag">benchmark</span><span class="exp-tag">time</span><span class="exp-tag">online</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>bash 2-detection/sar_experiments/benchmarks/time_benchmark_online.sh</code></div>
<a class="exp-details-link" href="../../experiments/sar_bench_online_time/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_bench_time</div>

</div>
<div class="exp-desc">Time benchmark for offline Gaussian and DCG GLRT detectors (CPU + GPU)</div>
<div class="exp-tags"><span class="exp-tag">benchmark</span><span class="exp-tag">time</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>bash 2-detection/sar_experiments/benchmarks/time_benchmark.sh</code></div>
<a class="exp-details-link" href="../../experiments/sar_bench_time/">Parameters &amp; details →</a>
</div>
</div>
</div>

<div class="exp-group">
<h3 class="exp-group-heading">SAR · Real Data</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_det_off_dcg</div>

</div>
<div class="exp-desc">Offline DCG GLRT change detection on real SAR data</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">dcg</span><span class="exp-tag">offline</span><span class="exp-tag">real-data</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/compute_detection_real_data/offline_dcg.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_det_off_dcg/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_det_off_gauss</div>

</div>
<div class="exp-desc">Offline Gaussian GLRT change detection on real SAR data</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">gaussian</span><span class="exp-tag">offline</span><span class="exp-tag">real-data</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/compute_detection_real_data/offline_gaussian.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_det_off_gauss/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_det_off_kron</div>

</div>
<div class="exp-desc">Offline Kronecker GLRT change detection on real SAR data</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">offline</span><span class="exp-tag">real-data</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/compute_detection_real_data/offline_kronecker.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_det_off_kron/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_det_on_dcg</div>

</div>
<div class="exp-desc">Online DCG GLRT change detection on real SAR data</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">dcg</span><span class="exp-tag">online</span><span class="exp-tag">real-data</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/compute_detection_real_data/online_dcg.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_det_on_dcg/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_det_on_gauss</div>

</div>
<div class="exp-desc">Online Gaussian GLRT change detection on real SAR data</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">gaussian</span><span class="exp-tag">online</span><span class="exp-tag">real-data</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/compute_detection_real_data/online_gaussian.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_det_on_gauss/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_det_on_kron</div>

</div>
<div class="exp-desc">Online Kronecker GLRT change detection on real SAR data</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">online</span><span class="exp-tag">real-data</span><span class="exp-tag">SAR</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/compute_detection_real_data/online_kronecker.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_det_on_kron/">Parameters &amp; details →</a>
</div>
</div>
</div>

<div class="exp-group">
<h3 class="exp-group-heading">SAR · Monte Carlo</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_kron_mse</div>

</div>
<div class="exp-desc">Mean squared error of the Kronecker estimators, offline and recursive, against the intrinsic Cramer-Rao bounds</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">estimation</span><span class="exp-tag">icrb</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_kron_mse_icrb.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_kron_mse/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_kron_struct</div>

</div>
<div class="exp-desc">What assuming a Kronecker structure buys: estimation error against the window size</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">estimation</span><span class="exp-tag">structure</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_kron_structure_vs_n.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_kron_struct/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_power</div>

</div>
<div class="exp-desc">Detection power against the number of dates, for the four change detectors, offline and online</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">puissance</span><span class="exp-tag">H1</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_power_detectors.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_power/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_roc</div>

</div>
<div class="exp-desc">ROC curves for the four change detectors, offline and online</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">roc</span><span class="exp-tag">H1</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_roc_detectors.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_roc/">Parameters &amp; details →</a>
</div>
</div>
</div>

<div class="exp-group">
<h3 class="exp-group-heading">Sonar · Detection</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sonar_pd_angle</div>

</div>
<div class="exp-desc">MC PD vs (theta1, theta2) angle map for sonar two-array detectors at fixed SNR</div>
<div class="exp-tags"><span class="exp-tag">sonar</span><span class="exp-tag">detection</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">angle-map</span></div>
<div class="exp-run"><code>uv run python 2-detection/sonar_experiments/mc_simulations/mc_pd_angle.py</code></div>
<a class="exp-details-link" href="../../experiments/sonar_pd_angle/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sonar_pd_snr</div>

</div>
<div class="exp-desc">MC detection probability vs SNR for sonar two-array detectors (M-NMF-G/R/I, adaptive 2TYL/SCM variants)</div>
<div class="exp-tags"><span class="exp-tag">sonar</span><span class="exp-tag">detection</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">pd-snr</span></div>
<div class="exp-run"><code>uv run python 2-detection/sonar_experiments/mc_simulations/mc_pd_snr.py</code></div>
<a class="exp-details-link" href="../../experiments/sonar_pd_snr/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sonar_pfa_threshold</div>

</div>
<div class="exp-desc">MC empirical PFA vs threshold for sonar two-array detectors — matrix-CFAR verification</div>
<div class="exp-tags"><span class="exp-tag">sonar</span><span class="exp-tag">detection</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">pfa-cfar</span></div>
<div class="exp-run"><code>uv run python 2-detection/sonar_experiments/mc_simulations/mc_pfa_threshold.py</code></div>
<a class="exp-details-link" href="../../experiments/sonar_pfa_threshold/">Parameters &amp; details →</a>
</div>
</div>
</div>

<div class="exp-group">
<h3 class="exp-group-heading">Sonar · Convergence</h3>
<div class="exp-grid">
<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sonar_tyler_conv</div>

</div>
<div class="exp-desc">2TYL fixed-point convergence — relative Frobenius deviation vs iteration</div>
<div class="exp-tags"><span class="exp-tag">sonar</span><span class="exp-tag">estimation</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">convergence</span></div>
<div class="exp-run"><code>uv run python 2-detection/sonar_experiments/mc_simulations/tyler_convergence.py</code></div>
<a class="exp-details-link" href="../../experiments/sonar_tyler_conv/">Parameters &amp; details →</a>
</div>
</div>
</div>
</div>
<!-- experiments-end -->
