# Chapter 2 · Detection

Change-detection in SAR and sonar imagery using robust covariance estimators under Gaussian, DCG, and Kronecker-structured models.

## Experiments

<!-- experiments-start -->
<div class="exp-chapter">
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
<span class="exp-results-badge">Results available</span>
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
<div class="exp-name">sar_mc_dcg_h0</div>

</div>
<div class="exp-desc">MC convergence test — OnlineDCGGLRT telescopes to DCGGLRT as T grows (H0)</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">dcg</span><span class="exp-tag">H0</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_dcg_h0.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_dcg_h0/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_dcg_h1</div>

</div>
<div class="exp-desc">MC power curve — OnlineDCGGLRT vs DCGGLRT under H1 (change detection)</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">dcg</span><span class="exp-tag">H1</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">power-curve</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_dcg_h1.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_dcg_h1/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_gauss_h0</div>

</div>
<div class="exp-desc">MC convergence test — OnlineGaussianGLRT telescopes to GaussianGLRT as T grows (H0)</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">gaussian</span><span class="exp-tag">H0</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_gaussian_h0.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_gauss_h0/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_gauss_h1</div>
<span class="exp-results-badge">Results available</span>
</div>
<div class="exp-desc">MC power curve — OnlineGaussianGLRT vs GaussianGLRT under H1 (change detection)</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">gaussian</span><span class="exp-tag">H1</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">power-curve</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_gaussian_h1.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_gauss_h1/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_kron_h0</div>

</div>
<div class="exp-desc">MC convergence test — OnlineKroneckerGLRT telescopes to KroneckerGLRT as T grows (H0)</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">H0</span><span class="exp-tag">monte-carlo</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_kronecker_h0.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_kron_h0/">Parameters &amp; details →</a>
</div>

<div class="exp-card">
<div class="exp-card-head">
<div class="exp-name">sar_mc_kron_h1</div>

</div>
<div class="exp-desc">MC power curve — OnlineKroneckerGLRT vs KroneckerGLRT under H1 (change detection)</div>
<div class="exp-tags"><span class="exp-tag">detection</span><span class="exp-tag">kronecker</span><span class="exp-tag">H1</span><span class="exp-tag">monte-carlo</span><span class="exp-tag">power-curve</span></div>
<div class="exp-run"><code>uv run python 2-detection/sar_experiments/mc_simulations/mc_kronecker_h1.py</code></div>
<a class="exp-details-link" href="../../experiments/sar_mc_kron_h1/">Parameters &amp; details →</a>
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

## Real-data pipeline

SAR scenes (Scene 1, 2, 4-cropped) must be downloaded and reformatted before running detection experiments:

```sh
cd 2-detection
bash data/download_sar.sh
uv run sar_experiments/compute_detection_real_data/prepare_data.py data/SAR/scene1.npy
```

## Adding a new backend

1. Add an optional extra to `pyproject.toml`.
2. Register the array type in `src/backend.py` → `BACKEND_TYPES`.
3. Implement the dispatch branches in `get_backend_module`, `get_data_on_device`, and `empty_cache`.
4. All estimators and detectors inherit the new backend automatically.
