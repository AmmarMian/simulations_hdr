#!/usr/bin/env bash
# Benchmark detection scripts using hyperfine.
# Grid: {cpu, gpu} x {gaussian, dcg} x {no_wavelet, wavelet}
# Results saved as JSON per cell, aggregated to CSV, bar chart generated at end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SCRIPT_DIR}/../../data/SAR/Scene_1.npy"
WINDOW_SIZE=7
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"
RUNS=10
WARMUP=2

mkdir -p "$RESULTS_DIR"

CPU_SCRIPT="${SCRIPT_DIR}/compare_time_detection_offline_cpu.py"
GPU_SCRIPT="${SCRIPT_DIR}/compare_time_detection_offline_gpu.py"

TOTAL=8
CURRENT=0

run_benchmark() {
  local label="$1"
  local cmd="$2"
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "[$CURRENT/$TOTAL] === $label ==="
  hyperfine \
    --runs "$RUNS" \
    --warmup "$WARMUP" \
    --export-json "${RESULTS_DIR}/${label}.json" \
    --show-output \
    "$cmd"
}

# CPU benchmarks
# run_benchmark "cpu_gaussian_no_wavelet" \
#   "uv run ${CPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors gaussian"
#
# run_benchmark "cpu_gaussian_wavelet" \
#   "uv run ${CPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors gaussian --wavelet"
#
# run_benchmark "cpu_dcg_no_wavelet" \
#   "uv run ${CPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors dcg --iteration-chunk 512"

run_benchmark "cpu_dcg_wavelet" \
  "uv run ${CPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors dcg --wavelet --iteration-chunk 128"

# GPU benchmarks
# run_benchmark "gpu_gaussian_no_wavelet" \
#   "uv run ${GPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --detectors gaussian --splitting '(3,3)'"
#
# run_benchmark "gpu_gaussian_wavelet" \
#   "uv run ${GPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --detectors gaussian --wavelet --splitting '(15,15)'"
#
# run_benchmark "gpu_dcg_no_wavelet" \
#   "uv run ${GPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --detectors dcg --splitting '(15,15)' --iteration-chunk 512"

run_benchmark "gpu_dcg_wavelet" \
  "uv run ${GPU_SCRIPT} ${DATA} ${WINDOW_SIZE} --detectors dcg --wavelet --splitting '(63,63)' --iteration-chunk 128"

echo ""
echo "All benchmarks done. Aggregating results and generating chart..."

uv run - "$RESULTS_DIR" <<'EOF'
import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplot2tikz

results_dir = Path(sys.argv[1])

labels = [
    "cpu_gaussian_no_wavelet",
    "cpu_gaussian_wavelet",
    "cpu_dcg_no_wavelet",
    "cpu_dcg_wavelet",
    "gpu_gaussian_no_wavelet",
    "gpu_gaussian_wavelet",
    "gpu_dcg_no_wavelet",
    "gpu_dcg_wavelet",
]

rows = []
for label in labels:
    json_path = results_dir / f"{label}.json"
    if not json_path.exists() or json_path.stat().st_size == 0:
        print(f"  WARNING: missing or empty {json_path}, skipping.")
        continue
    with open(json_path) as f:
        data = json.load(f)
    result = data["results"][0]
    mean = result["mean"]
    std = result["stddev"]
    rows.append({"label": label, "mean_s": mean, "std_s": std})
    print(f"  {label}: {mean:.2f} ± {std:.2f} s")

# Write CSV
csv_path = results_dir / "benchmark_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "mean_s", "std_s"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nCSV saved to {csv_path}")

# Bar chart
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(rows))
means = [r["mean_s"] for r in rows]
stds  = [r["std_s"]  for r in rows]
tick_labels = [r["label"] for r in rows]

# Color by backend
colors = ["steelblue" if r["label"].startswith("cpu") else "darkorange" for r in rows]

bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Time (s)")
ax.set_title("Detection time benchmark (mean ± std, 5 runs)")

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="steelblue",  label="CPU"),
    Patch(color="darkorange", label="GPU"),
])

fig.tight_layout()
chart_path = results_dir / "benchmark_chart.png"
fig.savefig(chart_path, dpi=150)
print(f"Chart saved to {chart_path}")

tikz_path = results_dir / "benchmark_chart.tex"
matplot2tikz.save(str(tikz_path))
print(f"TikZ saved to {tikz_path}")

plt.close(fig)
EOF

echo "Done. Results in ${RESULTS_DIR}/"
