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

COMPUTE_SCRIPT="${SCRIPT_DIR}/../compute_cd_offline.py"

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
#   "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors gaussian --quiet"
#
# run_benchmark "cpu_gaussian_wavelet" \
#   "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors gaussian --wavelet --quiet"
#
run_benchmark "cpu_dcg_no_wavelet" \
  "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors dcg --iteration-chunk 512 --quiet"
#
# run_benchmark "cpu_dcg_wavelet" \
#   "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cpu --detectors dcg --wavelet --iteration-chunk 512 --splitting '(3,4)' --quiet"

# GPU benchmarks
# run_benchmark "gpu_gaussian_no_wavelet" \
#   "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cuda --detectors gaussian --splitting '(3,3)' --quiet"
#
# run_benchmark "gpu_gaussian_wavelet" \
#   "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cuda --detectors gaussian --wavelet --splitting '(15,15)' --quiet"
#
# run_benchmark "gpu_dcg_no_wavelet" \
#   "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cuda --detectors dcg --splitting '(15,15)' --iteration-chunk 512 --quiet"

run_benchmark "gpu_dcg_wavelet" \
  "uv run ${COMPUTE_SCRIPT} ${DATA} ${WINDOW_SIZE} --backend torch-cuda --detectors dcg --wavelet --splitting '(31,31)' --iteration-chunk 512 --quiet"

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

from matplotlib.patches import Patch

def make_chart(rows, detector_name, results_dir):
    if not rows:
        print(f"  No data for {detector_name}, skipping chart.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    means = [r["mean_s"] for r in rows]
    stds  = [r["std_s"]  for r in rows]
    tick_labels = [r["label"] for r in rows]
    colors = ["steelblue" if r["label"].startswith("cpu") else "darkorange" for r in rows]

    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{detector_name} detection time (mean ± std)")
    ax.legend(handles=[
        Patch(color="steelblue",  label="CPU"),
        Patch(color="darkorange", label="GPU"),
    ])
    fig.tight_layout()

    stem = f"benchmark_{detector_name.lower()}"
    png_path = results_dir / f"{stem}.png"
    tex_path = results_dir / f"{stem}.tex"
    fig.savefig(png_path, dpi=150)
    print(f"Chart saved to {png_path}")
    matplot2tikz.save(str(tex_path))
    print(f"TikZ saved to {tex_path}")
    plt.close(fig)

gaussian_rows = [r for r in rows if "gaussian" in r["label"]]
dcg_rows      = [r for r in rows if "dcg"      in r["label"]]

make_chart(gaussian_rows, "Gaussian", results_dir)
make_chart(dcg_rows,      "DCG",      results_dir)
EOF

echo "Done. Results in ${RESULTS_DIR}/"
