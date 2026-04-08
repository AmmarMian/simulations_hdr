#!/usr/bin/env bash
# Memory benchmark for detection scripts.
# CPU: uses memray (pip install memray) — generates flamegraph + peak summary.
# GPU: uses torch.cuda.max_memory_allocated() reported by the script itself.
# Results saved per config, summary printed at end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SCRIPT_DIR}/../../data/SAR/Scene_1.npy"
WINDOW_SIZE=7
RESULTS_DIR="${SCRIPT_DIR}/memory_results"

mkdir -p "$RESULTS_DIR"

COMPUTE_SCRIPT="${SCRIPT_DIR}/../compute_cd_offline.py"

TOTAL=8
CURRENT=0

# ── CPU: run under memray, extract peak RSS from stats output ──────────────────
run_cpu_memory() {
  local label="$1"
  local extra_args="${2:-}"
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "[$CURRENT/$TOTAL] === $label (CPU memray) ==="

  local bin="${RESULTS_DIR}/${label}.bin"
  local html="${RESULTS_DIR}/${label}.html"

  uv run memray run --force -o "$bin" \
    "$COMPUTE_SCRIPT" "$DATA" "$WINDOW_SIZE" $extra_args --quiet

  uv run memray flamegraph --force -o "$html" "$bin"
  echo "  Flamegraph: $html"

  # Extract peak memory from memray stats (grep the "Peak memory" line)
  local peak
  peak=$(uv run memray stats "$bin" 2>/dev/null | grep -i "peak memory" | grep -oP '[\d.]+ \w+' | head -1 || echo "N/A")
  echo "  Peak memory: $peak"
  echo "${label},${peak}" >> "${RESULTS_DIR}/memory_summary.csv"
}

# ── GPU: script prints PEAK_GPU_MEMORY_BYTES=<n>, we parse it ─────────────────
run_gpu_memory() {
  local label="$1"
  local extra_args="${2:-}"
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "[$CURRENT/$TOTAL] === $label (GPU torch) ==="

  local output
  output=$(uv run "$COMPUTE_SCRIPT" "$DATA" "$WINDOW_SIZE" \
    --backend torch-cuda $extra_args --quiet --report-memory 2>&1)

  local peak_bytes
  peak_bytes=$(echo "$output" | grep "^PEAK_GPU_MEMORY_BYTES=" | cut -d= -f2 || echo "")

  if [ -n "$peak_bytes" ]; then
    # Convert bytes to MB for readability
    local peak_mb
    peak_mb=$(echo "scale=1; $peak_bytes / 1048576" | bc)
    echo "  Peak GPU memory: ${peak_mb} MB (${peak_bytes} bytes)"
    echo "${label},${peak_mb} MB" >> "${RESULTS_DIR}/memory_summary.csv"
  else
    echo "  Could not parse GPU memory."
    echo "${label},N/A" >> "${RESULTS_DIR}/memory_summary.csv"
  fi
}

# ── Init summary CSV ───────────────────────────────────────────────────────────
echo "label,peak_memory" > "${RESULTS_DIR}/memory_summary.csv"

# ── CPU benchmarks ─────────────────────────────────────────────────────────────
run_cpu_memory "cpu_gaussian_no_wavelet" \
  "--backend torch-cpu --detectors gaussian"

# run_cpu_memory "cpu_gaussian_wavelet" \
#   "--backend torch-cpu --detectors gaussian --wavelet"

run_cpu_memory "cpu_dcg_no_wavelet" \
  "--backend torch-cpu --detectors dcg --iteration-chunk 512"

# run_cpu_memory "cpu_dcg_wavelet" \
#   "--backend torch-cpu --detectors dcg --wavelet --iteration-chunk 512 --splitting '(3,4)'"

# ── GPU benchmarks ─────────────────────────────────────────────────────────────
run_gpu_memory "gpu_gaussian_no_wavelet" \
  "--detectors gaussian --splitting '(3,3)'"

# run_gpu_memory "gpu_gaussian_wavelet" \
#   "--detectors gaussian --wavelet --splitting '(15,15)'"

# run_gpu_memory "gpu_dcg_no_wavelet" \
#   "--detectors dcg --splitting '(15,15)' --iteration-chunk 512"

run_gpu_memory "gpu_dcg_wavelet" \
  "--detectors dcg --wavelet --splitting '(31,31)' --iteration-chunk 512"

echo ""
echo "Memory benchmark done. Summary:"
cat "${RESULTS_DIR}/memory_summary.csv"
echo ""
echo "Flamegraphs and raw .bin profiles saved in ${RESULTS_DIR}/"
