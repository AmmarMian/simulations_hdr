#!/usr/bin/env bash
# Memory benchmark for detection scripts.
# CPU: uses memray -- generates flamegraph + peak summary.
# GPU: uses torch.cuda.max_memory_allocated() reported by the script itself.
# Results saved per config, summary printed at end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SCRIPT_DIR}/../../data/SAR/Scene_1.npy"
WINDOW_SIZE=7
RESULTS_DIR="${SCRIPT_DIR}/memory_results"

# Parse --storage_path / --storage-path from CLI args (qanat passes this).
while [[ $# -gt 0 ]]; do
  case "$1" in
    --storage_path|--storage-path) RESULTS_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p "$RESULTS_DIR"

SCRIPT_GAUSSIAN="${SCRIPT_DIR}/../compute_detection_real_data/offline_gaussian.py"
SCRIPT_DCG="${SCRIPT_DIR}/../compute_detection_real_data/offline_dcg.py"

TOTAL=8
CURRENT=0

# ---- CPU: run under memray, extract peak RSS from stats output ---------------
run_cpu_memory() {
  local label="$1"
  local script="$2"
  local extra_args="${3:-}"
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "[$CURRENT/$TOTAL] === $label (CPU memray) ==="

  local bin="${RESULTS_DIR}/${label}.bin"
  local html="${RESULTS_DIR}/${label}.html"

  uv run python -m memray run --force -o "$bin" \
    "$script" "$DATA" "$WINDOW_SIZE" $extra_args --quiet

  uv run python -m memray flamegraph --force -o "$html" "$bin"
  echo "  Flamegraph: $html"

  # Extract peak memory from memray stats.
  # Use grep -E + head for macOS compatibility (grep -oP is GNU-only).
  local peak
  peak=$(uv run python -m memray stats "$bin" 2>/dev/null \
    | grep -i "peak memory" \
    | grep -Eo '[0-9]+(\.[0-9]+)? [A-Za-z]+' \
    | head -1 || echo "N/A")
  echo "  Peak memory: $peak"
  echo "${label},${peak}" >>"${RESULTS_DIR}/memory_summary.csv"
}

# ---- GPU: script prints PEAK_GPU_MEMORY_BYTES=<n>, we parse it ---------------
run_gpu_memory() {
  local label="$1"
  local script="$2"
  local extra_args="${3:-}"
  CURRENT=$((CURRENT + 1))
  echo ""
  echo "[$CURRENT/$TOTAL] === $label (GPU torch) ==="

  local output
  output=$(uv run "$script" "$DATA" "$WINDOW_SIZE" \
    --backend torch-cuda $extra_args --quiet --report-memory 2>&1)

  local peak_bytes
  peak_bytes=$(echo "$output" | grep "^PEAK_GPU_MEMORY_BYTES=" | cut -d= -f2 || echo "")

  if [ -n "$peak_bytes" ]; then
    local peak_mb
    peak_mb=$(echo "scale=1; $peak_bytes / 1048576" | bc)
    echo "  Peak GPU memory: ${peak_mb} MB (${peak_bytes} bytes)"
    echo "${label},${peak_mb} MB" >>"${RESULTS_DIR}/memory_summary.csv"
  else
    echo "  Could not parse GPU memory."
    echo "${label},N/A" >>"${RESULTS_DIR}/memory_summary.csv"
  fi
}

# ---- Init summary CSV --------------------------------------------------------
echo "label,peak_memory" >"${RESULTS_DIR}/memory_summary.csv"

# ---- CPU benchmarks ----------------------------------------------------------
run_cpu_memory "cpu_gaussian_no_wavelet" "$SCRIPT_GAUSSIAN" \
  "--backend torch-cpu"

run_cpu_memory "cpu_gaussian_wavelet" "$SCRIPT_GAUSSIAN" \
  "--backend torch-cpu --wavelet"

run_cpu_memory "cpu_dcg_no_wavelet" "$SCRIPT_DCG" \
  "--backend torch-cpu --iteration-chunk 512"

run_cpu_memory "cpu_dcg_wavelet" "$SCRIPT_DCG" \
  "--backend torch-cpu --wavelet --iteration-chunk 512 --splitting (3,4)"

# ---- GPU benchmarks ----------------------------------------------------------
run_gpu_memory "gpu_gaussian_no_wavelet" "$SCRIPT_GAUSSIAN" \
  "--splitting (3,3)"

run_gpu_memory "gpu_gaussian_wavelet" "$SCRIPT_GAUSSIAN" \
  "--wavelet --splitting (15,15)"

run_gpu_memory "gpu_dcg_no_wavelet" "$SCRIPT_DCG" \
  "--splitting (15,15) --iteration-chunk 512"

run_gpu_memory "gpu_dcg_wavelet" "$SCRIPT_DCG" \
  "--wavelet --splitting (31,31) --iteration-chunk 512"

echo ""
echo "Memory benchmark done. Summary:"
cat "${RESULTS_DIR}/memory_summary.csv"
echo ""
echo "Flamegraphs and raw .bin profiles saved in ${RESULTS_DIR}/"
