# Convert a SAR SITS .npy file from (rows, cols, features, times)
# to (times, rows, cols, features) for memory-efficient online processing.

import sys
import argparse
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert SITS .npy from (rows, cols, features, times) to (times, rows, cols, features)."
    )
    parser.add_argument("input", type=str, help="Input .npy file.")
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Output .npy file. Defaults to <input>_time_first.npy.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + "_time_first")

    print(f"Loading {input_path} ...")
    data = np.load(input_path, mmap_mode="r")
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")

    if data.ndim != 4:
        print(f"ERROR: expected 4D array, got {data.ndim}D.")
        sys.exit(1)

    n_rows, n_cols, n_features, n_times = data.shape
    print(f"  Interpreting as (rows={n_rows}, cols={n_cols}, features={n_features}, times={n_times})")
    print(f"Saving to {output_path} as (times, rows, cols, features) ...")

    # Write time-first contiguous array
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=data.dtype,
        shape=(n_times, n_rows, n_cols, n_features),
    )
    for t in range(n_times):
        out[t] = data[..., t]
        if (t + 1) % 10 == 0 or t == n_times - 1:
            print(f"  {t + 1}/{n_times} time steps written", end="\r")
    print()

    print("Done.")
