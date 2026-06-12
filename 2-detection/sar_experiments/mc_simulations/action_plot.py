#!/usr/bin/env python
"""Qanat action: find and run all auto-generated plot scripts in a run directory."""

import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--storage_path", required=True,
    help="Run directory injected by qanat.")
parser.add_argument("--no-save", action="store_true",
    help="Pass --no-save to each plot script (display only).")
args = parser.parse_args()

root = Path(args.storage_path)
scripts = sorted(root.rglob("*_plot.py"))

if not scripts:
    print(f"No *_plot.py scripts found under {root}")
    sys.exit(0)

extra = ["--no-save"] if args.no_save else []
for script in scripts:
    print(f"  {script.relative_to(root)}")
    subprocess.run([sys.executable, str(script)] + extra, check=True)
