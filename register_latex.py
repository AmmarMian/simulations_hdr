#!/usr/bin/env python
"""Qanat action: stage a figure for the HDR LaTeX dissertation.

Copies the PGFPlots .tex + merged provenance JSON from a qanat run directory
into ./hdr_exports/<id>/.  The user then manually copies/rsyncs from
hdr_exports/ to the dissertation repo's figures/generated/ when on the
right machine.

Run via qanat:
    qanat action run <exp> register <run_id> -- \\
        --id <fig-id> [--stem <stem>] [--seed-label deterministic] [--embed-npy]

Or directly (for testing):
    uv run python register_latex.py \\
        --storage_path results/sar_det_off_gauss/run_1 \\
        --id det-gauss-scene1 --seed-label deterministic

The action is intentionally repo-agnostic: it only writes to ./hdr_exports/
so it works on any machine (cluster, laptop, remote) without needing the
dissertation repo to be present.

Copy command (example):
    rsync -av hdr_exports/ ../hdr/figures/generated/

Requires a PGFPlots .tex file in the run directory (produced by --export-tikz
or --tikz when running the experiment).  The exporter .json sidecar is optional
— if absent, provenance is taken from qanat's info.yaml only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found — run: uv add pyyaml --dev")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--storage_path", required=True,
    help="Run directory injected by qanat (e.g. results/exp_name/run_1).")
parser.add_argument("--id", required=True, dest="fig_id",
    help="Stable figure id used in LaTeX, e.g. 'scm-mean-error'. "
         "Becomes hdr_exports/<id>/ and figures/generated/<id>/ in the hdr repo.")
parser.add_argument("--stem", default=None,
    help="Base filename of the .tex (without extension). "
         "Auto-detected when exactly one .tex is present; required when several exist.")
parser.add_argument("--seed-label", default=None, dest="seed_label",
    help="Seed value for deterministic experiments (e.g. '--seed-label deterministic'). "
         "Required when the run has no 'seed' in its exporter sidecar args.")
parser.add_argument("--embed-npy", action="store_true", dest="embed_npy",
    help="Also copy the .npy data array (for \\embedfigdata in LaTeX).")
parser.add_argument("--export-dir", default="hdr_exports", dest="export_dir",
    help="Local staging directory for LaTeX exports. Default: ./hdr_exports")
args = parser.parse_args()

# ── Resolve paths ─────────────────────────────────────────────────────────────
run_dir = Path(args.storage_path).resolve()
if not run_dir.is_dir():
    sys.exit(f"✗  Run directory not found: {run_dir}")

export_base = Path(args.export_dir)
fig_id = args.fig_id

# ── Read qanat info.yaml ──────────────────────────────────────────────────────
info_path = run_dir / "info.yaml"
if not info_path.exists():
    sys.exit(f"✗  info.yaml not found in {run_dir}")
info = yaml.safe_load(info_path.read_text())

exp_name   = run_dir.parent.name
run_id     = info.get("run_id", "?")
commit_sha = info.get("commit_sha", "unknown")
start_times = info.get("start_time", [])
run_date   = str(start_times[0])[:10] if start_times else "unknown"

# ── Find .tex sidecar in run dir ──────────────────────────────────────────────
tex_files = sorted(p for p in run_dir.glob("*.tex")
                   if not p.name.startswith("_"))
if not tex_files:
    sys.exit(
        "✗  No .tex figure found in:\n"
        f"   {run_dir}\n"
        "\n"
        "   Rerun the experiment with --export-tikz (or --tikz) to produce a\n"
        "   PGFPlots .tex file, then register again.\n"
        "\n"
        "   For MC experiments (mc.py / hdrlib.core.mc): pass --tikz\n"
        "   For detection experiments (offline_*.py):    pass --export-tikz\n"
        "   For context scripts:                         use --storage_path <run_dir>"
    )

if args.stem:
    tex_path = run_dir / f"{args.stem}.tex"
    if not tex_path.exists():
        sys.exit(f"✗  {args.stem}.tex not found in {run_dir}")
else:
    if len(tex_files) > 1:
        names = "\n    ".join(p.stem for p in tex_files)
        sys.exit(
            f"✗  Multiple .tex files in {run_dir} — pick one with --stem:\n"
            f"    {names}"
        )
    tex_path = tex_files[0]

stem = tex_path.stem

# ── Read exporter sidecar .json (optional) ────────────────────────────────────
json_path = run_dir / f"{stem}.json"
npy_path  = run_dir / f"{stem}.npy"

sidecar: dict = {}
if json_path.exists():
    sidecar = json.loads(json_path.read_text())
else:
    print(f"  (no .json sidecar found — using qanat info.yaml for provenance only)")

# ── Seed validation ───────────────────────────────────────────────────────────
sidecar_args = sidecar.get("args", {})
seed_val = sidecar_args.get("seed")
if seed_val is None and args.seed_label is None:
    sys.exit(
        f"✗  No 'seed' found in {json_path.name if json_path.exists() else 'sidecar'}\n"
        "\n"
        "   For deterministic experiments, pass:  --seed-label deterministic\n"
        "   For stochastic experiments without a sidecar, pass: --seed-label <value>\n"
        "   For stochastic experiments with a sidecar, rerun with --seed <N>."
    )
seed_str = str(seed_val) if seed_val is not None else args.seed_label

# ── Build merged provenance dict ──────────────────────────────────────────────
sha_short = commit_sha[:7] if len(commit_sha) >= 7 else commit_sha
run_month = run_date[:7]  # YYYY-MM for the LaTeX stamp

prov = {
    **sidecar,
    "hdr": {
        "id":            fig_id,
        "exp":           exp_name,
        "run_id":        run_id,
        "commit_sha":    commit_sha,
        "sha_short":     sha_short,
        "run_date":      run_date,
        "run_month":     run_month,
        "seed":          seed_str,
        "embed_npy":     args.embed_npy,
        "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    },
}

# ── Write into hdr_exports/ ───────────────────────────────────────────────────
dest_dir = export_base / fig_id
dest_dir.mkdir(parents=True, exist_ok=True)

shutil.copy2(tex_path, dest_dir / "plot.tex")
(dest_dir / "prov.json").write_text(json.dumps(prov, indent=2, default=str))

print(f"Written to {dest_dir}/")
print(f"  plot.tex")
print(f"  prov.json")

if args.embed_npy:
    if npy_path.exists():
        shutil.copy2(npy_path, dest_dir / "data.npy")
        print(f"  data.npy")
    else:
        print(f"  [warn] --embed-npy requested but {npy_path.name} not found; skipping.")

# ── Update hdr_exports/figures.toml ──────────────────────────────────────────
toml_path = export_base / "figures.toml"
existing = toml_path.read_text().splitlines() if toml_path.exists() else [
    "# figures.toml — staged figures for the HDR dissertation",
    "# Copy hdr_exports/ → hdr/figures/generated/ then run 'just figures' in the hdr repo.",
    "#",
    "# rsync -av hdr_exports/ ../hdr/figures/generated/",
    "",
]

# Remove stale block for this fig_id
new_lines: list[str] = []
skip = False
for line in existing:
    if line.strip() == f"[figures.{fig_id}]":
        skip = True
    elif skip and line.strip().startswith("["):
        skip = False
    if not skip:
        new_lines.append(line)

while new_lines and not new_lines[-1].strip():
    new_lines.pop()

new_lines += [
    "",
    f"[figures.{fig_id}]",
    f'exp      = "{exp_name}"',
    f"run_id   = {run_id}",
    f'sha      = "{sha_short}"',
    f'run_date = "{run_date}"',
    f'seed     = "{seed_str}"',
    f'stem     = "{stem}"',
    f"embed_npy = {'true' if args.embed_npy else 'false'}",
]
toml_path.write_text("\n".join(new_lines) + "\n")
print(f"  {toml_path}  →  [{fig_id}]")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n✓  Staged '{fig_id}'")
print(f"   exp={exp_name}  run={run_id}  sha={sha_short}  seed={seed_str}")
print()
print(f"Copy to the hdr repo when ready:")
print(f"  rsync -av {export_base}/ ../hdr/figures/generated/")
print(f"Then run in the hdr repo:")
print(f"  just figures")
print()
print(f"In your chapter:")
print(f"  \\begin{{figure}}")
print(f"    \\input{{figures/generated/{fig_id}/plot.tex}}")
print(f"    \\caption{{...\\dataref{{{fig_id}}}}}")
print(f"  \\end{{figure}}")
