#!/usr/bin/env python
"""Inject experiment cards into each chapter's markdown page.

Each chapter .md must contain the sentinel pair:
    <!-- experiments-start -->
    <!-- experiments-end -->

The script replaces everything between those markers with generated HTML cards.
Argparse arguments are extracted via AST from each experiment's executable script
(and any helper functions it imports from the same directory).

Run from the repo root:
    uv run python docs/scripts/gen_experiment_index.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found — run: uv add pyyaml --dev")

REPO_ROOT = Path(__file__).parent.parent.parent

CHAPTERS = [
    ("1-context",        "1 · Context",          "chapters/1-context.md"),
    ("2-detection",      "2 · Detection",         "chapters/2-detection.md"),
    ("3-machinelearning","3 · Machine Learning",  "chapters/3-machinelearning.md"),
    ("4-deeplearning",   "4 · Deep Learning",     "chapters/4-deeplearning.md"),
]

START_MARKER = "<!-- experiments-start -->"
END_MARKER   = "<!-- experiments-end -->"


# ── Argparse extraction ───────────────────────────────────────────────────────

def _ast_literal(node) -> Any:
    """Best-effort evaluation of an AST node to a Python value."""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{getattr(node.value, 'id', '?')}.{node.attr}"
    return None


def _collect_add_argument_calls(tree_nodes) -> list[dict]:
    """Walk AST nodes and collect all parser.add_argument() calls."""
    results = []
    for node in ast.walk(tree_nodes):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue

        # positional string args → flags like "--foo" / "-f"
        names = [_ast_literal(a) for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not names:
            continue

        info: dict[str, Any] = {"names": names}
        for kw in node.keywords:
            if kw.arg in ("type", "default", "help", "choices",
                          "action", "nargs", "required", "metavar", "dest"):
                info[kw.arg] = _ast_literal(kw.value)
        results.append(info)
    return results


def _find_helper_functions(script_tree, script_dir: Path) -> list[ast.AST]:
    """
    Find helper functions (e.g. add_mc_args) called with a parser argument
    in the script, and return their AST subtrees from imported modules.
    """
    # Map imported names → source module path
    import_map: dict[str, Path] = {}
    for node in ast.walk(script_tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod_path = script_dir / (node.module.replace(".", "/") + ".py")
            if mod_path.exists():
                for alias in node.names:
                    import_map[alias.asname or alias.name] = mod_path

    # Find calls where a 'parser'-named variable is an argument
    helper_names: list[str] = []
    for node in ast.walk(script_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Name) and "parser" in arg.id.lower():
                    helper_names.append(fn)
                    break

    subtrees: list[ast.AST] = []
    seen_files: set[Path] = set()
    for name in helper_names:
        mod_path = import_map.get(name)
        if not mod_path or mod_path in seen_files:
            continue
        seen_files.add(mod_path)
        try:
            mod_tree = ast.parse(mod_path.read_text())
            for node in ast.walk(mod_tree):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    subtrees.append(node)
        except Exception:
            pass
    return subtrees


def extract_args(executable: str) -> list[dict]:
    """Extract argparse arguments from a script via AST."""
    script_path = REPO_ROOT / executable
    if not script_path.exists():
        return []
    try:
        source = script_path.read_text()
        tree = ast.parse(source)
    except Exception:
        return []

    script_dir = script_path.parent
    collected = _collect_add_argument_calls(tree)

    for helper_tree in _find_helper_functions(tree, script_dir):
        collected.extend(_collect_add_argument_calls(helper_tree))

    # Deduplicate by primary flag name, preserve order
    seen: set[str] = set()
    unique = []
    for arg in collected:
        key = arg["names"][0]
        if key not in seen:
            seen.add(key)
            unique.append(arg)
    return unique


# ── Per-experiment page ───────────────────────────────────────────────────────

EXP_PAGES_DIR  = REPO_ROOT / "docs" / "docs" / "experiments"
DATA_DIR       = REPO_ROOT / "docs" / "docs" / "assets" / "data"


def _load_actual_args(stem: str) -> dict:
    """Read actual run args from the provenance JSON recorded in the source sidecar."""
    source_txt = DATA_DIR / f"{stem}.source.txt"
    if not source_txt.exists():
        return {}
    npz_path = Path(source_txt.read_text().strip())
    prov_path = npz_path.with_suffix(".json")
    if not prov_path.exists():
        return {}
    try:
        import json
        return json.loads(prov_path.read_text()).get("args", {})
    except Exception:
        return {}


def _figure_stems(name: str) -> list[tuple[str, str]]:
    """Return (stem, label) pairs for all published figures for this experiment.

    Covers both {name}.json (no label) and {name}.{label}.json.
    Sorted: unlabelled first, then labels alphabetically.
    """
    results = []
    # unlabelled
    if (DATA_DIR / f"{name}.json").exists():
        results.append((name, ""))
    # labelled: {name}.{label}.json — exclude .source.txt companions
    for p in sorted(DATA_DIR.glob(f"{name}.*.json")):
        label = p.stem[len(name) + 1:]   # strip "{name}."
        if label != "source":             # skip {name}.source.txt edge case
            results.append((p.stem, label))
    return results


def _param_table_md(params: list[dict]) -> str:
    """Render argparse params as a markdown table."""
    rows = ["| Flag | Type | Default | Description |",
            "|------|------|---------|-------------|"]
    for arg in params:
        flag = " / ".join(f"`{n}`" for n in arg["names"])
        typ  = str(arg.get("type", "")).replace("<class '", "").replace("'>", "") or "—"
        dflt = arg.get("default")
        dflt_str = f"`{dflt}`" if dflt is not None else "—"
        hlp  = (arg.get("help") or "").replace("|", "\\|")
        rows.append(f"| {flag} | {typ} | {dflt_str} | {hlp} |")
    return "\n".join(rows)


def _write_exp_page(exp: dict) -> None:
    """Write a dedicated markdown page for one experiment."""
    name  = exp.get("name", "unknown")
    desc  = exp.get("description", "")
    tags  = exp.get("tags", []) or []
    exe   = exp.get("executable", "")
    yaml_ = exp.get("_yaml", "")

    params = extract_args(exe) if exe else []

    lines = [
        f"# {name}",
        "",
        f"{desc}" if desc else "",
        "",
    ]

    if tags:
        tag_str = "  ".join(f"`{t}`" for t in tags)
        lines += [f"**Tags:** {tag_str}", ""]

    if exe:
        lines += [
            "## Run",
            "",
            "```sh",
            f"uv run python {exe}",
            "```",
            "",
        ]

    if params:
        lines += ["## Parameters", "", _param_table_md(params), ""]

    # Embed interactive figures for all published runs
    figures = _figure_stems(name)
    if figures:
        _INFRA = {"storage_path", "export_path"}
        lines += ["## Results", ""]
        for stem, label in figures:
            actual_args = _load_actual_args(stem)
            fig_title = f"{name} — {label}" if label else f"{name} — power curve"
            mn_rows = ""
            if params:
                for arg in params:
                    flag = arg["names"][0]
                    dest = arg.get("dest") or flag.lstrip("-").replace("-", "_")
                    if dest in _INFRA:
                        continue
                    value = actual_args.get(dest, arg.get("default"))
                    val_str = f" <span class='mn-default'>{value}</span>" if value is not None else ""
                    mn_rows += f'  <code>{flag}</code>{val_str}<br>\n'
            marginnote = (
                '<span class="marginnote">\n'
                f'  <span class="mn-label">{"Run · " + label if label else "Parameters"}</span>\n'
                f'{mn_rows}'
                '</span>\n'
            ) if mn_rows else ""
            lines += [
                marginnote +
                f'<div class="plotly-wrap" '
                f'data-src="../../assets/data/{stem}.json" '
                f'data-title="{fig_title}"></div>',
                "",
            ]

    if yaml_:
        lines += [
            "## Config",
            "",
            f"`{yaml_}`",
            "",
        ]

    EXP_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    out = EXP_PAGES_DIR / f"{name}.md"
    out.write_text("\n".join(lines))


# ── Card HTML ─────────────────────────────────────────────────────────────────

def _card(exp: dict) -> str:
    name  = exp.get("name", "—")
    desc  = exp.get("description", "")
    tags  = exp.get("tags", []) or []
    exe   = exp.get("executable", "")

    tag_html = "".join(f'<span class="exp-tag">{t}</span>' for t in tags)
    tags_block = f'<div class="exp-tags">{tag_html}</div>' if tags else ""

    run_cmd = f'<div class="exp-run"><code>uv run python {exe}</code></div>' if exe else ""

    has_results = bool(_figure_stems(name))
    results_badge = (
        '<span class="exp-results-badge">Results available</span>'
        if has_results else ""
    )

    page_link = f'../../experiments/{name}/'
    details_link = (
        f'<a class="exp-details-link" href="{page_link}">Parameters &amp; details →</a>'
    )

    return (
        f'<div class="exp-card">\n'
        f'<div class="exp-card-head">\n'
        f'<div class="exp-name">{name}</div>\n'
        f'{results_badge}\n'
        f'</div>\n'
        f'<div class="exp-desc">{desc}</div>\n'
        f'{tags_block}\n'
        f'{run_cmd}\n'
        f'{details_link}\n'
        f'</div>'
    )


# ── Chapter injection ─────────────────────────────────────────────────────────

def _load_experiments(chapter_dir: Path) -> list[dict]:
    exp_dir = chapter_dir / "experiments"
    if not exp_dir.is_dir():
        return []
    result = []
    for f in sorted(exp_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text()) or {}
            data["_yaml"] = str(f.relative_to(REPO_ROOT))
            result.append(data)
        except Exception:
            pass
    return result


def _inject(chapter_md: Path, cards_html: str) -> bool:
    """Replace content between sentinel markers in a chapter markdown file."""
    text = chapter_md.read_text()
    if START_MARKER not in text:
        print(f"  SKIP {chapter_md.name} — no {START_MARKER!r} marker")
        return False

    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{cards_html}\n{END_MARKER}"
    new_text, n = pattern.subn(replacement, text)
    if n == 0:
        print(f"  SKIP {chapter_md.name} — END marker missing")
        return False

    chapter_md.write_text(new_text)
    return True


def _chapter_block(exps: list[dict], label: str) -> str:
    if not exps:
        return (
            f'<div class="exp-chapter">\n'
            f'<p class="exp-empty">No experiments registered yet.</p>\n'
            f'</div>'
        )
    cards = "\n\n".join(_card(e) for e in exps)
    return (
        f'<div class="exp-chapter">\n'
        f'<div class="exp-grid">\n{cards}\n</div>\n'
        f'</div>'
    )


def main() -> None:
    total = 0
    for dir_name, label, md_rel in CHAPTERS:
        chapter_md = REPO_ROOT / "docs" / "docs" / md_rel
        exps = _load_experiments(REPO_ROOT / dir_name)
        total += len(exps)

        # Write per-experiment pages
        for exp in exps:
            _write_exp_page(exp)

        block = _chapter_block(exps, label)
        if _inject(chapter_md, block):
            print(f"  OK  {chapter_md.name} ({len(exps)} experiments)")
        # If no marker, silently skip — not all chapters have experiments yet

    print(f"\nDone — {total} experiments across {len(CHAPTERS)} chapters.")


if __name__ == "__main__":
    main()
