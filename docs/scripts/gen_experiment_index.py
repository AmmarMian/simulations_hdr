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


def _build_import_map(tree, script_dir: Path) -> dict[str, Path]:
    """Map imported names → source file path.

    Handles both absolute imports (searches script dir then repo root) and
    relative imports (e.g. ``from ..core.mc import`` resolves via level+parent).
    """
    import_map: dict[str, Path] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        rel = node.module.replace(".", "/") + ".py"
        candidates: list[Path] = []
        if node.level:
            # Relative import: go up `level` directories from script_dir
            base = script_dir
            for _ in range(node.level - 1):
                base = base.parent
            candidates.append(base / rel)
        else:
            # Absolute import: try script dir, then repo root
            candidates = [script_dir / rel, REPO_ROOT / rel]
        for mod_path in candidates:
            if mod_path.exists():
                for alias in node.names:
                    import_map[alias.asname or alias.name] = mod_path
                break
    return import_map


def _parser_passing_calls(tree_node) -> list[str]:
    """Return names of functions called with a 'parser'-named argument.

    Handles both plain calls (``add_mc_args(parser)``) and attribute calls
    (``smc.add_mc_args(parser)``), returning the function name in both cases.
    """
    names = []
    for node in ast.walk(tree_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            fn = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn = node.func.attr
        else:
            continue
        if fn == "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name) and "parser" in arg.id.lower():
                names.append(fn)
                break
    return names


def _build_module_alias_map(tree, script_dir: Path) -> dict[str, Path]:
    """Map module aliases to their source file (e.g. ``import mc as smc`` → mc.py)."""
    alias_map: dict[str, Path] = {}
    search_roots = [script_dir, REPO_ROOT]
    for node in ast.walk(tree):
        # ``from hdrlib.sonar import mc as smc``
        if isinstance(node, ast.ImportFrom) and node.module:
            base_rel = node.module.replace(".", "/")
            for alias in node.names:
                name = alias.asname or alias.name
                rel = f"{base_rel}/{alias.name}.py"
                if node.level:
                    base = script_dir
                    for _ in range(node.level - 1):
                        base = base.parent
                    candidates = [base / rel]
                else:
                    candidates = [script_dir / rel, REPO_ROOT / rel]
                for p in candidates:
                    if p.exists():
                        alias_map[name] = p
                        break
        # ``import hdrlib.sonar.mc as smc``
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[-1]
                rel = alias.name.replace(".", "/") + ".py"
                for root in search_roots:
                    p = root / rel
                    if p.exists():
                        alias_map[name] = p
                        break
    return alias_map


def _find_helper_functions(script_tree, script_dir: Path) -> list[ast.AST]:
    """Return AST FunctionDef nodes for all helpers that add argparse arguments.

    Follows the call chain recursively: if add_mc_args calls add_mc_base_args,
    both function bodies are returned so their add_argument() calls are captured.
    Handles plain calls (add_mc_args(parser)) and attribute calls (smc.add_mc_args(parser)).
    """
    import_map   = _build_import_map(script_tree, script_dir)
    module_aliases = _build_module_alias_map(script_tree, script_dir)

    def _resolve(fn_name: str, call_node: ast.Call,
                 imap: dict[str, Path], mmap: dict[str, Path]) -> "Path | None":
        """Return the source file for a parser-passing call."""
        func = call_node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            # smc.add_mc_args(parser) → look up the module alias
            return mmap.get(func.value.id)
        return imap.get(fn_name)

    subtrees: list[ast.AST] = []
    seen: set[tuple[Path, str]] = set()
    # Seed the queue from the top-level script
    queue: list[tuple[str, "Path | None", dict, dict]] = []
    for node in ast.walk(script_tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            fn = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn = node.func.attr
        else:
            continue
        if fn == "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name) and "parser" in arg.id.lower():
                queue.append((fn, _resolve(fn, node, import_map, module_aliases),
                              import_map, module_aliases))
                break

    while queue:
        name, mod_path, imap, mmap = queue.pop()
        if not mod_path or (mod_path, name) in seen:
            continue
        seen.add((mod_path, name))
        try:
            mod_tree = ast.parse(mod_path.read_text())
        except Exception:
            continue
        mod_imap = _build_import_map(mod_tree, mod_path.parent)
        mod_mmap = _build_module_alias_map(mod_tree, mod_path.parent)
        for node in ast.walk(mod_tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                subtrees.append(node)
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    if isinstance(child.func, ast.Name):
                        cfn = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        cfn = child.func.attr
                    else:
                        continue
                    if cfn == "add_argument":
                        continue
                    for arg in child.args:
                        if isinstance(arg, ast.Name) and "parser" in arg.id.lower():
                            queue.append((cfn,
                                          _resolve(cfn, child, mod_imap, mod_mmap),
                                          mod_imap, mod_mmap))
                            break
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
            fig_title = f"{name} — {label}" if label else name
            json_path = DATA_DIR / f"{stem}.json"
            run_date = ""
            if json_path.exists():
                from datetime import datetime
                mtime = json_path.stat().st_mtime
                run_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            mn_rows = ""
            if run_date:
                mn_rows += f'  <span class="mn-date">Generated: {run_date}</span><br>\n'
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
    for f in sorted(exp_dir.rglob("*.yaml")):
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
            '<div class="exp-chapter">\n'
            '<p class="exp-empty">No experiments registered yet.</p>\n'
            '</div>'
        )

    # Group by the optional `group` field; ungrouped experiments go under "".
    groups: dict[str, list[dict]] = {}
    for exp in exps:
        g = exp.get("group", "")
        groups.setdefault(g, []).append(exp)

    sections: list[str] = []
    for group_name, group_exps in groups.items():
        cards = "\n\n".join(_card(e) for e in group_exps)
        grid  = f'<div class="exp-grid">\n{cards}\n</div>'
        if group_name:
            header = f'<h3 class="exp-group-heading">{group_name}</h3>'
            sections.append(f'<div class="exp-group">\n{header}\n{grid}\n</div>')
        else:
            sections.append(f'<div class="exp-group">\n{grid}\n</div>')

    inner = "\n\n".join(sections)
    return f'<div class="exp-chapter">\n{inner}\n</div>'


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
