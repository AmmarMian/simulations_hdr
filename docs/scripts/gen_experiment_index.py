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
import html
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found — run: uv add pyyaml --dev")

REPO_ROOT = Path(__file__).parent.parent.parent

# Base for the "view on GitHub" button on each experiment page.
GITHUB_BLOB = "https://github.com/AmmarMian/simulations_hdr/blob/main"

CHAPTERS = [
    ("1-context",        "1 · Context",          "chapters/1-context.md"),
    ("2-detection",      "2 · Detection",         "chapters/2-detection.md"),
    ("3-learning",       "3 · Learning",          "chapters/3-learning.md"),
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
    """Read actual run args from the provenance JSON recorded in the source sidecar.

    Prefers a sidecar matching the data file's own stem (e.g. gaussian_offline.npy
    + gaussian_offline.json, written by ResultExporter). Falls back to any *.json
    sidecar in the run directory with an "args" key — scripts using the simpler
    write_prov_sidecar() helper name sidecars after each figure (mean.json,
    cov.json, ...) rather than after the shared results.npz/npy.
    """
    source_txt = DATA_DIR / f"{stem}.source.txt"
    if not source_txt.exists():
        return {}
    data_path = Path(source_txt.read_text().strip())
    import json

    candidates = [data_path.with_suffix(".json")]
    if data_path.parent.is_dir():
        candidates += sorted(data_path.parent.glob("*.json"))
    for prov_path in candidates:
        if not prov_path.exists():
            continue
        try:
            args = json.loads(prov_path.read_text()).get("args", {})
        except Exception:
            continue
        if args:
            return args
    return {}


LOG_MAX_CHARS = 20000


def _load_run_logs(stem: str) -> dict:
    """Read stdout.txt/stderr.txt from the run directory that produced this figure.

    The run directory is the parent of the .npz recorded in the source sidecar.
    """
    source_txt = DATA_DIR / f"{stem}.source.txt"
    if not source_txt.exists():
        return {}
    npz_path = Path(source_txt.read_text().strip())
    run_dir = npz_path.parent
    logs = {}
    for kind in ("stdout", "stderr"):
        log_path = run_dir / f"{kind}.txt"
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="replace")
        if len(text) > LOG_MAX_CHARS:
            text = "… (truncated) …\n" + text[-LOG_MAX_CHARS:]
        logs[kind] = text
    return logs


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


SRC_MAX_LINES = 1200


def _highlight_python(code: str) -> str:
    """Syntax-highlight source at build time.

    The markdown pipeline does not process fenced blocks nested inside raw
    HTML, so the <details> body is rendered here with the same Pygments
    class names the rest of the site's code blocks use.
    """
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
    except ImportError:
        return f'<div class="highlight"><pre>{html.escape(code)}</pre></div>'
    return highlight(
        code, PythonLexer(),
        HtmlFormatter(cssclass="highlight", nowrap=False),
    )


def _source_block(exe: str) -> str:
    """Render the collapsible source view plus the GitHub link for one script."""
    if not exe:
        return ""
    rel = exe.lstrip("./")
    src_path = REPO_ROOT / rel
    gh_url = f"{GITHUB_BLOB}/{rel}"

    gh_icon = (
        '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">'
        '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
        '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15'
        '-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51'
        '-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 '
        '0 0 .67-.21 2.2.82a7.42 7.42 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 '
        '2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95'
        '.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 '
        '8c0-4.42-3.58-8-8-8z"/></svg>'
    )
    code_icon = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>'
    )
    chev = (
        '<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<polyline points="6 9 12 15 18 9"/></svg>'
    )

    gh_btn = (
        f'<a class="src-btn" href="{gh_url}" target="_blank" rel="noopener">'
        f'{gh_icon}<span>View on GitHub</span></a>'
    )

    if not src_path.exists():
        return f'<div class="src-bar">{gh_btn}</div>'

    code = src_path.read_text(errors="replace")
    lines = code.splitlines()
    if len(lines) > SRC_MAX_LINES:
        code = "\n".join(lines[:SRC_MAX_LINES]) + f"\n\n# … truncated at {SRC_MAX_LINES} lines — see GitHub for the rest"
    n_lines = len(lines)

    return (
        '<div class="src-bar">\n'
        f'{gh_btn}\n'
        '</div>\n'
        '<details class="src-view">\n'
        f'<summary><span class="src-btn">{code_icon}'
        f'<span>Source code</span><span class="param-alias">{n_lines} lines</span>{chev}'
        '</span></summary>\n'
        '<div class="src-body">\n'
        f'<p class="src-path">{html.escape(rel)}</p>\n'
        f'{_highlight_python(code)}\n'
        '</div>\n'
        '</details>'
    )


def _param_cards(params: list[dict]) -> str:
    """Render argparse params as a responsive card list.

    A markdown table forces horizontal scrolling on a phone; each parameter
    gets its own card instead, which reflows to one or two columns.
    """
    cards = []
    for arg in params:
        names   = arg["names"]
        flag    = html.escape(names[0])
        aliases = "".join(
            f'<span class="param-alias">{html.escape(n)}</span>' for n in names[1:]
        )
        typ = str(arg.get("type", "")).replace("<class \'", "").replace("\'>", "")
        if not typ and arg.get("action") in ("store_true", "store_false"):
            typ = "flag"
        type_html = f'<span class="param-type">{html.escape(typ)}</span>' if typ else ""

        dflt = arg.get("default")
        dflt_html = (
            f'<span class="param-default">default <b>{html.escape(str(dflt))}</b></span>'
            if dflt is not None else ""
        )
        hlp = html.escape(arg.get("help") or "")
        help_html = f'<p class="param-help">{hlp}</p>' if hlp else ""

        choices = arg.get("choices")
        choices_html = (
            f'<p class="param-choices">choices: {html.escape(", ".join(map(str, choices)))}</p>'
            if isinstance(choices, (list, tuple)) and choices else ""
        )

        cards.append(
            '<div class="param">\n'
            '<div class="param-head">\n'
            f'<span class="param-flag">{flag}</span>{aliases}{type_html}{dflt_html}\n'
            '</div>\n'
            f'{help_html}{choices_html}\n'
            '</div>'
        )
    return '<div class="params">\n' + "\n".join(cards) + "\n</div>"


def _write_exp_page(exp: dict, chapter: "tuple[str, str] | None" = None) -> None:
    """Write a dedicated markdown page for one experiment.

    ``chapter`` is the (label, slug) of the chapter that owns the experiment;
    it is used for the breadcrumb and the back-link, since experiment pages
    are excluded from the nav and would otherwise be a dead end.
    """
    name  = exp.get("name", "unknown")
    desc  = exp.get("description", "")
    tags  = exp.get("tags", []) or []
    exe   = exp.get("executable", "")
    yaml_ = exp.get("_yaml", "")
    cmd_prefix = exp.get("executable_command") or "uv run python"

    params = extract_args(exe) if exe else []

    crumbs = ""
    if chapter:
        label, slug = chapter
        crumbs = (
            '<nav class="crumbs" aria-label="Breadcrumb">\n'
            '<a href="../../experiments-overview/">Experiments</a>\n'
            '<span class="sep">/</span>\n'
            f'<a href="../../chapters/{slug}/">{html.escape(label)}</a>\n'
            '<span class="sep">/</span>\n'
            f'<span class="here">{html.escape(name)}</span>\n'
            '</nav>'
        )

    lines = [
        crumbs,
        "",
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
            f"{cmd_prefix} {exe}",
            "```",
            "",
            _source_block(exe),
            "",
        ]

    if params:
        lines += ["## Parameters", "", _param_cards(params), ""]

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
            logs = _load_run_logs(stem)
            log_blocks = ""
            for kind in ("stdout", "stderr"):
                text = logs.get(kind)
                if not text:
                    continue
                log_blocks += (
                    f'<details class="exp-log">\n'
                    f'<summary>{kind}</summary>\n'
                    f'<div class="exp-log-text">{html.escape(text)}</div>\n'
                    f'</details>\n'
                )
            lines += [
                marginnote +
                '<div class="exp-result-card">\n'
                f'<div class="plotly-wrap" '
                f'data-src="../../assets/data/{stem}.json" '
                f'data-title="{fig_title}"></div>\n'
                f'{log_blocks}'
                '</div>',
                "",
            ]

    if yaml_:
        lines += [
            "## Config",
            "",
            f"`{yaml_}`",
            "",
        ]

    if chapter:
        label, slug = chapter
        lines += [
            f'<a class="back-link" href="../../chapters/{slug}/">'
            f'← All experiments in {html.escape(label)}</a>',
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
    cmd_prefix = exp.get("executable_command") or "uv run python"

    tag_html = "".join(f'<span class="exp-tag">{t}</span>' for t in tags)
    tags_block = f'<div class="exp-tags">{tag_html}</div>' if tags else ""

    run_cmd = f'<div class="exp-run"><code>{cmd_prefix} {exe}</code></div>' if exe else ""

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
        chapter_slug = Path(md_rel).stem
        for exp in exps:
            _write_exp_page(exp, chapter=(label, chapter_slug))

        block = _chapter_block(exps, label)
        if _inject(chapter_md, block):
            print(f"  OK  {chapter_md.name} ({len(exps)} experiments)")
        # If no marker, silently skip — not all chapters have experiments yet

    print(f"\nDone — {total} experiments across {len(CHAPTERS)} chapters.")


if __name__ == "__main__":
    main()
