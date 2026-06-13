import subprocess
from datetime import datetime
from pathlib import Path


def on_page_context(context, page, config, nav):
    try:
        src = Path(config["docs_dir"]) / page.file.src_path
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ai", "--", str(src)],
            capture_output=True, text=True,
            cwd=str(Path(config["docs_dir"]).parent.parent),
        )
        raw = result.stdout.strip()
        if result.returncode == 0 and raw:
            dt = datetime.fromisoformat(raw)
            context["git_date"] = dt.strftime("%B %-d, %Y")
        else:
            context["git_date"] = None
    except Exception:
        context["git_date"] = None
    return context
