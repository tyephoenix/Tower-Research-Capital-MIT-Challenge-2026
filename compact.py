"""Generate the submission notebook using rep2nb.

Notebook captures Tye's submission only — Deven's independent verification
scripts are excluded so the notebook stays a single, coherent pipeline.

Post-processes two things rep2nb gets wrong for this repo:
  1. Strips local module names from the auto-generated pip install cell.
  2. Rewrites chdir cells to use an absolute repo root anchor, so cells
     remain idempotent when re-run (the default `chdir('problem-1')`
     relative form drills into nested dirs if executed more than once).
"""
import json
import re
from pathlib import Path
from rep2nb import convert

OUTPUT = "limestone_data_challenge_2026.ipynb"
LOCAL_MODULES = {"tye", "deven", "candidates", "coefficients", "common", "matrix",
                 "trend", "em", "buy", "trade", "compute_sigma", "main"}

convert(
    repo_path=".",
    output=OUTPUT,
    exclude=["answers", "compact.py", "deven", ".pylibs"],
    include_pip_install=True,
)

path = Path(OUTPUT)
nb = json.loads(path.read_text())

ANCHOR_CELL = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": (
        "# Repo-root anchor: captured once, reused by every chdir below so\n"
        "# re-running section cells never drills into nested dirs.\n"
        "import os as _os\n"
        "_REPO_ROOT = _os.environ.setdefault('_LIMESTONE_REPO_ROOT', _os.getcwd())\n"
        "_os.chdir(_REPO_ROOT)\n"
        "del _os"
    ),
}

# Insert anchor cell right after any top-level pip install cell (else at top).
insert_at = 0
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] == "code" and "pip install" in src:
        insert_at = i + 1
        break
nb["cells"].insert(insert_at, ANCHOR_CELL)

CHDIR_INTO = re.compile(r"_os\.chdir\('([^']+)'\)")
CHDIR_UP = "_os.chdir('..')"
FILE_HEADER = re.compile(r"^# === (.+?) ===", re.MULTILINE)
FILE_LINE = re.compile(r"^__file__ = '([^']+)'", re.MULTILINE)

for cell in nb["cells"]:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if cell["cell_type"] != "code":
        continue
    new = src

    # Strip local module names from pip install lines
    if "pip install" in new:
        lines = new.split("\n")
        for i, line in enumerate(lines):
            if "pip install" not in line:
                continue
            head, _, tail = line.partition("pip install")
            parts = tail.split()
            kept = [p for p in parts if p.startswith("-") or p not in LOCAL_MODULES]
            lines[i] = f"{head}pip install {' '.join(kept)}"
        new = "\n".join(lines)

    # Rewrite section-chdir cells to be absolute + idempotent
    if "_os.chdir(" in new and "_REPO_ROOT" not in new:
        m = CHDIR_INTO.search(new)
        if m and m.group(1) != "..":
            target = m.group(1)
            new = (
                "import os as _os\n"
                f"_os.makedirs(_os.path.join(_REPO_ROOT, {target!r}), exist_ok=True)\n"
                f"_os.chdir(_os.path.join(_REPO_ROOT, {target!r}))\n"
                "del _os"
            )
        elif CHDIR_UP in new:
            new = new.replace(CHDIR_UP, "_os.chdir(_REPO_ROOT)")

    # Make __file__ absolute so Path(__file__).resolve().parent doesn't
    # depend on cwd — this protects HERE/REPO computation in main.py cells
    # against out-of-order or re-run execution.
    header_match = FILE_HEADER.search(new)
    file_match = FILE_LINE.search(new)
    if header_match and file_match:
        full_repo_path = header_match.group(1)  # e.g. 'problem-1/tye/candidates.py'
        parts = full_repo_path.split("/")
        import_os = "" if "import os as _os" in new else "import os as _os\n"
        abs_expr = ", ".join(repr(p) for p in parts)
        new = FILE_LINE.sub(
            f"{import_os}__file__ = _os.path.join(_REPO_ROOT, {abs_expr})",
            new, count=1,
        )

    cell["source"] = new

path.write_text(json.dumps(nb, indent=1))
print(f"  Post-processed: stripped local pip deps + idempotent chdirs ({path})")
