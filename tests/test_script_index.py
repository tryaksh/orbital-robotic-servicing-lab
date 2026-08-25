"""Every script must say what it is, and the index must stay current.

``scripts/`` holds seventy-six files with no structure beyond a naming
convention. The cost of that is paid by whoever arrives next and has to open
several to find the one they want, which is the kind of avoidable reading this
repository is trying to stop paying for.

``scripts/README.md`` is generated from each script's own first documentation
line, so it cannot describe a script differently from how the script describes
itself. These hold the two properties that make it worth trusting: it is not
stale, and no script is missing from it.

Source-level and CPU-only. No simulator, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_script_index as index_builder  # noqa: E402

INDEX = ROOT / "scripts" / "README.md"


def test_the_script_index_is_current() -> None:
    assert INDEX.read_text(encoding="utf-8") == index_builder.build(), (
        "scripts/README.md is stale. Run: python scripts/build_script_index.py"
    )


def test_every_script_is_listed() -> None:
    listed = INDEX.read_text(encoding="utf-8")
    for path in sorted((ROOT / "scripts").iterdir()):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".ps1"}:
            continue
        assert f"`{path.name}`" in listed, f"{path.name} is not in scripts/README.md"


def test_every_script_documents_itself() -> None:
    """A dash in the index means a script that never said what it was for.

    Two of these were train.py and play.py -- the entry points everything else
    wraps -- reading as blank because a byte-order mark made ``ast.parse`` raise
    before it reached their docstrings.
    """
    undocumented = [
        path.name
        for path in sorted((ROOT / "scripts").iterdir())
        if path.is_file()
        and path.suffix in {".py", ".sh", ".ps1"}
        and not index_builder.summary(path)
    ]
    assert not undocumented, (
        "these scripts have no first documentation line: " + ", ".join(undocumented)
    )
