"""Every relative link in a maintained document must point at something that exists.

``scripts/check_evidence_links.py`` already asks whether a document cites a
report that is missing or retracted. It does not ask the blunter question: does
this markdown link go anywhere?

That gap is not hypothetical here. ``docs/archive/`` became ``docs/handover/`` on
2026-09-13 and four files referenced the old path; nothing would have noticed.
A document nobody can navigate is worse than a short one, and a dead link is the
cheapest possible thing to catch.

Maintained documents only. ``docs/handover/`` is deliberately excluded: a handover
describes the repository as it was on the day it was written and is not updated
afterwards, so a link that has since gone stale there is history rather than a
defect.

Source-level and CPU-only.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: ``[text](target)``, capturing the target.
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

#: Links we cannot check by looking at the filesystem.
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def maintained_documents() -> list[pathlib.Path]:
    documents = [ROOT / name for name in ("README.md", "ROADMAP.md", "AGENTS.md", "CLAUDE.md")]
    documents += sorted(p for p in (ROOT / "docs").glob("*.md"))
    documents += [ROOT / "evidence" / "RETRACTED.md", ROOT / "scripts" / "README.md"]
    return [p for p in documents if p.is_file()]


DOCUMENTS = maintained_documents()


def test_there_are_documents_to_check() -> None:
    """Guard against the walk passing because it found nothing."""
    assert len(DOCUMENTS) > 8, f"only {len(DOCUMENTS)} documents found; the walk is wrong"


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_resolve(path: pathlib.Path) -> None:
    broken: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for target in LINK.findall(line):
            if target.startswith(SKIP_PREFIXES):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            if not (resolved.is_file() or resolved.is_dir()):
                broken.append(f"  {path.relative_to(ROOT).as_posix()}:{line_no} -> {target}")
    assert not broken, "these links do not point at anything:\n" + "\n".join(broken)
