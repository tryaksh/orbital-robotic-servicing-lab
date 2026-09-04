#!/usr/bin/env python3
"""Every evidence file the documentation names must exist.

The discipline this repository claims is that no number appears anywhere without
naming the file it came from. That claim is only worth something if the names
resolve: a reader who clicks a filename and finds nothing has been told the
evidence exists when it does not, which is worse than not citing it.

Text only. No simulator, no checkpoints, so it runs in CI on every commit
alongside the two currency checks:

* ``check_evidence_currency.py`` -- does a report describe the checkpoint a run
  actually loaded?
* ``check_criterion_currency.py`` -- has the criterion moved since the report?
* this -- does the file a document cites exist at all, and is it still live?

The second half was added after an audit found nine citations of retracted
reports sitting in the working documents with nothing on the line to say so. A
reader -- or a second agent drafting a manuscript from these documents, which is
exactly what is about to happen -- has no way to tell a live citation from a
withdrawn one. Naming a retracted report is allowed, and sometimes necessary, but
only where the surrounding text says it is retracted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Documents a stranger or the next implementation agent is expected to read.
DOCUMENTS = (
    "README.md",
    "CLAUDE.md",
    "docs/NOW.md",
    "docs/NEXT_WORK.md",
    "docs/compute_service_demo.md",
    "docs/service_interface_spec.md",
    "evidence/RETRACTED.md",
)
#: A bare report name with no directory, e.g. ``grapple_extract_v14reset_certification.json``.
BARE = re.compile(r"`([A-Za-z0-9_\-]+\.json)`")
#: A path-qualified reference, e.g. ``evidence/module_pose_head.json``.
QUALIFIED = re.compile(r"evidence/([A-Za-z0-9_\-]+\.json)")
#: Words that make a retracted citation legitimate: the text is disclosing the
#: withdrawal rather than resting on the number.
DISCLOSED = ("retract", "superseded", "withdrawn", "must not be quoted", "out of agreement")
#: How far from the citation the disclosure may sit. A citation and its caveat
#: belong in the same paragraph; three lines is a generous reading of that.
WINDOW = 3


def retracted_from_manifest() -> set[str]:
    """The withdrawn set, taken from the index this repository calls authoritative."""

    path = ROOT / "evidence" / "MANIFEST.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("retracted", {}))


def undisclosed_retractions(text: str, retracted: set[str]) -> list[tuple[int, str]]:
    """Lines citing a retracted report with no disclosure within ``WINDOW`` lines."""

    lines = text.splitlines()
    found: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        for name in retracted:
            if name not in line:
                continue
            lo = max(0, index - WINDOW)
            context = " ".join(lines[lo : index + WINDOW + 1]).lower()
            if not any(word in context for word in DISCLOSED):
                found.append((index + 1, name))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="Print only the failures.")
    args = parser.parse_args()

    available = {path.name for path in (ROOT / "evidence").glob("*.json")}
    missing: dict[str, set[str]] = {}
    cited: set[str] = set()

    for name in DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            missing.setdefault(name, set()).add("<the document itself is missing>")
            continue
        text = path.read_text(encoding="utf-8")
        names = set(QUALIFIED.findall(text)) | set(BARE.findall(text))
        for candidate in names:
            # A bare .json in prose may be an artifact rather than evidence; only
            # hold a name to account if evidence/ is where it claims to live or
            # if a file of that name is meant to be there.
            if candidate in available:
                cited.add(candidate)
            elif f"evidence/{candidate}" in text or candidate.endswith("_certification.json"):
                missing.setdefault(name, set()).add(candidate)

    if not args.quiet:
        print(f"  {len(available)} files in evidence/, {len(cited)} of them cited by the documents")
        uncited = sorted(available - cited)
        print(f"  {len(uncited)} never cited (not an error; sweeps and superseded runs live here too)")

    retracted = retracted_from_manifest()
    undisclosed = 0
    for name in DOCUMENTS:
        path = ROOT / name
        if not path.exists() or name == "evidence/RETRACTED.md":
            continue
        for line_no, report in undisclosed_retractions(path.read_text(encoding="utf-8"), retracted):
            print(f"RETRACTED  {name}:{line_no} cites {report} without saying it is retracted")
            undisclosed += 1

    for document, names in sorted(missing.items()):
        for candidate in sorted(names):
            print(f"MISSING  {document} cites evidence/{candidate}, which does not exist")

    total = sum(len(names) for names in missing.values())
    print(f"  {total} broken evidence references")
    print(f"  {undisclosed} undisclosed citations of retracted evidence")
    return 1 if (total or undisclosed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
