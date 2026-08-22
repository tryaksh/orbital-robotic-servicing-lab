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
* this -- does the file a document cites exist at all?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Documents a stranger or the next implementation agent is expected to read.
DOCUMENTS = (
    "README.md",
    "CLAUDE.md",
    "docs/claude_opus_5_handoff.md",
    "docs/compute_service_demo.md",
    "docs/service_interface_spec.md",
    "evidence/RETRACTED.md",
)
#: A bare report name with no directory, e.g. ``grapple_extract_v14reset_certification.json``.
BARE = re.compile(r"`([A-Za-z0-9_\-]+\.json)`")
#: A path-qualified reference, e.g. ``evidence/module_pose_head.json``.
QUALIFIED = re.compile(r"evidence/([A-Za-z0-9_\-]+\.json)")


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

    for document, names in sorted(missing.items()):
        for candidate in sorted(names):
            print(f"MISSING  {document} cites evidence/{candidate}, which does not exist")

    total = sum(len(names) for names in missing.values())
    print(f"  {total} broken evidence references")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
