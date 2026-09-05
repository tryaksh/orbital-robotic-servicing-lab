#!/usr/bin/env python3
"""Re-read every published paired comparison with a directional test.

`compare_paired_arms.mcnemar_exact` reported `one_sided_p` from the smaller
discordant tail, which carries no direction. This walks the reports that field
reached, adds the two directional values and the direction itself, and keeps
the original number under `one_sided_p_direction_blind` with a note, because a
published value is a record of what was published and deleting it would hide
the defect rather than fix it.

Only one report changes meaning: `rack_prescription_paired_n192`, where the
treatment lost 74 discordant episodes to 28 and the direction-blind value read
2.95e-06. Every other report ran the way its prose already said.

    python scripts/correct_paired_direction.py --check    # exit 1 if any report is stale
    python scripts/correct_paired_direction.py            # rewrite in place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_paired_arms import mcnemar_exact  # noqa: E402

NOTE = (
    "Computed from the smaller discordant tail, so it did not depend on which arm "
    "won. Kept as a record of what was published; read improvement_p or "
    "deterioration_p instead. See tests/test_paired_comparison_direction.py."
)


def _walk(node):
    """Yield every dict that looks like a paired-test block."""

    if isinstance(node, dict):
        if "gained" in node and "lost" in node and "two_sided_p" in node:
            yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def correct(path: Path) -> tuple[bool, list[str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    notes: list[str] = []
    for block in _walk(document):
        gained, lost = int(block["gained"]), int(block["lost"])
        fresh = mcnemar_exact(gained, lost)
        if "one_sided_p" in block:
            block["one_sided_p_direction_blind"] = block.pop("one_sided_p")
            block["one_sided_p_note"] = NOTE
            changed = True
        for key in ("improvement_p", "deterioration_p", "direction"):
            if block.get(key) != fresh[key]:
                block[key] = fresh[key]
                changed = True
        if fresh["direction"] == "lost":
            notes.append(
                f"{path.name}: treatment LOST {lost} to {gained}; "
                f"improvement_p = {fresh['improvement_p']:.6g}"
            )
    if changed:
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return changed, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report without writing")
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence")
    args = parser.parse_args()

    stale: list[Path] = []
    warnings: list[str] = []
    for path in sorted(args.evidence.glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "one_sided_p" not in text and "improvement_p" not in text:
            continue
        if args.check:
            document = json.loads(text)
            for block in _walk(document):
                fresh = mcnemar_exact(int(block["gained"]), int(block["lost"]))
                if "one_sided_p" in block or block.get("improvement_p") != fresh["improvement_p"]:
                    stale.append(path)
                    break
        else:
            changed, notes = correct(path)
            warnings.extend(notes)
            if changed:
                print(f"corrected {path.name}")

    for warning in warnings:
        print(f"  ! {warning}")
    if args.check and stale:
        print(f"{len(stale)} report(s) carry a direction-blind paired test:")
        for path in stale:
            print(f"  {path.name}")
        return 1
    if args.check:
        print("every paired report carries a directional test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
