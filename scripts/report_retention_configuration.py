#!/usr/bin/env python3
"""Which cohorts were run with the destination rack's pawls fitted, and which were not?

Every workflow report carries a `destination_rack_retention` block saying whether
the destination bay had its retention mechanism at all. That block has been in
the reports the whole time and nothing has ever read it across cohorts, so a
scoping fact large enough to change several published conclusions has sat in
plain sight:

**The boundary sweep was run with retention disabled.**
`artifacts/robustness192_section` and the traced re-run of it both record
`enabled: false, mechanism: "none"`. Every boundary verdict that rests on that
sweep -- the module-section points, the clearance axis, "not qualified" -- was
measured on a bay with no pawls, while the success definition re-checks the
module after a 0.70 s **free-module** window. In zero gravity a released module
with residual velocity coasts, and with no pawls nothing brings it back.

Cohorts that do fit the pawls report `max_rack_to_module_position_drift_m = 0.0`
on every environment where they engaged. The pawls do not rescue a bad
insertion -- they engage only after the unchanged insertion predicate fires --
but where they engage, drift is zero.

This script reads the block from every report it is given and groups them, so
the scope of any claim can be checked in one command rather than inferred.

    python scripts/report_retention_configuration.py --report evidence/retention_configuration_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402


def describe(path: Path) -> dict | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    block = document.get("destination_rack_retention")
    if not isinstance(block, dict):
        return None
    observed = block.get("observed_per_environment") or []
    engaged = [row for row in observed if row.get("engaged_after_measured_seating")]
    drifts = [
        row.get("max_rack_to_module_position_drift_m")
        for row in engaged
        if row.get("max_rack_to_module_position_drift_m") is not None
    ]
    return {
        "report": path.as_posix(),
        "enabled": bool(block.get("enabled")),
        "mechanism": block.get("mechanism"),
        "rated_force_n": block.get("rated_force_n"),
        "environments": len(observed),
        "environments_engaged": len(engaged),
        "max_drift_while_rack_only_m": max(drifts) if drifts else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--glob",
        default="artifacts/**/*_report.json",
        help="which reports to read, relative to the repository root",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = [
        entry
        for entry in (describe(path) for path in sorted(ROOT.glob(args.glob)))
        if entry is not None
    ]
    if not rows:
        print("no report carries a destination_rack_retention block")
        return 1

    enabled = [r for r in rows if r["enabled"]]
    disabled = [r for r in rows if not r["enabled"]]
    print(f"{len(rows)} reports carry the block: {len(enabled)} with pawls, {len(disabled)} without\n")

    engaged_total = sum(r["environments_engaged"] for r in enabled)
    environments_total = sum(r["environments"] for r in enabled)
    drifts = [
        r["max_drift_while_rack_only_m"]
        for r in enabled
        if r["max_drift_while_rack_only_m"] is not None
    ]
    print("With pawls fitted:")
    print(f"  environments where they engaged: {engaged_total}/{environments_total}")
    if drifts:
        print(f"  worst module drift while the rack alone holds: {max(drifts) * 1000:.6f} mm")
    print()
    print("Without pawls, the module is released free in zero gravity. Reports:")
    for entry in disabled[:12]:
        print(f"  {entry['report']}")
    if len(disabled) > 12:
        print(f"  ... and {len(disabled) - 12} more")

    document = {
        "title": "Which cohorts had the destination rack's retention fitted",
        "evidence_type": "second_reading_of_existing_reports",
        "generated_utc": datetime.now(UTC).isoformat(),
        "question": (
            "Every workflow report records whether the destination bay had its "
            "retention mechanism. Which published cohorts were run without it?"
        ),
        "source_revision": git_source_revision(ROOT),
        "reports_with_pawls": len(enabled),
        "reports_without_pawls": len(disabled),
        "environments_engaged_where_fitted": engaged_total,
        "environments_where_fitted": environments_total,
        "worst_drift_while_rack_only_m": max(drifts) if drifts else None,
        "mechanisms_seen": dict(Counter(str(r["mechanism"]) for r in rows)),
        "per_report": rows,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "This is a second reading of existing reports, not a new measurement.",
            "A cohort run without pawls is not thereby wrong. It measured what it "
            "measured; the point is that its scope is narrower than the prose citing "
            "it has assumed, because the destination bay had no retention at all.",
            "The pawls engage only after the unchanged insertion predicate fires, so "
            "they preserve a good insertion and do not rescue a bad one. A zero drift "
            "figure is conditional on engagement, not a claim that every episode holds.",
            "The fixed-joint abstraction for the pawls is disclosed in every report "
            "and is not a validated hardware load capacity.",
        ],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
