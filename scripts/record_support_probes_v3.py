"""Preserve the exploratory probes that decided the v3 registered support.

These ran before the study contract was frozen, to answer three questions the
contract then depended on: does mount compliance defeat the competent first
attempt, which layouts settle at all, and does the half-length cable model
survive a finer integration step. Their run directories are scratch and are not
retained, so the case definition and the outcome of every request are written
here instead, and the record is what the README and roadmap quote.

Nothing here is a study result. The probes are exploratory by construction and
none of them entered the registered block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.cable_study_v3 import outcome_label  # noqa: E402

PROBES = {
    "mount_compliance_does_not_defeat_competence": {
        "question": "Does a finite-stiffness port mount leave a residual failure the competent first "
                    "attempt cannot prevent?",
        "design": "force_guided_insertion, installed loop 4 mm, lateral stiffness "
                  "{rigid, 8000, 4000, 2000, 1000} N/m crossed with lateral mounting offsets "
                  "{0, 2, 4} mm. Axial stiffness is 6.25x lateral throughout.",
        "directory": "probe1",
    },
    "layout_settling_screen": {
        "question": "Which layout variations reach an accepted settled configuration at all?",
        "design": "Three screens. First, run direction rotated with the cable exit rotated to match. "
                  "Second, run direction rotated with the cable exit held at the registered free "
                  "direction. Third, clip position along the shelf crossed with a lateral dogleg in "
                  "the route, at the two extreme installed loops.",
        "directory": ["pilot_layout", "pilot_layout2", "pilot_layout3"],
    },
    "parametric_action_shakedown": {
        "question": "Does the continuous clearance action produce a usable spread of outcomes?",
        "design": "Two decision times crossed with retreat {0, 40, 80, 100} mm and four "
                  "bearing/excursion combinations, at the registered clearance speed.",
        "directory": "probe3",
    },
}


def collect(run_dir: Path) -> list[dict]:
    rows = []
    for directory in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        path = directory / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        case = result["case"]
        rows.append({
            "case": case["id"], "outcome": outcome_label(result),
            "failure_reason": result["job"]["failure_reason"],
            "settled": result["settled"],
            "clip_retained": result["terminal_clip_retained"],
            "installed_loop_m": case.get("installed_loop_m"),
            "run_direction_xy": case.get("run_direction_xy"),
            "outward_xy": case.get("outward_xy"),
            "fixture_offset_m": case.get("fixture_offset_m"),
            "route_waypoints_along_across_m": case.get("route_waypoints_along_across_m"),
            "fixture_overrides": case.get("fixture_overrides"),
            "cable_overrides": case.get("cable_overrides"),
            "clocks_override": case.get("clocks_override"),
            "port_compliance": case.get("port_compliance"),
            "repair_action": case.get("repair_action"),
            "forced_macro_at_s": case.get("forced_macro_at_s"),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True,
                        help="Directory holding the scratch probe run directories.")
    parser.add_argument("--out", type=Path, default=Path("evidence/cable_support_probes_v3.json"))
    args = parser.parse_args()

    probes = {}
    for name, spec in PROBES.items():
        directories = spec["directory"] if isinstance(spec["directory"], list) else [spec["directory"]]
        rows = [row for d in directories for row in collect(args.probe_root / d)]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["outcome"]] = counts.get(row["outcome"], 0)+1
        probes[name] = {"question": spec["question"], "design": spec["design"],
                        "requests": len(rows), "outcome_counts": counts,
                        "settled": sum(1 for r in rows if r["settled"]), "requests_detail": rows}

    report = {
        "schema": 1, "id": "cable_support_probes_v3", "created_on": "2026-09-11",
        "status": "exploratory_probes_preserved_none_entered_the_registered_block",
        "scope": "Probes run before the safe-repair boundary contract was frozen, to choose its "
                 "registered support. Not a study result, not a comparison and not a hardware claim. "
                 "Scratch run directories were not retained; the case definitions here are sufficient "
                 "to rebuild every request.",
        "probes": probes,
        "reading": {
            "mount_compliance": "Every request completed with the required clip retained across an "
                                "eightfold range of lateral mount stiffness crossed with mounting "
                                "offsets. Compliance is not a fault on this observation interface, "
                                "because every arm is handed the port's live pose at the servo rate "
                                "and therefore tracks a target that moves under contact.",
            "layouts": "Rotating the run direction together with the cable exit is fragile. Holding "
                       "the exit at the registered free direction and varying the clip position and "
                       "route is more robust. The five layouts the contract registers are the "
                       "surviving families.",
            "parametric_action": "The clearance action spans the boundary: outcomes include seating "
                                 "with the clip retained, clip release and load abort within the same "
                                 "grid, so the registered study is not measuring a degenerate space.",
        },
        "limitations": [
            "Exploratory: no pre-registration, no held-out split, no repeated seeds.",
            "Settling rejections here are properties of a construction, not of a controller.",
            "These counts are not comparable with the registered block's denominator.",
        ],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False, default=float), encoding="utf-8")
    print(json.dumps({name: {"requests": p["requests"], "outcomes": p["outcome_counts"]}
                      for name, p in probes.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
