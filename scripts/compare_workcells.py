"""Print the before/after table straight from the evidence files.

The PR and the README both carry a table comparing `main`'s workcell at
`ROBOT_ROOT_POS` x = -0.45 against this branch's at
`GRAPPLE_ROBOT_ROOT_POS` x = -0.65. A table typed by hand drifts from the files
it cites -- this repository has recorded that failure twice -- so it is generated
instead, and a reviewer can regenerate it.

Two conventions it does not paper over:

* the **insert** gate is the worse bay, not the pool, so where the two differ
  both are shown;
* the **vision** arms are not gated at 95%; their gate is the camera arm within
  10 points of the oracle and clearly above blind, which is a comparison between
  rows rather than a threshold on one. `passed` is therefore not printed for
  them, and `--gates` states the rule instead of pretending a number implies it.

CPU only. Reads no checkpoints and imports nothing from Isaac Lab.

Usage::

    python scripts/compare_workcells.py            # the before/after table
    python scripts/compare_workcells.py --gates    # promotion gate per report
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "evidence"

#: label, report on main, report on this branch. ``None`` where a side has no
#: counterpart -- the relocation is the whole point of the branch and `main` has
#: never produced one.
PAIRS = (
    ("Capture, alone", "grapple_grasp_v5_certification.json", "grapple_grasp_v6w65_certification.json"),
    ("Extract", "grapple_extract_v14reset_certification.json", "grapple_extract_v16w65_certification.json"),
    (
        "Insert, both bays",
        "grapple_insert_two_slot_certification.json",
        "grapple_insert_two_slot_w65_certification.json",
    ),
    ("Removal chain", "workflow_remove_retain_certification.json", "workflow_remove_w65_certification.json"),
    (
        "Installation chain",
        "workflow_install_clock30retain_certification.json",
        "workflow_install_w65_certification.json",
    ),
    ("**Relocation chain**", None, "workflow_relocate_certification.json"),
    (
        "Vision, oracle",
        "vision_workflow_oracle_twoslot_certification.json",
        "vision_workflow_oracle_twoslot_w65_certification.json",
    ),
    (
        "Vision, camera",
        "vision_workflow_camera_twoslot_certification.json",
        "vision_workflow_camera_twoslot_w65_certification.json",
    ),
    (
        "Vision, blind",
        "vision_workflow_blind_twoslot_certification.json",
        "vision_workflow_blind_twoslot_w65_certification.json",
    ),
)


def read(name: str | None) -> dict | None | str:
    if name is None:
        return None
    path = ROOT / name
    if not path.is_file():
        return "missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    overall, gate = data.get("overall", {}), data.get("gate", {})
    interval = overall.get("success_rate_wilson_95") or {}
    return {
        "rate": overall.get("success_rate"),
        "episodes": overall.get("episodes"),
        "low": interval.get("low"),
        "high": interval.get("high"),
        "worst": gate.get("worst_stage_success_rate"),
        "passed": gate.get("passed"),
        "instability": overall.get("instability_terminations"),
        "non_finite": overall.get("non_finite_metric_episodes"),
    }


def cell(block, show_gate: bool) -> str:
    if block is None:
        return "not attempted"
    if block == "missing":
        return "not yet run"
    if block["rate"] is None:
        return "?"
    text = f"{block['rate'] * 100:.2f}%"
    if block["worst"] is not None and abs(block["worst"] - block["rate"]) > 5e-5:
        text += f" (worse stage {block['worst'] * 100:.2f}%)"
    if block["low"] is not None:
        text += f" [{block['low'] * 100:.2f}, {block['high'] * 100:.2f}]"
    if block["episodes"]:
        text += f", n={block['episodes']}"
    if block["instability"] or block["non_finite"]:
        text += f" — **{block['instability']} unstable, {block['non_finite']} non-finite**"
    if show_gate:
        text += "  " + ("PASS" if block["passed"] else "FAIL")
    return text


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", action="store_true", help="Show each report's promotion gate verdict.")
    args = parser.parse_args()

    print("| | `main`, base x = −0.45 | this branch, base x = −0.65 |")
    print("| --- | ---: | ---: |")
    for label, before, after in PAIRS:
        vision = label.startswith("Vision")
        print(f"| {label} | {cell(read(before), args.gates and not vision)} | {cell(read(after), args.gates and not vision)} |")

    if args.gates:
        print()
        print("Promotion gate: at least the stated rate pooled AND in every stage, zero")
        print("instability terminations, zero non-finite terminal metrics.")
        print()
        print("The vision rows carry no PASS/FAIL because 95% is not their gate. Theirs is a")
        print("comparison between rows: the camera arm within 10 points of the oracle, and")
        print("clearly above blind. Read those three together or not at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
