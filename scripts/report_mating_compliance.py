"""Pool the mating-compliance runs into one table, and one specification number.

Carrying and mating want opposite things from the same mechanism: rigid, the
module cannot be aligned by the rack, and released, it is not held at all. The
lock therefore has a middle state, and this file is what says how stiff it has
to be — a number the interface specification quotes and nothing else in this
repository can supply.

Reads workflow reports written by ``scripts/run_workflow_demo.py``. CPU only.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _row(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    interface = report.get("capture_interface") or {}
    transit = report.get("robot_carried_transit") or {}
    drift = transit.get("max_tool_to_module_position_drift_m") or {}
    chain = report.get("chain") or {}
    return {
        "source": path.name,
        "mating_mode": interface.get("mating_mode"),
        "stiffness_n_per_m": interface.get("mating_compliance_n_per_m"),
        "stiffness_nm_per_rad": interface.get("mating_compliance_nm_per_rad"),
        "stroke_m": interface.get("mating_stroke_m"),
        "force_cap_n": interface.get("mating_force_cap_n"),
        "destination_channel_relief_m": interface.get("destination_channel_relief_m"),
        "reached_phase": report.get("reached_phase") or (chain.get("furthest_phase_reached")),
        "seated": report.get("seated_conditions_still_held_after_settling"),
        "predicate_fired": report.get("predicate_fired"),
        "success_rate": chain.get("success_rate"),
        "max_tool_to_module_drift_m": drift.get("max"),
        "terminal_module_x_m": (
            (transit.get("observed_per_environment") or [{}])[0].get("module_travel_m")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    rows = [_row(path) for path in arguments.reports if path.is_file()]
    rows.sort(key=lambda row: (row["mating_mode"] or "", row["stiffness_n_per_m"] or 0.0))
    print(
        f"{'mode':<10} {'K N/m':>9} {'cap N':>8} {'relief mm':>10} "
        f"{'reached':<10} {'seated':<7} {'drift mm':>9}"
    )
    for row in rows:
        drift = row["max_tool_to_module_drift_m"]
        print(
            f"{str(row['mating_mode']):<10} {row['stiffness_n_per_m'] or 0:>9.0f} "
            f"{row['force_cap_n'] or 0:>8.0f} "
            f"{(row['destination_channel_relief_m'] or 0.0) * 1000.0:>10.2f} "
            f"{str(row['reached_phase']):<10} {str(row['seated']):<7} "
            f"{(drift * 1000.0) if drift is not None else float('nan'):>9.1f}"
        )

    if arguments.output is not None:
        seated = [row for row in rows if row["seated"]]
        result = {
            "title": "Mating compliance for a robot-carried module",
            "generated_utc": datetime.now(UTC).isoformat(),
            "evidence_type": "simulation_only",
            "status": "passed" if seated else "failed",
            "question": (
                "How compliant does the robot-side form lock have to become for the rack's own "
                "lead-in to seat a module the robot carried to it?"
            ),
            "rows": rows,
            "scope_and_limitations": [
                "Simulation only. No result here was produced on real hardware.",
                "One environment per row: this is a mechanism sweep, not a reliability rate.",
                "The compliance is a bounded spring-damper between wrist_3_link and the module "
                "with a finite stroke; its hardware is authored on the wrist without colliders, "
                "so the jaws' contact with the pin is not simulated.",
            ],
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
