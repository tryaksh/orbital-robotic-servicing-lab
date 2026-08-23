"""Pool the robot-carried transit runs into one interface result.

The question this file answers is the one the branch exists for: **can the
six-axis robot carry the module between bays, and what does it take.** It is a
comparison, not a single number, because the answer is only meaningful against
its control:

* the *passive* arm is the same workflow with nothing but the finger pads
  holding the module. It is expected to fail, and its failure is the
  measurement that makes the mechanism below a necessity rather than a
  convenience;
* the *latched* arm adds the robot-side form lock;
* the *rating sweep*, if given, says what the lock has to be rated at, which is
  the specification deliverable.

Reads workflow reports written by ``scripts/run_workflow_demo.py``. Runs on the
CPU and imports nothing from Isaac Lab, so it can run while the GPU is busy.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _arm(report: dict, label: str) -> dict:
    transit = report.get("robot_carried_transit") or {}
    interface = report.get("capture_interface") or {}
    chain = report.get("chain") or {}
    observed = transit.get("observed_per_environment") or []
    entered = [row for row in observed if row.get("entered_transit")]
    latch_rows = interface.get("observed_per_environment") or []
    return {
        "arm": label,
        "capture_interface": interface.get("type"),
        "latch_rated_force_n": interface.get("rated_force_n"),
        "latch_rated_torque_nm": interface.get("rated_torque_nm"),
        "latch_states": interface.get("states"),
        "mating_compliance_n_per_m": interface.get("mating_compliance_n_per_m"),
        "destination_channel_relief_m": interface.get("destination_channel_relief_m"),
        "carrier": transit.get("carrier"),
        "episodes": chain.get("episodes"),
        "successes": chain.get("successes"),
        "success_rate": chain.get("success_rate"),
        "furthest_phase_reached": chain.get("furthest_phase_reached"),
        "environments_that_entered_transit": len(entered),
        "environments_retaining_the_planned_transform_throughout": transit.get(
            "environments_retaining_the_planned_transform_throughout"
        ),
        "latch_engagements": sum(1 for row in latch_rows if row.get("ever_engaged")),
        "latch_softened_for_mating": sum(1 for row in latch_rows if row.get("softened_for_mating")),
        "latch_released_after_seating": sum(1 for row in latch_rows if row.get("released_after_seating")),
        "hands_opened_after_settling_verification": sum(
            1 for row in latch_rows if row.get("hand_opened_after_settling_verification")
        ),
        "engagements_refused_out_of_seek_travel": sum(
            int(row.get("engagements_refused_out_of_seek_travel") or 0) for row in latch_rows
        ),
        "max_tool_to_module_position_drift_m": transit.get("max_tool_to_module_position_drift_m"),
        "max_tool_to_module_orientation_drift_rad": transit.get(
            "max_tool_to_module_orientation_drift_rad"
        ),
        "max_grip_error_m": transit.get("max_grip_error_m"),
        "tool_travel_m": transit.get("tool_travel_m"),
        "module_travel_m": transit.get("module_travel_m"),
        "control_steps_before_retention_was_lost": transit.get(
            "control_steps_before_retention_was_lost"
        ),
        "measured_front_overhang_m": transit.get("measured_front_overhang_m"),
        "retreat_clear_blade_centre_x_m": transit.get("retreat_clear_blade_centre_x_m"),
        "terminal_tool_to_handle_error_m": (chain.get("terminal_metrics") or {}).get(
            "tool_to_handle_error_m"
        ),
        "policy_set_sha256": report.get("policy_set_sha256"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passive", type=Path, required=True)
    parser.add_argument("--latched", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, nargs="*", default=())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Robot-carried bay-to-bay transit, with and without a form lock")
    arguments = parser.parse_args()

    passive = json.loads(arguments.passive.read_text(encoding="utf-8"))
    latched = json.loads(arguments.latched.read_text(encoding="utf-8"))
    arms = [_arm(passive, "passive_parallel_jaw_only"), _arm(latched, "robot_side_form_lock")]
    sweep = []
    for path in arguments.sweep:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        row = _arm(report, "robot_side_form_lock")
        row["source"] = Path(path).name
        sweep.append(row)
    sweep.sort(key=lambda row: (row["latch_rated_force_n"] or 0.0))

    retained = arms[1]["environments_retaining_the_planned_transform_throughout"]
    result = {
        "title": arguments.title,
        "generated_utc": datetime.now(UTC).isoformat(),
        "evidence_type": "simulation_only",
        "status": "passed" if (retained or 0) > 0 else "failed",
        "policy": {
            "checkpoint_sha256": latched.get("policy_set_sha256"),
            "per_phase_checkpoint_sha256": latched.get("checkpoint_sha256"),
            "grasp_model": "physical_finger_pad_contact_plus_robot_side_form_lock",
        },
        "question": (
            "Can the six-axis robot itself carry the compute module from bay 0 to bay 1, with a "
            "bounded tool-to-module transform, and what interface does that require?"
        ),
        "protocol": {
            "task": latched.get("task"),
            "workflow": latched.get("workflow"),
            "seed": latched.get("seed"),
            "gravity": "zero",
            "policy_rate_hz": 30,
            "physics_rate_hz": 120,
            "learned_phases": latched.get("learned_phases"),
            "scripted_phases": latched.get("scripted_phases"),
            "retention_limit_position_m": (latched.get("robot_carried_transit") or {}).get(
                "retention_limit_position_m"
            ),
            "retention_limit_orientation_rad": (latched.get("robot_carried_transit") or {}).get(
                "retention_limit_orientation_rad"
            ),
            "retention_limit_derivation": (latched.get("robot_carried_transit") or {}).get(
                "retention_limit_derivation"
            ),
            "reference_frame": (latched.get("robot_carried_transit") or {}).get("reference_frame"),
            "no_world_payload_stage": (
                (passive.get("transit_planner") or {}).get("physical_payload_stage_handoff") is False
                and (latched.get("transit_planner") or {}).get("physical_payload_stage_handoff") is False
            ),
            "no_robot_or_payload_state_writes": (
                (latched.get("transit_planner") or {}).get("robot_or_payload_state_writes") is False
            ),
        },
        "arms": arms,
        "rating_sweep": sweep,
        "scope_and_limitations": [
            "Simulation only. No result here was produced on real hardware.",
            "The form lock's load path is a break-rated PhysX fixed joint between wrist_3_link and "
            "the module, engaged only after learned extraction clears the rails and released before "
            "insertion. Its visible hardware is authored on the wrist and its clearances are derived "
            "in evidence/service_latch_clearance.json; the hardware carries no collider, so contact "
            "between the jaws and the pin is not simulated.",
            "The passive arm is the control and is expected to fail. It is published as a measured "
            "interface limitation, not as a failed policy.",
            "Wrist reaction to the lock is carried by the joint; the arm's joint torques under that "
            "reaction are not separately measured.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    for row in arms:
        drift = row["max_tool_to_module_position_drift_m"] or {}
        print(
            f"  {row['arm']:<28} retained "
            f"{row['environments_retaining_the_planned_transform_throughout']}/"
            f"{row['environments_that_entered_transit']}  "
            f"drift p50 {drift.get('p50', float('nan')) * 1000.0:8.2f} mm  "
            f"success {row['success_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
