"""Audit a recorded mission against the existing strict completion contract.

This checks recorded measurements, not physical truth or statistical reliability.
Missing, malformed, or contradictory measurements fail closed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .models import MissionVerification

LIVE_TASK = "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
SEATED_CONDITIONS = (
    "axial_depth", "lateral_alignment", "orientation", "linear_velocity",
    "angular_velocity", "grasp_position", "grasp_orientation",
)
# The workflow rounds reported SI measurements to six decimals.
POSITION_LIMIT_M = 0.0025
ORIENTATION_LIMIT_RAD = 0.05236
RACK_HOLD_S = 0.70


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row(section: dict[str, Any]) -> dict[str, Any]:
    rows = section.get("observed_per_environment")
    if not isinstance(rows, list) or len(rows) != 1:
        return {}
    row = _object(rows[0])
    return row if type(row.get("env")) is int and row["env"] == 0 else {}


def _number(value: object, *, minimum: float = 0.0, maximum: float = math.inf) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum)


def verify_mission(report: object, *, video_dir: Path | None = None) -> MissionVerification:
    """Audit one camera-driven relocation; optionally require a saved MP4."""
    report = _object(report)
    transit = _object(report.get("robot_carried_transit"))
    carried = _row(transit)
    rack = _object(report.get("destination_rack_retention"))
    held = _row(rack)
    released = _row(_object(report.get("capture_interface")))
    perception = _object(report.get("perception"))
    availability = _object(perception.get("detector_availability"))
    attempts = availability.get("attempts")
    detections = availability.get("detections")
    failures = availability.get("failures")
    conditions = _object(report.get("insertion_conditions"))
    planning = _object(report.get("planning"))
    checks = {
        "single_camera_episode": (
            report.get("task") == LIVE_TASK and report.get("workflow") == "relocate"
            and type(report.get("num_envs")) is int and report["num_envs"] == 1
        ),
        "terminal_seating": (
            report.get("completed") is True and report.get("predicate_fired") is True
            and report.get("reached_phase") == "done"
            and report.get("seated_conditions_still_held_after_settling") is True
            and all(conditions.get(key) is True for key in SEATED_CONDITIONS)
        ),
        "both_robot_supports_released": (
            report.get("all_conditions_including_released_gripper") is True
            and released.get("released_after_seating") is True
            and released.get("hand_opened_after_settling_verification") is True
        ),
        "rack_only_hold": (
            rack.get("enabled") is True and rack.get("world_constraint") is False
            and rack.get("module_pose_write") is False
            and rack.get("joint_body0") == "Rack" and rack.get("joint_body1") == "SpareBlade"
            and held.get("engaged_after_measured_seating") is True
            and held.get("full_rack_only_recheck_observed") is True
            and _number(held.get("rack_only_interval_s"), minimum=RACK_HOLD_S)
            and _number(held.get("max_rack_to_module_position_drift_m"), maximum=POSITION_LIMIT_M)
            and _number(held.get("max_rack_to_module_orientation_drift_rad"), maximum=ORIENTATION_LIMIT_RAD)
        ),
        "bounded_robot_carried_transit": (
            transit.get("carrier") == "six_axis_robot"
            and carried.get("entered_transit") is True and carried.get("retained_throughout") is True
            and _number(carried.get("samples"), minimum=1)
            and _number(carried.get("tool_travel_m"), minimum=0.001)
            and _number(carried.get("module_travel_m"), minimum=0.001)
            and _number(carried.get("max_position_drift_m"), maximum=POSITION_LIMIT_M)
            and _number(carried.get("max_orientation_drift_rad"), maximum=ORIENTATION_LIMIT_RAD)
        ),
        "live_rgbd": (
            perception.get("source") == "rgb_fiducial_calibrated_pnp"
            and all(type(value) is int for value in (attempts, detections, failures))
            and _number(detections, minimum=1) and _number(failures)
            and attempts == detections + failures
            and perception.get("terminal_bay_occupancy_scores") == [0.0, 1.0]
        ),
        "visual_occupancy_plan": planning.get("source_occupied_destination_clear") is True,
        "executed_controllers": (
            report.get("learned_phases") == ["capture", "extract"]
            and isinstance(report.get("scripted_phases"), list)
            and {"transit", "guarded_insert"}.issubset(report["scripted_phases"])
            and _number(released.get("guarded_advance_steps"), minimum=1)
        ),
    }
    if video_dir is not None:
        # Full decoding remains a media review step. Hashes detect later changes.
        valid = False
        for path in sorted(video_dir.glob("*.mp4")):
            if path.is_file() and not path.is_symlink() and path.stat().st_size > 24:
                with path.open("rb") as stream:
                    valid |= stream.read(12)[4:8] == b"ftyp"
        checks["saved_video"] = valid
    return MissionVerification(
        passed=all(checks.values()), checks=checks,
        failed_checks=[name for name, passed in checks.items() if not passed],
        transit_position_drift_mm=(carried["max_position_drift_m"] * 1000
                                   if _number(carried.get("max_position_drift_m")) else None),
        rack_only_hold_s=(held["rack_only_interval_s"] if _number(held.get("rack_only_interval_s")) else None),
        detections=detections if type(detections) is int and detections >= 0 else None,
        detection_attempts=attempts if type(attempts) is int and attempts >= 0 else None,
    )
