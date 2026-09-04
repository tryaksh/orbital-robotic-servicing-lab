"""Compare the derived serviceability boundary with preserved simulation arms.

This is deliberately a validator, not a curve fitter.  Its rules are frozen in
``PROTOCOL`` and it never changes a geometric tolerance.  A simulated loss is
called separated only when its Wilson upper bound is below the nominal Wilson
lower bound.  Every other result remains visible as support, mismatch, or a
missing comparison.

The command refuses non-canonical inputs, recomputes the CPU geometry model,
and refuses a dirty tracked worktree.  It therefore produces a report tied to
both the input bytes and the committed source that interpreted them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zero_g_blade_swap.provenance import git_source_revision  # noqa: E402

SCHEMA_VERSION = 2
REQUIRED_INPUTS = (
    "chain_robustness_sweep.json",
    "workcell_geometry_check.json",
    "insert_attitude_wall_moved.json",
    "service_latch_clearance.json",
    "robot_carried_rigid_mating_refuted.json",
    "workflow_robot_carried_m130pin_guarded_c11065_certification.json",
)
PROTOCOL = {
    "name": "serviceability_boundary_validation_v2",
    "simulation_loss_rule": (
        "point Wilson-95 upper bound is below nominal Wilson-95 lower bound"
    ),
    "analytical_law_ratio_gate": {"low": 0.85, "high": 1.15},
    "parked_base_kinematic_gate": {
        "maximum_position_residual_m": 1.0e-4,
        "maximum_attitude_residual_rad": 1.0e-4,
        "minimum_worst_axis_authority": 0.9,
    },
    "decision_rule": (
        "a feasible point supports the boundary only without a separated loss; "
        "an infeasible point supports it only with a separated loss"
    ),
    "qualification_rule": (
        "every named dimension must have supporting analytical and simulation evidence; "
        "a mismatch, missing arm, or idealized-only arm fails qualification"
    ),
    "tolerances_changed_for_this_report": False,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence")
    parser.add_argument("--manifest", type=Path, default=ROOT / "evidence" / "MANIFEST.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--robustness-sweep",
        type=Path,
        default=None,
        help=(
            "Use this sweep report in place of evidence/chain_robustness_sweep.json. "
            "The 16-episode sweep is a ranking instrument whose Wilson intervals are "
            "about twenty points wide; a re-measured point at a larger sample answers "
            "the same question with a narrower one. The protocol and every tolerance "
            "are unchanged, and the substitution is recorded in the report."
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_inputs(evidence_dir: Path, manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load only named canonical reports and bind their exact bytes."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = manifest.get("canonical")
    if not isinstance(canonical, dict):
        raise ValueError("manifest has no canonical evidence group")
    reports: dict[str, Any] = {}
    bindings: list[dict[str, Any]] = []
    for name in REQUIRED_INPUTS:
        if name not in canonical:
            raise ValueError(f"required input is not canonical: {name}")
        path = evidence_dir / name
        if not path.is_file():
            raise ValueError(f"required canonical input is missing: {path}")
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
        bindings.append({"file": name, "sha256": _sha256(path), "manifest_status": "canonical"})
    return reports, bindings


def recompute_geometry() -> dict[str, Any]:
    """Run the existing simulator-validated analytical model on current source."""

    path = ROOT / "scripts" / "check_workcell_geometry.py"
    spec = importlib.util.spec_from_file_location("serviceability_geometry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shipped = tuple(float(value) for value in module._literal("GRAPPLE_ROBOT_ROOT_POS"))
    return {
        "lateral_clearance_window": module.lateral_clearance_window(),
        "module_section_envelope": module.section_envelope(),
        "recorded_seating_sweep_against_the_law": module.explain_seating_sweep(),
        "boundary_section_envelope": module.section_envelope(
            widths_m=(0.120, 0.130, 0.140), heights_m=(0.016, 0.020, 0.026)
        ),
        "parked_base_offset_profiles": {
            "nominal": module.parked_base_offset_profile(
                base_x_m=shipped[0], stop_error_y_m=0.0, tool_z_m=0.72, base_z_m=shipped[2]
            ),
            "base_x_-0.70": module.parked_base_offset_profile(
                base_x_m=-0.70, stop_error_y_m=0.0, tool_z_m=0.72, base_z_m=shipped[2]
            ),
            "base_y_+10mm": module.parked_base_offset_profile(
                base_x_m=shipped[0], stop_error_y_m=0.010, tool_z_m=0.72, base_z_m=shipped[2]
            ),
        },
    }


def require_current_geometry(recorded: dict[str, Any], recomputed: dict[str, Any]) -> None:
    """Fail if the preserved analytical report no longer describes current code."""

    for field in (
        "lateral_clearance_window",
        "module_section_envelope",
        "recorded_seating_sweep_against_the_law",
    ):
        current = recomputed[field]
        if recorded.get(field) != current:
            raise ValueError(
                f"workcell_geometry_check.json field {field!r} is stale; "
                "regenerate and review it before validating the boundary"
            )
    validation = recorded.get("kinematic_validation", {})
    if validation.get("passed") is not True:
        raise ValueError("the analytical model did not pass its simulator-configuration check")


def _wilson(point: dict[str, Any]) -> tuple[float, float]:
    interval = point.get("wilson_95") or point.get("success_rate_wilson_95")
    if not isinstance(interval, dict) or "low" not in interval or "high" not in interval:
        raise ValueError("simulation point has no Wilson-95 interval")
    return float(interval["low"]), float(interval["high"])


def classify_simulation_point(
    *, label: str, analytically_feasible: bool, point: dict[str, Any], nominal: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen support rule without tuning or discarding the losing arm."""

    nominal_low, nominal_high = _wilson(nominal)
    point_low, point_high = _wilson(point)
    separated_loss = point_high < nominal_low
    supports = (analytically_feasible and not separated_loss) or (
        not analytically_feasible and separated_loss
    )
    return {
        "label": label,
        "analytically_feasible": analytically_feasible,
        "simulation": {
            "episodes": int(point["episodes"]),
            "success_rate": float(point["success_rate"]),
            "wilson_95": {"low": point_low, "high": point_high},
            "nominal_success_rate": float(nominal["success_rate"]),
            "nominal_wilson_95": {"low": nominal_low, "high": nominal_high},
            "statistically_separated_loss": separated_loss,
        },
        "comparison": "supports_boundary" if supports else "does_not_support_boundary",
    }


def _dimension_status(points: list[dict[str, Any]], note: str) -> dict[str, Any]:
    """Derive a dimension's status from its own rows instead of asserting it.

    **These statuses used to be literals.** `"status": "mismatch"` was written
    beside the rows rather than computed from them, which was true when it was
    written and would have stayed "true" if the rows changed underneath it. A
    fail-closed validator cannot have an authored verdict: the corrected
    clearance arm is exactly the case that would have been reported as a
    mismatch while its own rows said otherwise.
    """

    dissenting = [row["label"] for row in points if row["comparison"] != "supports_boundary"]
    if dissenting:
        reason = f"{note} Points that do not support the boundary: {', '.join(dissenting)}."
    else:
        reason = f"{note} Every point supports the boundary."
    return {
        "status": "mismatch" if dissenting else "supported_in_simulation",
        "points_not_supporting": dissenting,
        "reason": reason,
    }


def _section(geometry: dict[str, Any], width: float, height: float) -> dict[str, Any]:
    rows = geometry["boundary_section_envelope"]["sections"]
    matches = [
        row
        for row in rows
        if abs(float(row["width_m"]) - width) < 1.0e-9
        and abs(float(row["height_m"]) - height) < 1.0e-9
    ]
    if len(matches) != 1:
        raise ValueError(f"analytical section grid has {len(matches)} rows for {width} x {height}")
    return matches[0]


def _parked_base_feasible(profile: dict[str, Any]) -> bool:
    gate = PROTOCOL["parked_base_kinematic_gate"]
    return bool(
        float(profile["maximum_position_residual_m"])
        <= float(gate["maximum_position_residual_m"])
        and float(profile["maximum_attitude_residual_rad"])
        <= float(gate["maximum_attitude_residual_rad"])
        and float(profile["minimum_worst_axis_authority"])
        >= float(gate["minimum_worst_axis_authority"])
    )


def _load_path_rows(rigid: dict[str, Any], compliant: dict[str, Any]) -> dict[str, Any]:
    overall = compliant["overall"]
    rigid_base = rigid["base_mount_compliance"]
    return {
        "status": "idealized_unpaired_only",
        "compliant_form_lock_arm": {
            "episodes": int(overall["episodes"]),
            "successes": int(overall["successes"]),
            "success_rate": float(overall["success_rate"]),
            "joint_model": "bounded spring-damper between wrist and module",
        },
        "rigid_form_lock_arm": {
            "episodes": 1,
            "completed": bool(rigid["completed"]),
            "joint_model": "break-rated fixed joint between wrist and module",
            "final": rigid.get("final", {}),
        },
        "paired_states_and_seeds": False,
        "contact_reaction_loads_available": False,
        "base_mount_ablation": {
            "authored": bool(rigid_base["authored"]),
            "robot_root_fixed_to_world": bool(rigid_base["robot_root_fixed_to_world"]),
            "in_load_path": bool(rigid_base["in_load_path"]),
            "max_measured_mount_rotation_rad": float(rigid_base["max_measured_mount_rotation_rad"]),
            "claim_supported": bool(rigid_base["claim_supported_about_base_compliance_tolerance"]),
            "status": "excluded_fixed_root",
        },
        "conclusion": (
            "The rigid failure and compliant-chain success motivate compliance, but they are not a "
            "paired causal qualification and neither arm exposes physical contact/load reactions."
        ),
    }


def build_report(
    reports: dict[str, Any],
    input_bindings: list[dict[str, Any]],
    source_revision: dict[str, Any],
    current_geometry: dict[str, Any],
) -> dict[str, Any]:
    geometry = reports["workcell_geometry_check.json"]
    require_current_geometry(geometry, current_geometry)
    sweep = reports["chain_robustness_sweep.json"]
    points = sweep["points"]
    nominal = points["nominal"]

    clearance = geometry["lateral_clearance_window"]
    lower = float(clearance["lower_bound_m"])
    upper = float(clearance["upper_bound_m"])
    clearance_rows = []
    for label, value, point_name in (
        ("6 mm per side", 0.006, "rack_lat_6mm"),
        ("as built", float(clearance["as_built_m"]), "nominal"),
        ("16 mm per side", 0.016, "rack_lat_16mm"),
    ):
        row = classify_simulation_point(
            label=label,
            analytically_feasible=lower <= value <= upper,
            point=points[point_name],
            nominal=nominal,
        )
        row["clearance_per_side_m"] = value
        clearance_rows.append(row)

    section_rows = []
    for label, width, height, point_name in (
        ("120 x 16 mm", 0.120, 0.016, "section_120x16"),
        ("130 x 20 mm", 0.130, 0.020, "nominal"),
        ("140 x 26 mm", 0.140, 0.026, "section_140x26"),
    ):
        analytical = _section(current_geometry, width, height)
        row = classify_simulation_point(
            label=label,
            analytically_feasible=bool(analytical["accepted"]),
            point=points[point_name],
            nominal=nominal,
        )
        row["analytical"] = analytical
        section_rows.append(row)

    base_profiles = current_geometry["parked_base_offset_profiles"]
    base_rows = []
    for label, profile_name, point_name in (
        ("nominal parked base", "nominal", "nominal"),
        ("base x = -0.70 m", "base_x_-0.70", "base_x_-0.70"),
        ("base y stop error = +10 mm", "base_y_+10mm", "base_y_+10mm"),
    ):
        profile = base_profiles[profile_name]
        row = classify_simulation_point(
            label=label,
            analytically_feasible=_parked_base_feasible(profile),
            point=points[point_name],
            nominal=nominal,
        )
        row["analytical"] = profile
        base_rows.append(row)

    ratio_gate = PROTOCOL["analytical_law_ratio_gate"]
    seating = current_geometry["recorded_seating_sweep_against_the_law"]
    ratios = [float(row["ratio"]) for row in seating["points"]]
    wall = reports["insert_attitude_wall_moved.json"]
    wall_ratios = [float(arm["floor_over_wall"]) for arm in wall["arms"]]
    entry_supported = all(ratio_gate["low"] <= ratio <= ratio_gate["high"] for ratio in ratios + wall_ratios)

    latch = reports["service_latch_clearance.json"]
    latch_checks_pass = bool(latch["checks"]) and all(bool(check["passed"]) for check in latch["checks"])
    load_path = _load_path_rows(
        reports["robot_carried_rigid_mating_refuted.json"],
        reports["workflow_robot_carried_m130pin_guarded_c11065_certification.json"],
    )

    dimensions = {
        "rack_clearance": {
            **_dimension_status(
                clearance_rows,
                "The two-sided lateral clearance bound, against the swept clearance points.",
            ),
            "analytical_window_m_per_side": {"low": lower, "high": upper},
            "points": clearance_rows,
        },
        "module_section": {
            **_dimension_status(
                section_rows,
                "The intersection of the entry and grip bounds, against the swept module sections.",
            ),
            "points": section_rows,
        },
        "base_offset": {
            **_dimension_status(
                base_rows,
                (
                    "The parked-base kinematic gate, against the swept base positions. A rail stop "
                    "error that clears this gate is kinematically feasible; a simulated loss at one "
                    "that does is controller, rail-stop or hand-off sensitivity rather than a "
                    "kinematic boundary, and the closed form does not claim to predict it."
                ),
            ),
            "kinematic_gate": PROTOCOL["parked_base_kinematic_gate"],
            "points": base_rows,
        },
        "entry_attitude": {
            "status": "supported_in_simulation" if entry_supported else "mismatch",
            "ratio_gate": ratio_gate,
            "recorded_clearance_sweep_ratios": ratios,
            "unchanged_checkpoint_wall_move_ratios": wall_ratios,
            "unchanged_checkpoint": True,
            "reason": "The measured seating wall follows 2c/L within the frozen ratio gate.",
        },
        "capture_geometry": {
            "status": "analytical_only",
            "analytical_clearance_checks": len(latch["checks"]),
            "all_analytical_clearance_checks_pass": latch_checks_pass,
            "canonical_contact_or_load_arm": None,
            "reason": "Visual bounding-volume clearance is derived; contact retention/load capacity is not canonically validated.",
        },
        "load_path_type": load_path,
    }
    qualification_passed = all(
        dimension.get("status") == "supported_in_simulation" for dimension in dimensions.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Serviceability boundary validation against simulation",
        "evidence_type": "analytical_boundary_against_simulation",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision,
        "protocol": PROTOCOL,
        "input_bindings": input_bindings,
        "analytical_model_check": {
            "recomputed_from_current_source": True,
            "matches_preserved_geometry_report": True,
            "simulator_configurations_checked": int(geometry["kinematic_validation"]["configurations_checked"]),
            "simulator_configuration_check_passed": True,
        },
        "dimensions": dimensions,
        "decision": {
            "qualified": qualification_passed,
            "status": "qualified" if qualification_passed else "not_qualified",
            "supported_dimensions": [
                name for name, value in dimensions.items() if value.get("status") == "supported_in_simulation"
            ],
            "blocking_dimensions": [
                name for name, value in dimensions.items() if value.get("status") != "supported_in_simulation"
            ],
        },
        "required_next_arms": [
            "Increase episodes around both rack-clearance bounds; keep the 6 mm, nominal, and 16 mm arms.",
            "Repeat the 120 x 16, nominal, and 140 x 26 mm section arms at qualification episode counts.",
            "Bracket the base-y controller/rail-stop failure around +10 mm on identical seeds.",
            "Run a canonical contact-retention/load arm for the capture interface; do not substitute visual overlap.",
            "Pair rigid and compliant form-lock arms on identical initial states and seeds with reaction-load telemetry.",
            "Make base compliance part of the articulation load path before claiming a compliance tolerance.",
        ],
        "scope_and_limitations": [
            "Simulation only; no hardware qualification is represented.",
            "Robustness points have 16 episodes each, so overlapping Wilson intervals remain unresolved.",
            "The latch is visual geometry and the simulator force readings are not treated as hardware load evidence.",
            "The robot root is fixed to the world; compliant-base tolerance is explicitly excluded.",
            "No failed or losing arm was removed, and no tolerance was widened for this report.",
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    source = git_source_revision(ROOT)
    if not source.get("available") or not source.get("commit"):
        raise SystemExit("cannot identify the source revision")
    if source.get("dirty") is not False:
        raise SystemExit("refusing to generate boundary evidence from a dirty tracked worktree")
    reports, bindings = load_inputs(args.evidence_dir, args.manifest)
    substitution: dict[str, Any] | None = None
    if args.robustness_sweep is not None:
        # The substituted sweep is required to be canonical too, and its own
        # per-point episode counts travel into the report, so a narrower
        # interval cannot be quoted without saying what produced it.
        name = args.robustness_sweep.name
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if name not in manifest.get("canonical", {}):
            raise SystemExit(f"substituted sweep is not canonical: {name}")
        replacement = json.loads(args.robustness_sweep.read_text(encoding="utf-8"))
        reports["chain_robustness_sweep.json"] = replacement
        bindings.append(
            {
                "file": name,
                "sha256": _sha256(args.robustness_sweep),
                "manifest_status": "canonical",
                "substituted_for": "chain_robustness_sweep.json",
            }
        )
        substitution = {
            "substituted_for": "chain_robustness_sweep.json",
            "with": name,
            "why": (
                "the shipped sweep is a 16-episode ranking instrument whose Wilson interval is "
                "about twenty points wide, which cannot separate a loss from the nominal point"
            ),
            "episodes_per_point": {
                tag: int(point.get("episodes", 0)) for tag, point in replacement["points"].items()
            },
            "protocol_or_tolerances_changed": False,
        }
    report = build_report(reports, bindings, source, recompute_geometry())
    if substitution is not None:
        report["input_substitution"] = substitution
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    return 0 if report["decision"]["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
