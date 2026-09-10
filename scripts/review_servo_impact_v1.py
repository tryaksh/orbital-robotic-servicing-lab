"""CPU verification of the registered fixed-480-Hz servo impact cause test."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import sha256, write_json  # noqa: E402
from scripts import review_contact_impact_v1 as metrics  # noqa: E402
from scripts import review_contact_impact_v3 as inherited  # noqa: E402

SERVO_REGISTRATION = "configs/servo_impact_registration_v1.json"
SERVO_SOURCES = (
    "src/assembly_recovery/servo_impact_timing_v1.py",
    "src/assembly_recovery/servo_impact_env_v1.py",
    "scripts/probe_servo_impact_v1.py",
    "scripts/review_servo_impact_v1.py",
)
PAIRING_KEYS = (
    "initial", "sensor_initial", "physical_initial", "sensor_draws",
    "cuda_rng_before_steps", "dead_zone", "requested_action",
)


def servo_condition_checks(report, data):
    """Check the changed feedback cadence while retaining original sensor timing.

    Observed servo ticks come from the worker's actual update-count increment.
    Merely labelling a run 480 Hz or repeating the expected modulo is insufficient.
    """
    r = report["refinement"]
    if type(r) is not int or r not in (4, 8):
        raise ValueError("Servo cause test accepts only native 480/960 Hz")
    steps, stride = 480 * r, r // 4
    native = data["native"]
    expected_ticks = torch.arange(steps) % stride == 0
    condition = report["controller_condition"]
    return {
        "timing_counts": report["native_physics_steps"] == steps
            and report["servo_updates"] == 1920 and report["sensor_updates"] == 480,
        "held_native_torque": torch.equal(
            native["torques"], native["torques"][::stride].repeat_interleave(stride, 0)),
        "held_native_gripper_targets": torch.equal(
            native["gripper_targets"], native["gripper_targets"][::stride].repeat_interleave(stride, 0)),
        "explicit_servo_condition": all(condition[key] == expected for key, expected in {
            "external_servo_hz": 480, "external_sensor_hz": 120, "policy_hz": 15,
            "native_physics_hz": 120 * r, "initialization_physics_hz": 120,
        }.items()),
        "observed_native_servo_ticks": native["servo_tick"].dtype == torch.bool
            and torch.equal(native["servo_tick"], expected_ticks),
        "observed_native_servo_update_counts": torch.equal(
            native["servo_updates_completed"], expected_ticks.to(torch.float64).cumsum(0)),
    }


def verify_single(run: Path):
    try:
        return _verify_single(Path(run))
    except Exception as exc:
        return {
            "status": "verification_error", "checks": {"servo_verifier_completed": False},
            "failed_checks": ["servo_verifier_completed"], "error": f"{type(exc).__name__}: {exc}",
            "verifier_version": "servo_impact_review_v1",
        }


def _verify_single(run):
    # Preserve every physical, source, replay, actor, force, gravity and noise
    # check from v3. Only the three explicitly changed cadence checks are replaced.
    result = inherited._verify_single(run)
    superseded = {
        key: result["checks"][key]
        for key in ("timing_counts", "held_native_torque", "held_native_gripper_targets")
    }
    manifest = json.loads((run / "manifest.json").read_text())
    report, data = metrics.load_trace(run)
    result["checks"].update(servo_condition_checks(report, data))
    with zipfile.ZipFile(run / "source.zip") as archive:
        raw = archive.read(SERVO_REGISTRATION)
        registration = json.loads(raw)
        result["checks"]["archived_servo_registration"] = (
            hashlib.sha256(raw).hexdigest() == manifest["servo_registration_sha256"]
            and registration == manifest["servo_registration"]
        )
        result["checks"]["registered_servo_condition"] = all(
            registration[key] == expected for key, expected in {
                "version": "servo_impact_registration_v1", "native_physics_hz": [480, 960],
                "external_servo_hz": 480, "external_sensor_hz": 120, "policy_hz": 15,
                "seed": 10071, "requests_per_run": 28, "prefix_controls": 60, "initialization_hz": 120,
            }.items()
        )
        result["checks"]["registered_material_margins_unchanged"] = (
            registration["acceptance"] == manifest["registration"]["acceptance"]
        )
        result["checks"]["actual_servo_sources_match_archive"] = all(
            hashlib.sha256(archive.read(name)).hexdigest() == manifest["source_hashes"][name]
            == sha256(ROOT / name) for name in SERVO_SOURCES
        )
        continuation = archive.read("configs/contact_impact_continuation_v2.json")
        result["checks"]["required_archived_continuation_registration"] = (
            hashlib.sha256(continuation).hexdigest() == manifest["continuation_registration_sha256"]
            and json.loads(continuation) == manifest["continuation_registration"]
        )
    result["checks"]["launcher_selected_servo_worker"] = any(
        Path(part).name == "probe_servo_impact_v1.py" for part in manifest["command"]
    )
    result["checks"]["no_inapplicable_coarse_reference"] = result["reference"] is None
    result.update(
        status="verified" if all(result["checks"].values()) else "check_failed",
        failed_checks=[key for key, passed in result["checks"].items() if not passed],
        verifier_version="servo_impact_review_v1", verifier_source_sha256=sha256(Path(__file__)),
        inherited_verifier_source_sha256=sha256(ROOT / "scripts/review_contact_impact_v3.py"),
        servo_registration_sha256=manifest["servo_registration_sha256"],
        controller_condition=report["controller_condition"], superseded_timing_checks=superseded,
        timing_change={
            "external_servo_hz": 480, "external_sensor_hz": 120, "policy_hz": 15,
            "native_torque_hold_stride": report["refinement"] // 4,
            "reason": "Registered feedback condition changes servo updates from480 to1920 per four-second prefix. Sensors remain120Hz.",
            "scope": "A changed controller condition, not an original120Hz reference replay or candidate-policy effect.",
        },
    )
    return result


def describe_run(report, data, *, servo_hz=480):
    """Keep original metrics; label its 120-Hz aggregates as sensor intervals."""
    result = metrics.describe_run(report, data)
    result["external_servo_hz"] = servo_hz
    result["external_sensor_hz"] = 120
    result["policy_hz"] = 15
    for row in result["cases"]:
        intervals = row.pop("contact_servo_intervals")
        for interval in intervals:
            interval["sensor_interval_index"] = interval.pop("servo_interval_index")
            interval["external_servo_updates_per_complete_interval"] = servo_hz // 120
        row["contact_sensor_intervals"] = intervals
    result["interval_scope"] = (
        "Inherited1/120-second block aggregates are sensor intervals; each complete interval contains "
        f"{servo_hz // 120} external servo updates. Blocks crossing native terminals stay marked."
    )
    return result


def common_active_actions(left_trace, right_trace):
    left, right = left_trace["applied_action"], right_trace["applied_action"]
    active_left, active_right = left_trace["active"], right_trace["active"]
    if left.shape != right.shape or active_left.shape != left.shape[:2] or active_right.shape != left.shape[:2]:
        raise ValueError("Applied actions require paired control/environment activity masks")
    if active_left.dtype != torch.bool or active_right.dtype != torch.bool:
        raise ValueError("Recorded pre-control activity must be boolean")
    common = active_left & active_right
    differences = (left != right).any(-1)
    return {
        "common_active_actions_equal": torch.equal(left[common], right[common]),
        "compared_active_controls": int(common.sum()),
        "excluded_controls": int((~common).sum()),
        "excluded_action_differences": int((differences & ~common).sum()),
        "scope": "Only controls active in both jobs gate applied-action parity. Different native terminal times may legitimately change absorbing holds.",
    }


def pairing_checks(left_report, left_trace, right_report, right_trace):
    checks = {"identical_cases": left_report["cases"] == right_report["cases"]}
    checks.update({key: metrics.equal_tree(left_trace[key], right_trace[key]) for key in PAIRING_KEYS})
    checks["applied_actions_common_active"] = common_active_actions(left_trace, right_trace)["common_active_actions_equal"]
    return checks


def _read_verified_runs(paths, *, servo_hz):
    verifications, summaries, traces, reports = [], {}, {}, {}
    for run in paths:
        verification = verify_single(run) if servo_hz == 480 else inherited.verify_single(run)
        verification = {**verification, "run_path": str(run), "external_servo_hz": servo_hz}
        verifications.append(verification)
        if not all(verification["checks"].values()):
            continue
        report, trace = metrics.load_trace(run)
        hz = 120 * report["refinement"]
        if hz in summaries:
            raise ValueError("Duplicate native resolutions cannot be pooled")
        if hz not in (480, 960):
            raise ValueError("Both controller conditions require native480/960Hz")
        summaries[hz] = describe_run(report, trace, servo_hz=servo_hz)
        traces[hz], reports[hz] = trace, report
    return verifications, summaries, traces, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=2, type=Path, required=True)
    parser.add_argument("--original-runs", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence; choose a unique output")
    registration = json.loads((ROOT / SERVO_REGISTRATION).read_text())
    verifications, summaries, traces, reports = _read_verified_runs(args.runs, servo_hz=480)
    complete_pair = sorted(summaries) == [480, 960]
    cross_checks, cross_action_scope = {}, {}
    pair = None
    if complete_pair:
        cross_checks = pairing_checks(reports[480], traces[480], reports[960], traces[960])
        cross_action_scope = common_active_actions(traces[480], traces[960])
        pair = metrics.compare_pair(summaries[480], summaries[960], traces[480], traces[960])
        pair["external_servo_hz"] = 480
        pair["external_sensor_hz"] = 120
        pair["interpretation"] += " Both resolutions use the registered480Hz servo condition."
    registration_checks = {
        str(run): verification.get("servo_registration_sha256") == sha256(ROOT / SERVO_REGISTRATION)
        for run, verification in zip(args.runs, verifications, strict=True)
    }
    original_verifications, originals, original_traces, original_reports = [], {}, {}, {}
    servo_pairing, servo_effects, original_pair = {}, [], None
    servo_action_scope = {}
    if args.original_runs:
        original_verifications, originals, original_traces, original_reports = _read_verified_runs(args.original_runs, servo_hz=120)
        if sorted(originals) == [480, 960]:
            original_pair = metrics.compare_pair(
                originals[480], originals[960], original_traces[480], original_traces[960])
            original_pair["external_servo_hz"] = 120
        for hz in sorted(set(originals) & set(summaries)):
            servo_pairing[str(hz)] = pairing_checks(
                original_reports[hz], original_traces[hz], reports[hz], traces[hz])
            servo_action_scope[str(hz)] = common_active_actions(original_traces[hz], traces[hz])
            comparison = metrics.compare_pair(originals[hz], summaries[hz], original_traces[hz], traces[hz])
            comparison.update(
                native_physics_hz=hz, external_servo_hz=[120, 480],
                interpretation="Descriptive effect of changing feedback frequency at fixed native resolution; inherited margins are descriptive and cannot certify this controller condition. All active-exposure differences retained.",
            )
            servo_effects.append(comparison)
    new_verified = (
        complete_pair and all(all(v["checks"].values()) for v in verifications)
        and all(cross_checks.values()) and all(registration_checks.values())
    )
    original_verified = not args.original_runs or (
        sorted(originals) == [480, 960]
        and all(all(v["checks"].values()) for v in original_verifications)
        and len(servo_pairing) == 2 and all(all(row.values()) for row in servo_pairing.values())
    )
    verified = new_verified and original_verified
    result = {
        "schema": 1, "status": "verified_diagnostic" if verified else "check_failed", "research_result": False,
        "verifier_version": "servo_impact_review_v1",
        "servo_registration_sha256": sha256(ROOT / SERVO_REGISTRATION),
        "registered_acceptance": registration["acceptance"],
        "controller_condition": {"external_servo_hz": 480, "external_sensor_hz": 120, "policy_hz": 15},
        "verification": verifications, "registration_matches_current": registration_checks,
        "required_native_pair_present": complete_pair,
        "cross_resolution_initialization_rng_action_checks": cross_checks,
        "cross_resolution_applied_action_scope": cross_action_scope,
        "runs": [summaries[hz] for hz in sorted(summaries)], "pairs": [pair] if pair else [],
        "original_condition_verification": original_verifications,
        "original_condition_runs": [originals[hz] for hz in sorted(originals)],
        "original_condition_pair": original_pair,
        "cross_servo_initialization_rng_action_checks": servo_pairing,
        "cross_servo_applied_action_scope": servo_action_scope,
        "fixed_native_servo_effects": servo_effects,
        "decision": {
            "finest_prefix_pair_pass": bool(new_verified and pair and pair["material_margins_pass"]),
            "finest_pair_hz": [480, 960] if pair else None,
            "servo_effect_comparisons_performed": bool(args.original_runs and original_verified and new_verified),
            "original_pair_material_margins_pass": original_pair["material_margins_pass"] if original_pair else None,
            "training_authorized_by_prefix": False,
        },
        "relative_difference_definition": "abs(a-b)/max(abs(a),abs(b),1e-12); original per-case active-prefix medians over all28, including zero-contact cases.",
        "scope_and_limitations": registration["scope_and_limitations"] + [
            "All inherited force, observation, noise, RNG, physical accounting, source and native CPU replay checks remain required.",
            "The only replaced checks are servo update count and native torque/gripper-target hold stride, with additional observed-cadence controls.",
            "No120Hz prefix reference parity applies to this explicitly changed480Hz controller condition.",
            "Normal contact impulses and wrist wrench magnitudes are different quantities; this does not establish a momentum balance or uncensored peak convergence.",
            "A prefix stability result alone cannot establish full-job policy validity, training benefit or hardware fidelity.",
            "Without --original-runs, this artifact contains only within-condition refinement and does not estimate a change-of-servo effect.",
        ],
    }
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "failed_checks": [v.get("failed_checks", []) for v in verifications + original_verifications]}))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
