"""CPU replay and analysis of the final registered Cartesian-reference cause test."""
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
from scripts import review_servo_impact_v1 as inherited  # noqa: E402

REGISTRATION = "configs/approach_slew_registration_v1.json"
SOURCES = (
    "src/assembly_recovery/approach_slew_v1.py",
    "src/assembly_recovery/approach_slew_env_v1.py",
    "scripts/probe_approach_slew_v1.py",
    "scripts/review_approach_slew_v1.py",
)
POSITION_FIELDS = (
    "requested_cartesian_reference", "slew_reference", "servo_tool_pos_before",
    "clamped_impedance_target", "final_impedance_target", "reward_delta_pos",
)


def _position_comparison(actual, expected, *scale_points):
    # Two independently rounded position calculations use a conservative four
    # unit-roundoff allowance at their world-coordinate scale. This is a CPU
    # arithmetic check only; no force, job or reference-speed setting changes.
    values = (actual, expected, *scale_points)
    scale = torch.stack([torch.linalg.vector_norm(value.double(), dim=-1) for value in values]).amax(0)
    tolerance = 4 * (torch.finfo(torch.float32).eps / 2) * (scale + 0.02 / 480)
    error = (actual.double() - expected.double()).abs().amax(-1)
    return bool((error <= tolerance).all()), float(error.max())


def replay_reference(report, data):
    """Replay every update from the captured initial reference and native poses."""
    r = report["refinement"]
    if type(r) is not int or r not in (4, 8):
        raise ValueError("Reference replay requires native480/960Hz")
    native, initial = data["native"], data["initial"]
    steps, environments = native["tool_pos"].shape[:2]
    if any(native[name].shape != (steps, environments, 3) for name in POSITION_FIELDS):
        raise ValueError("Every reference/position/delta trace needs native-time/environment/XYZ shape")
    previous = data["controller_initial"]["slew_reference"].double()
    tool_initial = initial["fingertip_midpoint_pos"].double()
    hold = tool_initial.clone()
    hold_quat = initial["fingertip_midpoint_quat"].clone()
    active_before = torch.ones(environments, dtype=torch.bool)
    checks = {
        "reference_initial_is_between_job_tool_pose": torch.equal(
            data["controller_initial"]["slew_reference"], initial["fingertip_midpoint_pos"]),
        "desired_reference_from_noisy_actor_frame": True,
        "pre_servo_tool_matches_prior_native_pose": True,
        "euclidean_reference_recurrence": True,
        "reference_step_bound": True,
        "original_position_error_clamp": True,
        "original_reward_delta_pos": True,
        "terminal_hold_target": True,
        "quaternion_terminal_hold_unchanged": True,
        "incoming_quaternion_normalized": bool((native["incoming_impedance_quat"].double().norm(dim=-1) - 1).abs().max() < 1e-5),
        "original_zero_gripper_command_and_targets": bool((native["incoming_gripper_command"] == 0).all() and (native["gripper_targets"] == 0).all()),
        "reference_records_held_between_servo_ticks": True,
        "slew_count_matches_actual_servo_count": torch.equal(
            native["slew_updates_completed"], native["servo_updates_completed"]),
    }
    maxima = {key: 0. for key in (
        "desired_error_m", "tool_before_error_m", "reference_recurrence_error_m",
        "reference_step_m", "reference_step_rounding_allowance_m",
        "clamp_error_m", "reward_delta_error_m", "terminal_hold_error_m")}
    frame = data["sensor_initial"]["fixed_pos_obs_frame"].double()
    noise = initial["init_fixed_pos_obs_noise"].double()
    bounds = torch.tensor(report["controller_position_bounds"], dtype=torch.float64)
    threshold = initial["pos_threshold"].double()
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    increment = 0.02 / 480
    for step in range(steps):
        if bool(native["servo_tick"][step]):
            control = step // (8 * r)
            desired = frame + noise + data["applied_action"][control, :, :3].double() * bounds
            tool = tool_initial if step == 0 else native["tool_pos"][step - 1].double()
            recorded_desired = native["requested_cartesian_reference"][step].double()
            recorded_reference = native["slew_reference"][step].double()
            matched, error = _position_comparison(recorded_desired, desired, tool)
            checks["desired_reference_from_noisy_actor_frame"] &= matched
            maxima["desired_error_m"] = max(maxima["desired_error_m"], error)
            matched, error = _position_comparison(native["servo_tool_pos_before"][step], tool)
            checks["pre_servo_tool_matches_prior_native_pose"] &= matched
            maxima["tool_before_error_m"] = max(maxima["tool_before_error_m"], error)
            difference = recorded_desired - previous
            length = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
            expected_reference = previous + difference * (increment / length.clamp_min(torch.finfo(torch.float64).tiny)).clamp(max=1)
            matched, error = _position_comparison(recorded_reference, expected_reference, previous, recorded_desired)
            checks["euclidean_reference_recurrence"] &= matched
            maxima["reference_recurrence_error_m"] = max(maxima["reference_recurrence_error_m"], error)
            displacement = torch.linalg.vector_norm(recorded_reference - previous, dim=-1)
            position_scale = torch.maximum(torch.linalg.vector_norm(previous, dim=-1),
                                           torch.linalg.vector_norm(recorded_reference, dim=-1))
            # One rounded reference addition plus conservative direction/norm
            # arithmetic; permits representation error, not a faster setting.
            step_allowance = 2 * unit_roundoff * position_scale + 8 * unit_roundoff * increment
            checks["reference_step_bound"] &= bool((displacement <= increment + step_allowance).all())
            maxima["reference_step_m"] = max(maxima["reference_step_m"], float(displacement.max()))
            maxima["reference_step_rounding_allowance_m"] = max(maxima["reference_step_rounding_allowance_m"], float(step_allowance.max()))
            expected_clamp = tool + torch.minimum(torch.maximum(recorded_reference - tool, -threshold), threshold)
            matched, error = _position_comparison(native["clamped_impedance_target"][step], expected_clamp, tool, recorded_reference)
            checks["original_position_error_clamp"] &= matched
            maxima["clamp_error_m"] = max(maxima["clamp_error_m"], error)
            matched, error = _position_comparison(
                native["reward_delta_pos"][step], recorded_desired - tool, recorded_desired, tool)
            checks["original_reward_delta_pos"] &= matched
            maxima["reward_delta_error_m"] = max(maxima["reward_delta_error_m"], error)
            expected_final = torch.where(active_before[:, None], expected_clamp, hold)
            matched, error = _position_comparison(native["final_impedance_target"][step], expected_final, tool)
            checks["terminal_hold_target"] &= matched
            maxima["terminal_hold_error_m"] = max(maxima["terminal_hold_error_m"], error)
            expected_quat = torch.where(active_before[:, None], native["incoming_impedance_quat"][step], hold_quat)
            checks["quaternion_terminal_hold_unchanged"] &= torch.equal(native["final_impedance_quat"][step], expected_quat)
            # Local recurrence is checked before accepting each represented
            # float32 state, starting from the exact captured initial state.
            previous = recorded_reference
        elif step:
            checks["reference_records_held_between_servo_ticks"] &= all(
                torch.equal(native[name][step], native[name][step - 1]) for name in POSITION_FIELDS + ("incoming_impedance_quat", "final_impedance_quat", "incoming_gripper_command"))
        active_after = native["active_after"][step]
        newly_terminal = active_before & ~active_after
        hold = torch.where(newly_terminal[:, None], native["tool_pos"][step].double(), hold)
        hold_quat = torch.where(newly_terminal[:, None], native["tool_quat"][step], hold_quat)
        active_before = active_after
    return checks, {
        **maxima, "reference_increment_m": increment, "reference_speed_m_s": 0.02,
        "reference_state_scope": "Local float32 recurrence replay from captured initial tool pose. Every update, clamp, original reward delta and terminal hold is checked; represented previous state is accepted only after its recurrence check.",
        "speed_bound_scope": "Reference increment only, with stated float32 coordinate-addition allowance. No actual tool-speed or force ceiling is inferred.",
    }


def verify_single(run):
    try:
        return _verify_single(Path(run))
    except Exception as exc:
        return {"status": "verification_error", "checks": {"approach_verifier_completed": False},
                "failed_checks": ["approach_verifier_completed"], "error": f"{type(exc).__name__}: {exc}"}


def _verify_single(run):
    result = inherited._verify_single(run)
    # The new worker intentionally supersedes only the old command-path check.
    # All servo/physics/source/reward/force checks remain in the final verdict.
    old_worker_check = result["checks"].pop("launcher_selected_servo_worker")
    manifest = json.loads((run / "manifest.json").read_text())
    report, data = metrics.load_trace(run)
    checks, diagnostics = replay_reference(report, data)
    result["checks"].update(checks)
    result["checks"]["explicit_reference_condition"] = report["controller_condition"]["cartesian_reference_slew_m_s"] == 0.02
    result["checks"]["launcher_selected_approach_worker"] = any(
        Path(part).name == "probe_approach_slew_v1.py" for part in manifest["command"])
    with zipfile.ZipFile(run / "source.zip") as archive:
        raw = archive.read(REGISTRATION)
        registration = json.loads(raw)
        result["checks"]["archived_approach_registration"] = (
            hashlib.sha256(raw).hexdigest() == manifest["approach_registration_sha256"]
            and registration == manifest["approach_registration"])
        result["checks"]["registered_reference_condition"] = all(
            registration[key] == expected for key, expected in {
                "version": "approach_slew_registration_v1", "native_physics_hz": [480, 960],
                "external_servo_hz": 480, "external_sensor_hz": 120, "policy_hz": 15,
                "cartesian_reference_slew_m_s": 0.02, "reference_increment_m_per_servo": 0.02 / 480,
                "seed": 10071, "requests_per_run": 28, "prefix_controls": 60,
            }.items())
        result["checks"]["registered_material_margins_unchanged"] = (
            registration["acceptance"] == manifest["servo_registration"]["acceptance"] == manifest["registration"]["acceptance"])
        result["checks"]["actual_approach_sources_match_archive"] = all(
            hashlib.sha256(archive.read(name)).hexdigest() == manifest["source_hashes"][name] == sha256(ROOT / name)
            for name in SOURCES)
    result.update(
        status="verified" if all(result["checks"].values()) else "check_failed",
        failed_checks=[key for key, passed in result["checks"].items() if not passed],
        verifier_version="approach_slew_review_v1", verifier_source_sha256=sha256(Path(__file__)),
        inherited_servo_verifier_source_sha256=sha256(ROOT / "scripts/review_servo_impact_v1.py"),
        approach_registration_sha256=manifest["approach_registration_sha256"],
        reference_replay=diagnostics, superseded_worker_path_check=old_worker_check,
    )
    return result


def _read_runs(paths, *, approach):
    verifications, summaries, traces, reports = [], {}, {}, {}
    for run in paths:
        verification = verify_single(run) if approach else inherited.verify_single(run)
        verifications.append({**verification, "run_path": str(run)})
        if not all(verification["checks"].values()):
            continue
        report, trace = metrics.load_trace(run)
        hz = 120 * report["refinement"]
        if hz not in (480, 960) or hz in summaries:
            raise ValueError("Require one native480 and one native960 run per condition")
        summary = inherited.describe_run(report, trace)
        summary["cartesian_reference_slew_m_s"] = 0.02 if approach else None
        summaries[hz], traces[hz], reports[hz] = summary, trace, report
    return verifications, summaries, traces, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=2, type=Path, required=True)
    parser.add_argument("--original-runs", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence; choose a unique output")
    registration = json.loads((ROOT / REGISTRATION).read_text())
    verification, summaries, traces, reports = _read_runs(args.runs, approach=True)
    paired = sorted(summaries) == [480, 960]
    pairing, action_scope, pair = {}, {}, None
    if paired:
        pairing = inherited.pairing_checks(reports[480], traces[480], reports[960], traces[960])
        pairing["controller_initial"] = metrics.equal_tree(traces[480]["controller_initial"], traces[960]["controller_initial"])
        action_scope = inherited.common_active_actions(traces[480], traces[960])
        pair = metrics.compare_pair(summaries[480], summaries[960], traces[480], traces[960])
        pair.update(external_servo_hz=480, cartesian_reference_slew_m_s=0.02)
    registration_checks = [
        row.get("approach_registration_sha256") == sha256(ROOT / REGISTRATION) for row in verification]
    original_verification, originals, original_traces, original_reports = [], {}, {}, {}
    effects, condition_pairing, condition_action_scope, original_pair = [], {}, {}, None
    if args.original_runs:
        original_verification, originals, original_traces, original_reports = _read_runs(args.original_runs, approach=False)
        if sorted(originals) == [480, 960]:
            original_pair = metrics.compare_pair(originals[480], originals[960], original_traces[480], original_traces[960])
        for hz in sorted(set(originals) & set(summaries)):
            condition_pairing[str(hz)] = inherited.pairing_checks(
                original_reports[hz], original_traces[hz], reports[hz], traces[hz])
            condition_action_scope[str(hz)] = inherited.common_active_actions(original_traces[hz], traces[hz])
            effect = metrics.compare_pair(originals[hz], summaries[hz], original_traces[hz], traces[hz])
            effect.update(native_physics_hz=hz, external_servo_hz=480, cartesian_reference_slew_m_s=[None, 0.02],
                          interpretation="Descriptive reference-interface effect at fixed native and servo rates; it cannot certify numerical correctness or candidate-policy benefit.")
            effects.append(effect)
    new_verified = paired and all(all(row["checks"].values()) for row in verification) and all(pairing.values()) and all(registration_checks)
    baseline_verified = not args.original_runs or (
        sorted(originals) == [480, 960] and all(all(row["checks"].values()) for row in original_verification)
        and len(condition_pairing) == 2 and all(all(row.values()) for row in condition_pairing.values()))
    support_floor = registration["support_requirement"]["minimum_active_prefix_requests_per_run"]
    support = {str(hz): summaries[hz]["prefix_summary"]["active_at_prefix_end"] >= support_floor for hz in summaries}
    verified = new_verified and baseline_verified
    result = {
        "schema": 1, "status": "verified_diagnostic" if verified else "check_failed", "research_result": False,
        "verifier_version": "approach_slew_review_v1", "approach_registration_sha256": sha256(ROOT / REGISTRATION),
        "registered_acceptance": registration["acceptance"], "registered_support_requirement": registration["support_requirement"],
        "verification": verification, "registration_matches_current": registration_checks,
        "cross_resolution_initialization_rng_action_checks": pairing, "cross_resolution_applied_action_scope": action_scope,
        "runs": [summaries[hz] for hz in sorted(summaries)], "pairs": [pair] if pair else [],
        "original_condition_verification": original_verification,
        "original_condition_runs": [originals[hz] for hz in sorted(originals)],
        "original_condition_pair": original_pair, "fixed_native_reference_effects": effects,
        "cross_condition_initialization_rng_action_checks": condition_pairing,
        "cross_condition_applied_action_scope": condition_action_scope,
        "active_prefix_support_checks": support,
        "decision": {
            "finest_prefix_pair_pass": bool(new_verified and pair and pair["material_margins_pass"]),
            "active_prefix_support_floor_pass": bool(new_verified and len(support) == 2 and all(support.values())),
            "earns_full_job_validation": bool(new_verified and pair and pair["material_margins_pass"] and all(support.values())),
            "training_authorized_by_prefix": False,
            "reference_effect_comparisons_performed": bool(args.original_runs and verified),
        },
        "scope_and_limitations": registration["scope_and_limitations"] + [
            "Original physical/raw20N abort/native replay and float32 precision checks remain required.",
            "Original quaternion generation remains inherited source behavior; quaternion terminal holds, zero gripper commands/targets and Cartesian reference/clamp/reward/hold receive native checks.",
            "All28requests remain in denominators. The active-prefix support floor is separate from stability, stall eligibility and success.",
            "Only common-active applied actions are compared across conditions; reference states and physical trajectories intentionally differ.",
            "This final cause-test variant can earn only full-job validation. It provides no ordinary-versus-recovery-teaching result.",
        ],
    }
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "failed_checks": [row.get("failed_checks", []) for row in verification + original_verification]}))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
