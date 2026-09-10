"""Replay frozen controllers, native physics, reward, sensors and paired timesteps."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.evaluation import JobCriteria  # noqa: E402
from assembly_recovery.faults import development_cases  # noqa: E402
from assembly_recovery.physics_validation_v1 import PhysicsTimingV1, physics_cost_v1, whole_job_summary_v1  # noqa: E402
from assembly_recovery.post_stall import post_stall_completion, summarize_endpoints  # noqa: E402
from assembly_recovery.protocol import sha256, write_json  # noqa: E402
from assembly_recovery.retry_controller import ActorRetryController, RetrySettings  # noqa: E402
from assembly_recovery.study_ppo import StudyPolicy  # noqa: E402
from scripts.probe_training import replay_physics  # noqa: E402
from scripts.verify_training_contract import verify_assets  # noqa: E402

REGISTRATION = ROOT / "configs/learned_physics_validation_v1.json"


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return all(finite_tree(v) for v in value.values()) if isinstance(value, dict) else True


def maximum_error(left, right):
    return float((left.double() - right.double()).abs().max())


def quaternion_product_v1(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), dim=-1)


def rotation_checks_v1(data, report):
    sensor, initial, draws = data["sensors"], data["sensor_initial"], data["sensor_draws"]
    axis = torch.nn.functional.normalize(draws[..., 3:6], dim=-1)
    half = draws[..., 6:7] * math.radians(report["sensor_noise_parameters"]["rotation_deg"]) / 2
    noise = torch.nn.functional.normalize(torch.cat((half.cos(), axis * half.sin()), dim=-1), dim=-1)
    expected = quaternion_product_v1(sensor["fingertip_midpoint_quat"], noise)
    expected[..., [0, 3]] = 0.
    expected *= initial["flip_quats"][None, :, None]
    current = torch.nn.functional.normalize(sensor["noisy_fingertip_quat"], dim=-1)
    previous = torch.cat((initial["_previous_noisy_quat"][None], current[:-1]))
    conjugate = previous.clone()
    conjugate[..., 1:] *= -1
    rotation = quaternion_product_v1(current, conjugate)
    rotation *= torch.where(rotation[..., :1] < 0, -1., 1.)
    angle = 2 * torch.atan2(rotation[..., 1:].norm(dim=-1), rotation[..., 0])
    scale = torch.where(angle.abs() > 1e-6, torch.sin(angle / 2) / angle, .5 - angle.square() / 48)
    velocity = rotation[..., 1:] / scale[..., None] * 120
    velocity[..., :2] = 0.
    return {"noisy_quaternion_from_original_draws": maximum_error(expected, sensor["noisy_fingertip_quat"]) < 2e-6,
            "angular_derivative_120hz": maximum_error(velocity, sensor["ee_angvel_fd"]) < 2e-4}


def verify_rewards_v1(data, report):
    timing = PhysicsTimingV1(report["refinement"])
    discounts = .995 ** torch.arange(450, dtype=torch.float64)
    for i, job in enumerate(report["jobs"]):
        terminal = round(job["elapsed_s"] / timing.physics_dt)
        included = terminal // timing.decimation
        notification = (terminal + timing.decimation - 1) // timing.decimation - 1
        ledger = sum(float((v[:included, i].double() * discounts[:included]).sum()) for v in data["reward_terms"].values())
        credit = float((data["completion_credit"][:, i].double() * discounts).sum())
        ret = float((data["job_reward"][:, i].double() * discounts).sum())
        expected_credit = (35 / 12) * .995 ** included * (1 - .995 ** (450 - included)) / (1 - .995) if job["success"] else 0.
        if (abs(ledger + credit - ret) >= .001 or abs(ret - report["discounted_job_returns"][i]) >= 1e-5
                or abs(credit - expected_credit) >= .002
                or not bool((data["job_reward"][notification + 1:, i] == 0).all())):
            return False
    return (maximum_error(sum(data["reward_terms"].values()), data["raw_reward"]) < 1e-5
            and torch.equal(data["job_reward"], data["upstream_job_reward"] + data["completion_credit"]))


def verify_actions_v1(data, report):
    controllers = [ActorRetryController(row, step_dt=1 / 15,
        position_bounds=report["controller_position_bounds"], seated_height_m=report["controller_seated_height_m"],
        retry=report["mode"] == "retry", settings=RetrySettings(**report["controller_settings"]))
        for row in data["actor_before"][0].tolist()]
    script_steps = {"retry": 450, "prefix_learned": 60, "direct_learned": 0}[report["mode"]]
    for step in range(script_steps):
        decisions = [ctrl.act(row, step * report["controller_step_dt"]) for ctrl, row in
                     zip(controllers, data["actor_before"][step].tolist(), strict=True)]
        expected = torch.tensor([d[0] for d in decisions], dtype=data["requested_action"].dtype)
        if (not torch.equal(expected, data["requested_action"][step])
                or [d[1]["phase"] for d in decisions] != report["script_phases"][step]):
            return False
    if report["mode"] != "retry" and not torch.equal(data["requested_action"][script_steps:], data["mean_action"][script_steps:]):
        return False
    previous = data["initial"]["actions"]
    for requested, applied in zip(data["requested_action"], data["applied_action"], strict=True):
        # Frozen prepare_actions retains finite absorbing requests in EMA history.
        # Physical servo targets are held independently by finished_mask.
        target = torch.where(torch.isfinite(requested).all(-1)[:, None], requested.clamp(-1, 1), previous)
        expected = data["initial"]["ema_factor"] * target + (1 - data["initial"]["ema_factor"]) * previous
        if maximum_error(expected, applied) > 2e-7:
            return False
        previous = applied
    return bool((data["applied_action"].abs() <= 1).all())


def sensor_checks_v1(data, report):
    timing = PhysicsTimingV1(report["refinement"])
    native, sensor, initial = data["native"], data["sensors"], data["sensor_initial"]
    ticks = torch.arange(1, timing.deadline_steps + 1) % timing.refinement == 0
    checks = {
        "sensor_tick_grid": torch.equal(native["sensor_tick"], ticks),
        "sensor_time_120hz": maximum_error(sensor["timestamp"], torch.arange(1, 3601).double() / 120) < 2e-6,
        "native_simulator_time": maximum_error(native["sim_time"], torch.arange(1, timing.deadline_steps + 1).double() * timing.physics_dt) < 2e-6,
        "held_torque_between_native_substeps": timing.refinement == 1 or torch.equal(native["torques"][::2], native["torques"][1::2]),
        "held_gripper_targets_between_native_substeps": timing.refinement == 1 or torch.equal(native["gripper_targets"][::2], native["gripper_targets"][1::2]),
    }
    error = maximum_error(native["contact_force"], native["contact_impulse"] / timing.physics_dt)
    checks["native_impulse_to_force"] = error < 3e-5
    raw = native["raw_wrist"][..., :3].norm(dim=-1)
    checks["raw_wrist_native_evaluator_input"] = maximum_error(raw, data["physics"][..., 2]) < 1e-5
    contacts = native["contact_force"].norm(dim=-1).sum(dim=2)
    checks["contact_native_evaluator_inputs"] = maximum_error(contacts, data["physics"][..., [11, 9, 10]]) < 1e-5
    checks["fixture_contact_positive_control"] = bool((contacts[..., 0] > .1).any())
    checks["bilateral_finger_positive_control"] = bool((contacts[..., 1:] > .01).all(-1).any())
    alpha = report["sensor_noise_parameters"]["force_alpha"]
    previous = initial["force_sensor_world_smooth"]
    smooth_error = 0.
    for step, measured in enumerate(sensor["force_sensor_world_smooth"]):
        expected = alpha * native["raw_wrist"][(step + 1) * timing.refinement - 1] + (1 - alpha) * previous
        smooth_error = max(smooth_error, maximum_error(expected, measured))
        previous = measured
    checks["force_filter_120hz_no_begin_reset"] = smooth_error < 2e-5
    checks["noisy_position_from_original_draws"] = maximum_error(sensor["noisy_fingertip_pos"],
        sensor["fingertip_midpoint_pos"] + data["sensor_draws"][..., :3] * report["sensor_noise_parameters"]["position"]) < 2e-7
    checks["noisy_force_from_original_draws"] = maximum_error(sensor["noisy_force"],
        sensor["force_sensor_smooth"][..., :3] + data["sensor_draws"][..., 7:10] * report["sensor_noise_parameters"]["force"]) < 1e-5
    previous_positions = torch.cat((initial["_previous_noisy_pos"][None], sensor["noisy_fingertip_pos"][:-1]))
    checks["noisy_derivative_120hz"] = maximum_error(sensor["ee_linvel_fd"], (sensor["noisy_fingertip_pos"] - previous_positions) * 120) < 2e-5
    held_filter = torch.cat((initial["force_sensor_smooth"][None], sensor["force_sensor_smooth"]))
    indices = torch.arange(1, timing.deadline_steps + 1) // timing.refinement
    checks["filter_held_on_fine_substeps"] = maximum_error(held_filter[indices, :, :3].norm(dim=-1), data["physics"][..., 3]) < 1e-5
    checks.update(rotation_checks_v1(data, report))
    checks["roll_pitch_velocity_disabled"] = bool((sensor["ee_angvel_fd"][..., :2] == 0).all())
    return checks, {"contact_conversion_max_error_n": error, "force_filter_max_error_n": smooth_error}


def load_run(path):
    manifest = verify_assets(path)
    report = json.loads((path / "validation/report.json").read_text())
    data = torch.load(path / "validation/trajectory.pt", weights_only=True, map_location="cpu")
    return manifest, report, data


def verify_single_v1(path, registration):
    manifest, report, data = load_run(path)
    timing = PhysicsTimingV1(report["refinement"])
    criteria = JobCriteria(physics_dt=timing.physics_dt)
    study = json.loads((ROOT / "configs/study.json").read_text())
    checks = {
        "completed_artifacts": manifest["status"] == report["status"] == "completed" and manifest["returncode"] == 0 and not manifest["timed_out"],
        "registered_before_launch": datetime.fromisoformat(registration["created_at_utc"]) < datetime.fromisoformat(manifest["started_at_utc"]),
        "source_and_registration": manifest["registration_sha256"] == sha256(REGISTRATION)
            and all(manifest["source_hashes"][p] == h for p, h in registration["source_sha256"].items()),
        "all_development_cases": report["seed"] == 10071 and report["cases"] == development_cases(study, 10071)
            and len(report["jobs"]) == len(report["initialization"]) == 28,
        "valid_initialization": all(v["valid"] for v in report["initialization"]),
        "unchanged_initial_state_at_resolution_switch": all(report["preparation_checks"].values()),
        "unchanged_begin_state": all(report["begin_checks"].values()),
        "native_criteria": report["criteria"] == asdict(criteria),
        "native_physics_shape": tuple(data["physics"].shape) == (timing.deadline_steps, 28, 14),
        "sensor_draw_shape": tuple(data["sensor_draws"].shape) == (3600, 28, 10),
        "timing": report["servo_updates"] == report["sensor_updates"] == 3600
            and report["native_physics_steps"] == timing.deadline_steps and report["controller_step_dt"] == 1 / 15,
        "native_contact_conversion": report["contact_force_conversion_dt"] == report["actual_solver_dt"] == timing.physics_dt,
        "native_prefix_conversion": report["prefix_controls"] == 60 and report["prefix_physics_steps"] == timing.handoff_step,
        "gravity": report["held_part_gravity_enabled"] and report["robot_gravity_disabled"]
            and report["world_gravity"][0] == [0., 0., -1.] and abs(report["world_gravity"][1] - 9.81) < 1e-5,
        "finite_arrays": finite_tree(data),
        "unchanged_script_settings": report["controller_settings"] == asdict(RetrySettings(**json.loads((ROOT / "configs/retry_unload_realign_v1.json").read_text()))),
        "script_and_action_ema_replay": verify_actions_v1(data, report),
        "reward_credit_and_return_replay": verify_rewards_v1(data, report),
        "no_forbidden_events": all(not j["forbidden_events"] for j in report["jobs"]),
    }
    replay = replay_physics(data["physics"], criteria, report["jobs"], "diagnosis", report["recovery_witness"])
    checks["native_cpu_evaluator_and_witness_replay"] = replay["exact_cpu_evaluator_parity"] and replay["contact_witness_parity"]
    endpoints = [post_stall_completion(j, w, criteria, handoff_step=timing.handoff_step) for j, w in zip(report["jobs"], report["recovery_witness"], strict=True)]
    checks["post_stall_endpoint_replay"] = endpoints == report["post_stall_endpoint"] and summarize_endpoints(endpoints, report["jobs"]) == report["post_stall_summary"]
    loaded = torch.load(ROOT / registration["checkpoint_path"], map_location="cpu", weights_only=True)
    spec = loaded["spec"]
    model = StudyPolicy(spec["actor_size"], spec["critic_size"], spec["action_size"], spec["widths"])
    model.load_state_dict(loaded["model"], strict=True)
    model.eval()
    with torch.no_grad():
        means = model.distribution(model.actor_norm(data["actor_before"])).mean
    mean_error = maximum_error(means, data["mean_action"])
    checks["frozen_complete_budget_checkpoint"] = (sha256(ROOT / registration["checkpoint_path"]) == registration["checkpoint_sha256"]
        == report["checkpoint_sha256"] == manifest["input_checkpoint"]["sha256"]
        and loaded["status"] == "budget_complete" and loaded["completed_cohorts"] == 22
        and loaded["optimizer_version"] == "value_unclipped_v4" and mean_error <= 2e-5)
    terminal_steps = [round(j["elapsed_s"] / timing.physics_dt) for j in report["jobs"]]
    expected_cost = physics_cost_v1(initialization_steps=67, job_steps=timing.deadline_steps, num_envs=28,
        active_physics=sum(terminal_steps), completed_controls=450,
        active_controls=sum(math.ceil(s / timing.decimation) for s in terminal_steps), refinement=timing.refinement)
    checks["all_native_work_accounted"] = report["cost"] == expected_cost
    sensor_checks, diagnostics = sensor_checks_v1(data, report)
    checks.update(sensor_checks)
    compatibility = None
    if timing.refinement == 1 and report["mode"] in ("prefix_learned", "retry"):
        old = registration["compatibility_references"][report["mode"]]
        old_path = ROOT / old["path"]
        old_manifest = verify_assets(old_path)
        prior = torch.load(old_path / "diagnosis/trajectory.pt", weights_only=True, map_location="cpu")
        old_report = json.loads((old_path / "diagnosis/report.json").read_text())
        parity = {k: torch.equal(v, data[k]) for k, v in prior.items() if isinstance(v, torch.Tensor)}
        for name in ("initial", "reward_terms"):
            parity[name] = all(torch.equal(v, data[name][k]) for k, v in prior[name].items())
        parity["original_manifest"] = sha256(old_path / "manifest.json") == old["manifest_sha256"] and old_manifest["status"] == "completed"
        parity["native_jobs"] = report["jobs"] == old_report["jobs"]
        parity["witness"] = report["recovery_witness"] == old_report["recovery_witness"]
        parity["post_stall"] = report["post_stall_endpoint"] == old_report["post_stall_endpoint"]
        checks.update({"confirmed_120hz_" + k: v for k, v in parity.items()})
        compatibility = parity
    return {"status": "verified" if all(checks.values()) else "check_failed", "checks": checks,
            "failed_checks": [k for k, v in checks.items() if not v], "mode": report["mode"],
            "refinement": report["refinement"], "manifest_sha256": sha256(path / "manifest.json"),
            "whole_job": whole_job_summary_v1(report["jobs"]), "post_stall": report["post_stall_summary"],
            "cost": report["cost"], "cpu_mean_max_error": mean_error,
            "sensor_diagnostics": diagnostics, "compatibility": compatibility}


def summarize_pairs_v1(root, registration):
    results, traces, reports, checks = {}, {}, {}, {}
    for row in registration["run_order"]:
        label = row["label"]
        result = verify_single_v1(root / label, registration)
        results[label] = result
        _, reports[label], traces[label] = load_run(root / label)
        checks[label] = result["status"] == "verified"
    reference = traces["120_prefix_learned"]
    initial_differences = {}
    for label, data in traces.items():
        diffs = {group + "/" + k: maximum_error(v, data[group][k]) for group in ("initial", "sensor_initial", "physical_initial") for k, v in reference[group].items()}
        initial_differences[label] = diffs
        checks[label + "_identical_initialized_state"] = not any(diffs.values())
        for key in ("sensor_draws", "cuda_rng_before_steps", "dead_zone"):
            checks[label + "_paired_" + key] = torch.equal(reference[key], data[key])
    pairing = {}
    for hz in (120, 240):
        learned, retry = traces[f"{hz}_prefix_learned"], traces[f"{hz}_retry"]
        end = 480 * (hz // 120)
        for key in ("actor_before", "critic_before", "requested_action", "applied_action", "job_reward"):
            checks[f"{hz}_prefix_{key}"] = torch.equal(learned[key][:60], retry[key][:60])
        checks[f"{hz}_native_prefix"] = torch.equal(learned["physics"][:end], retry["physics"][:end])
        left, right = reports[f"{hz}_prefix_learned"], reports[f"{hz}_retry"]
        checks[f"{hz}_same_stalled_cohort"] = [r["eligible_at_handoff"] for r in left["post_stall_endpoint"]] == [r["eligible_at_handoff"] for r in right["post_stall_endpoint"]]
        pairing[str(hz)] = {}
        for assay, learned_mode in (("direct", "direct_learned"), ("prefix", "prefix_learned")):
            lj, rj = reports[f"{hz}_{learned_mode}"]["jobs"], right["jobs"]
            pairing[str(hz)][assay] = {
                "requests": 28, "both_complete": sum(a["success"] and b["success"] for a, b in zip(lj, rj, strict=True)),
                "learned_only": sum(a["success"] and not b["success"] for a, b in zip(lj, rj, strict=True)),
                "retry_only": sum(not a["success"] and b["success"] for a, b in zip(lj, rj, strict=True)),
                "neither_complete": sum(not a["success"] and not b["success"] for a, b in zip(lj, rj, strict=True))}
    sensitivity = {}
    for mode in ("prefix_learned", "retry", "direct_learned"):
        left, right = reports[f"120_{mode}"], reports[f"240_{mode}"]
        changes = []
        for case, a, b, ae, be in zip(left["cases"], left["jobs"], right["jobs"], left["post_stall_endpoint"], right["post_stall_endpoint"], strict=True):
            changes.append({"case_id": case["case_id"], "bin_id": case["bin_id"],
                "outcome_120": a["outcome"], "outcome_240": b["outcome"], "time_120_s": a["elapsed_s"], "time_240_s": b["elapsed_s"],
                "peak_force_120_n": a["peak_raw_wrist_force_n"], "peak_force_240_n": b["peak_raw_wrist_force_n"],
                "eligible_120": ae["eligible_at_handoff"], "eligible_240": be["eligible_at_handoff"],
                "post_stall_120": ae["post_stall_completion"], "post_stall_240": be["post_stall_completion"],
                "withdrawal_120": ae["original_withdrawal_recovery"], "withdrawal_240": be["original_withdrawal_recovery"]})
        sensitivity[mode] = {
            "completion_delta_240_minus_120": sum(int(b["success"]) - int(a["success"]) for a, b in zip(left["jobs"], right["jobs"], strict=True)),
            "outcome_changed": sum(r["outcome_120"] != r["outcome_240"] for r in changes),
            "eligibility_changed": sum(r["eligible_120"] != r["eligible_240"] for r in changes),
            "common_eligible": sum(r["eligible_120"] and r["eligible_240"] for r in changes),
            "post_stall_120_on_common": sum(r["eligible_120"] and r["eligible_240"] and r["post_stall_120"] for r in changes),
            "post_stall_240_on_common": sum(r["eligible_120"] and r["eligible_240"] and r["post_stall_240"] for r in changes), "cases": changes}
    charged = sum(r["cost"]["charged_reference_transitions"] for r in results.values())
    checks["registered_native_cost"] = charged == registration["expected_charged_reference_transitions"]
    material = any(abs(v["completion_delta_240_minus_120"]) >= 2 or v["outcome_changed"] >= 3 for v in sensitivity.values())
    return {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed", "research_result": False,
            "checks": checks, "failed_checks": [k for k, v in checks.items() if not v], "runs": results,
            "pairing": pairing, "sensitivity": sensitivity, "initial_maximum_absolute_differences": initial_differences,
            "charged_reference_transitions": charged,
            "charged_control_equivalent_transitions": sum(r["cost"]["charged_control_equivalent_transitions"] for r in results.values()),
            "decision": {"material_sensitivity": material, "training_screen_ready": False, "original_competence_gate_passed": False,
                "reason": "Two resolutions establish sensitivity only. Material changes require a bounded physics investigation; this worker does not authorize training."},
            "scope_and_limitations": registration["scope_and_limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing verification records")
    torch.set_num_threads(1)
    registration = json.loads(REGISTRATION.read_text())
    result = (verify_single_v1 if args.single else summarize_pairs_v1)(args.run.resolve(), registration)
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}), flush=True)
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
