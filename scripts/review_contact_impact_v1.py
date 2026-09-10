"""CPU verification and descriptive analysis of the registered native-impact prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.evaluation import JobCriteria  # noqa: E402
from assembly_recovery.faults import development_cases  # noqa: E402
from assembly_recovery.protocol import LAB_COMMIT, sha256, write_json  # noqa: E402
from assembly_recovery.retry_controller import ActorRetryController, RetrySettings  # noqa: E402
from scripts.probe_training import replay_physics  # noqa: E402
from scripts.verify_learned_physics_v3 import finite_tree, maximum_error, rotation_checks_v1  # noqa: E402


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(equal_tree(v, right[k]) for k, v in left.items())
    return left == right


def _prefix_parity(data, refinement):
    if refinement not in (1, 2):
        return {}, None
    reference = ROOT / "artifacts/assembly/learned_physics_v3_r01" / f"{120 * refinement}_prefix_learned"
    saved = torch.load(reference / "validation/trajectory.pt", map_location="cpu", weights_only=True)
    parity = {}
    for key, value in saved.items():
        if isinstance(value, torch.Tensor):
            count = 480 * refinement if key == "physics" else 480 if key == "sensor_draws" else 60
            parity[key] = torch.equal(value[:count], data[key])
        elif key in ("initial", "sensor_initial", "physical_initial"):
            parity[key] = equal_tree(value, data[key])
        elif key in ("native", "sensors", "reward_terms"):
            count = 480 * refinement if key == "native" else 480 if key == "sensors" else 60
            for name, tensor in value.items():
                parity[f"{key}.{name}"] = torch.equal(tensor[:count], data[key][name])
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    parity["reference_process_complete"] = reference_manifest["status"] == "completed" and reference_manifest["returncode"] == 0
    recorded = {Path(item["path"]).name: item["sha256"] for item in reference_manifest["artifacts"]}
    parity["reference_trajectory_hash"] = sha256(reference / "validation/trajectory.pt") == recorded["trajectory.pt"]
    return parity, str(reference)


def verify_single(run: Path):
    """Return explicit failure flags; safe for the launcher's completed-run check."""
    try:
        return _verify_single(Path(run))
    except Exception as exc:
        return {"status": "verification_error", "checks": {"verifier_completed": False},
                "failed_checks": ["verifier_completed"], "error": f"{type(exc).__name__}: {exc}"}


def _verify_single(run):
    torch.set_num_threads(1)
    manifest = json.loads((run / "manifest.json").read_text())
    report = json.loads((run / "validation/report.json").read_text())
    data = torch.load(run / "validation/trajectory.pt", map_location="cpu", weights_only=True)
    r = report["refinement"]
    if type(r) is not int or r not in (1, 2, 4, 8):
        raise ValueError("Unregistered native refinement")
    dt, steps = 1 / (120 * r), 480 * r
    criteria = JobCriteria(physics_dt=dt)
    checks = {
        "process_completed": manifest["status"] == report["status"] == "completed" and manifest["returncode"] == 0
            and not manifest["timed_out"] and not manifest.get("resource_guard_stop"),
        "native_criteria_unchanged": report["criteria"] == asdict(criteria),
        "all_28_requests": len(report["jobs"]) == len(report["initialization"]) == 28,
        "initializations_valid": all(v["valid"] for v in report["initialization"]),
        "preparation_preserves_state_rng": all(report["preparation_checks"].values()),
        "begin_preserves_state_rng": all(report["begin_checks"].values()),
        "prefix_controller_label": report["controller"] == "frozen_scripted_prefix" and report["mode"] == "frozen_script_prefix",
        "physical_prefix_shape": tuple(data["physics"].shape) == (steps, 28, 14),
        "sensor_draws_shape": tuple(data["sensor_draws"].shape) == (480, 28, 10),
        "control_shape": tuple(data["requested_action"].shape) == (60, 28, 7),
        "timing_counts": report["native_physics_steps"] == steps and report["servo_updates"] == report["sensor_updates"] == 480,
        "control_and_prefix_duration": report["controller_step_dt"] == 1 / 15 and report["prefix_controls"] == 60
            and report["prefix_physics_steps"] == steps and report["expected_total_simulated_seconds"] == 4.,
        "gravity": report["held_part_gravity_enabled"] and report["robot_gravity_disabled"]
            and report["world_gravity"][0] == [0., 0., -1.] and abs(report["world_gravity"][1] - 9.81) < 1e-5,
        "finite_arrays": finite_tree(data),
        "no_forbidden_events": all(not job["forbidden_events"] for job in report["jobs"]),
    }
    with zipfile.ZipFile(run / "source.zip") as archive:
        checks["source_archive_hash"] = sha256(run / "source.zip") == manifest["source_archive_sha256"]
        checks["all_archived_source_hashes"] = all(hashlib.sha256(archive.read(name)).hexdigest() == digest
            for name, digest in manifest["source_hashes"].items())
        study = json.loads(archive.read("configs/study.json"))
        frozen = json.loads(archive.read("configs/learned_physics_validation_v3.json"))
        settings = json.loads(archive.read("configs/retry_unload_realign_v1.json"))
        registration_bytes = archive.read("configs/contact_impact_registration_v1.json")
        checks["archived_registration"] = hashlib.sha256(registration_bytes).hexdigest() == manifest["registration_sha256"]
        checks["registration_values"] = json.loads(registration_bytes) == manifest["registration"]
        checks["executing_diagnostic_sources_match_archive"] = all(sha256(ROOT / name) == manifest["source_hashes"][name] for name in (
            "src/assembly_recovery/contact_impact_env_v1.py", "scripts/probe_contact_impact_v1.py",
            "scripts/review_contact_impact_v1.py"))
    checks["fixed_development_cases"] = report["seed"] == 10071 and report["cases"] == development_cases(study, 10071)
    checks["pinned_upstream_revision"] = manifest["upstream_commit"] == LAB_COMMIT
    checks["installed_source_hashes"] = all(sha256(ROOT / name) == digest for name, digest in manifest["installed_source_sha256"].items())
    checks["frozen_checkpoint_hash"] = sha256(ROOT / frozen["checkpoint_path"]) == frozen["checkpoint_sha256"] == report["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    checks["frozen_script_settings"] = settings == report["controller_settings"]
    native, sensors = data["native"], data["sensors"]
    native_steps = torch.arange(1, steps + 1)
    ticks = native_steps % r == 0
    checks["actual_native_dt"] = bool((native["integration_dt"] == dt).all()) and report["actual_solver_dt"] == dt
    checks["native_callback_time"] = maximum_error(native["sim_time"], native_steps.double() * dt) < 2e-6
    checks["contact_conversion_dt"] = report["contact_force_conversion_dt"] == dt and report["scene_metadata_dt"] == 1 / 120
    checks["sensor_grid"] = torch.equal(native["sensor_tick"], ticks)
    checks["sensor_times"] = maximum_error(sensors["timestamp"], torch.arange(1, 481).double() / 120) < 2e-6
    checks["held_native_torque"] = torch.equal(native["torques"], native["torques"][::r].repeat_interleave(r, 0))
    checks["held_native_gripper_targets"] = torch.equal(native["gripper_targets"], native["gripper_targets"][::r].repeat_interleave(r, 0))
    conversion_error = maximum_error(native["contact_force"], native["contact_impulse"] / dt)
    checks["native_impulse_to_force"] = conversion_error < 3e-5
    raw = native["raw_wrist"][..., :3].norm(dim=-1)
    contacts = native["contact_force"].norm(dim=-1).sum(dim=2)
    checks["raw_wrist_evaluator_input"] = maximum_error(raw, data["physics"][..., 2]) < 1e-5
    checks["contact_evaluator_input"] = maximum_error(contacts, data["physics"][..., [11, 9, 10]]) < 1e-5
    checks["fixture_positive_control"] = bool((contacts[..., 0] > .1).any())
    checks["bilateral_finger_positive_control"] = bool((contacts[..., 1:] > .01).all(-1).any())
    terminal_steps = [round(job["elapsed_s"] / dt) for job in report["jobs"]]
    force_ok, active_ok = True, True
    for i, (job, end) in enumerate(zip(report["jobs"], terminal_steps, strict=True)):
        force_ok &= 0 < end <= steps and bool((raw[:end - 1, i] <= 20).all())
        force_ok &= bool(raw[end - 1, i] > 20) if job["outcome"] == "force_abort" else bool((raw[:end, i] <= 20).all())
        expected_active = native_steps < end if job["outcome"] != "probe_end" else torch.ones(steps, dtype=torch.bool)
        active_ok &= torch.equal(native["active_after"][:, i], expected_active)
    checks["first_raw_20n_crossing_ends_active_job"] = bool(force_ok)
    checks["native_active_and_absorbing_masks"] = bool(active_ok)
    replay = replay_physics(data["physics"], criteria, report["jobs"], "diagnosis", report["recovery_witness"])
    checks["independent_cpu_job_and_witness_replay"] = replay["exact_cpu_evaluator_parity"] and replay["contact_witness_parity"]
    c = report["cost"]
    checks["full_native_cost"] = (c["initialization_physics_env_steps"] == 67 * 28 and c["rollout_physics_env_steps"] == steps * 28
        and c["charged_reference_transitions"] == (67 + steps) * 28 / 8
        and c["charged_control_equivalent_transitions"] == 1914.5 and c["rollout_control_transitions"] == 1680
        and c["active_physics_env_steps"] == sum(terminal_steps)
        and c["active_control_transitions"] == sum(math.ceil(end / (8 * r)) for end in terminal_steps)
        and c["partial_control_physics_env_steps"] == 0)
    previous = data["sensor_initial"]["force_sensor_world_smooth"]
    smooth_error = 0.
    for step, actual in enumerate(sensors["force_sensor_world_smooth"]):
        expected = report["sensor_noise_parameters"]["force_alpha"] * native["raw_wrist"][(step + 1) * r - 1] + (1 - report["sensor_noise_parameters"]["force_alpha"]) * previous
        smooth_error = max(smooth_error, maximum_error(expected, actual))
        previous = actual
    checks["120hz_force_filter"] = smooth_error < 2e-5
    held_filter = torch.cat((data["sensor_initial"]["force_sensor_smooth"][None], sensors["force_sensor_smooth"]))
    checks["filter_held_between_ticks"] = maximum_error(held_filter[native_steps // r, :, :3].norm(dim=-1), data["physics"][..., 3]) < 1e-5
    noise = report["sensor_noise_parameters"]
    checks["noisy_position"] = maximum_error(sensors["noisy_fingertip_pos"], sensors["fingertip_midpoint_pos"] + data["sensor_draws"][..., :3] * noise["position"]) < 2e-7
    checks["noisy_force"] = maximum_error(sensors["noisy_force"], sensors["force_sensor_smooth"][..., :3] + data["sensor_draws"][..., 7:10] * noise["force"]) < 1e-5
    previous_pos = torch.cat((data["sensor_initial"]["_previous_noisy_pos"][None], sensors["noisy_fingertip_pos"][:-1]))
    checks["sensor_velocity_120hz"] = maximum_error(sensors["ee_linvel_fd"], (sensors["noisy_fingertip_pos"] - previous_pos) * 120) < 2e-5
    checks.update(rotation_checks_v1(data, report))
    controllers = [ActorRetryController(row, step_dt=1 / 15, position_bounds=report["controller_position_bounds"],
        seated_height_m=report["controller_seated_height_m"], retry=False, settings=RetrySettings(**settings))
        for row in data["actor_before"][0].tolist()]
    action_ok, previous = True, data["initial"]["actions"]
    for step in range(60):
        decisions = [ctrl.act(row, step * report["controller_step_dt"]) for ctrl, row in zip(controllers, data["actor_before"][step].tolist(), strict=True)]
        requested = torch.tensor([row[0] for row in decisions], dtype=data["requested_action"].dtype)
        action_ok &= torch.equal(requested, data["requested_action"][step]) and [row[1]["phase"] for row in decisions] == report["script_phases"][step]
        applied = data["initial"]["ema_factor"] * requested.clamp(-1, 1) + (1 - data["initial"]["ema_factor"]) * previous
        action_ok &= maximum_error(applied, data["applied_action"][step]) <= 2e-7
        previous = data["applied_action"][step]
    checks["frozen_script_and_action_ema"] = bool(action_ok)
    parity, reference = _prefix_parity(data, r)
    checks.update({f"saved_{120 * r}hz_prefix_{key}": value for key, value in parity.items()})
    return {"status": "verified" if all(checks.values()) else "check_failed", "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed], "refinement": r,
            "physics_hz": 120 * r, "cost": c, "prefix_summary": report["prefix_summary"],
            "reference": reference, "diagnostics": {"contact_conversion_max_error_n": conversion_error,
            "force_filter_max_error_n": smooth_error},
            "trajectory_sha256": sha256(run / "validation/trajectory.pt")}


def relative_difference(a, b):
    """Symmetric relative difference; both-zero physical quantities agree."""
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def load_trace(run):
    return (json.loads((run / "validation/report.json").read_text()),
            torch.load(run / "validation/trajectory.pt", weights_only=True, map_location="cpu"))


def _window_metrics(native, case, begin, end, dt):
    if end <= begin:
        return {"native_steps": 0, "duration_s": 0., "wrist_peak_n": None, "wrist_impulse_magnitude_ns": 0.,
                "fixture_peak_n": None, "fixture_impulse_magnitude_ns": 0., "fixture_contact_duration_s": 0.}
    wrist = native["raw_wrist"][begin:end, case, :3].double().norm(dim=-1)
    fixture = native["contact_force"][begin:end, case, 0, 0].double().norm(dim=-1)
    impulses = native["contact_impulse"][begin:end, case, 0, 0].double()
    return {"native_steps": end - begin, "duration_s": (end - begin) * dt,
            "wrist_peak_n": float(wrist.max()), "wrist_impulse_magnitude_ns": float(wrist.sum()) * dt,
            "fixture_peak_n": float(fixture.max()), "fixture_impulse_magnitude_ns": float(impulses.norm(dim=-1).sum()),
            "fixture_vector_impulse_ns": impulses.sum(0).tolist(),
            "fixture_contact_duration_s": int((fixture > .1).sum()) * dt,
            "wrist_over_20_duration_s": int((wrist > 20).sum()) * dt}


def describe_run(report, data):
    r, native = report["refinement"], data["native"]
    dt, steps = 1 / (120 * r), 480 * r
    rows = []
    for i, (case, job) in enumerate(zip(report["cases"], report["jobs"], strict=True)):
        end = round(job["elapsed_s"] / dt)
        fixture = native["contact_force"][:end, i, 0, 0].double().norm(dim=-1)
        contact_indices = torch.nonzero(fixture > .1).flatten()
        onset = int(contact_indices[0]) if len(contact_indices) else None
        active = _window_metrics(native, i, 0, end, dt)
        absorbing = _window_metrics(native, i, end, steps, dt)
        # Every complete servo block is described; blocks crossing a native
        # terminal are marked, never silently treated as active exposure.
        block_rows = []
        wrist = native["raw_wrist"][:, i, :3].double().norm(dim=-1)
        impulse = native["contact_impulse"][:, i, 0, 0].double().norm(dim=-1)
        for block in range(math.ceil(end / r)):
            a, b = block * r, min((block + 1) * r, steps)
            if bool((fixture[a:min(b, end)] > .1).any()) or bool((wrist[a:b] > 20).any()):
                block_rows.append({"servo_interval_index": block, "start_s": a * dt, "end_s": b * dt,
                    "native_wrist_peak_n": float(wrist[a:b].max()),
                    "wrist_impulse_magnitude_ns": float(wrist[a:b].sum()) * dt,
                    "wrist_interval_average_n": float(wrist[a:b].mean()),
                    "fixture_impulse_magnitude_ns": float(impulse[a:b].sum()),
                    "fixture_interval_average_n": float(impulse[a:b].sum()) * 120,
                    "end_sample_wrist_n": float(wrist[b - 1]),
                    "contains_post_terminal_steps": b > end, "active_native_steps": max(0, min(b, end) - a)})
        position = native["gripper_joint_pos"][:end, i].double()
        velocity = native["gripper_joint_vel"][:end, i].double()
        rows.append({"index": i, "case_id": case["case_id"], "bin_id": case["bin_id"], "outcome": job["outcome"],
            "active_end_s": job["elapsed_s"], "censored_prefix": job["outcome"] == "probe_end",
            "contact_onset_s": (onset + 1) * dt if onset is not None else None,
            "contact_duration_is_thresholded_at_n": .1,
            "active": active, "absorbing": absorbing,
            "active_inter_sensor_peak_n": float(wrist[:end][~native["sensor_tick"][:end]].max()) if r > 1 and bool((~native["sensor_tick"][:end]).any()) else None,
            "active_sensor_tick_peak_n": float(wrist[:end][native["sensor_tick"][:end]].max()) if bool(native["sensor_tick"][:end].any()) else None,
            "gripper_position_span_m": (position.max(0).values - position.min(0).values).tolist(),
            "gripper_peak_speed_m_s": velocity.abs().max(0).values.tolist(),
            "gripper_final_position_m": position[-1].tolist(),
            "held_peak_speed_m_s": float(native["held_linvel"][:end, i].double().norm(dim=-1).max()),
            "tool_peak_speed_m_s": float(native["tool_linvel"][:end, i].double().norm(dim=-1).max()),
            "contact_servo_intervals": block_rows,
        })
    return {"physics_hz": 120 * r, "refinement": r, "prefix_summary": report["prefix_summary"], "cost": report["cost"], "cases": rows}


def compare_pair(coarse, fine, coarse_trace=None, fine_trace=None):
    rows = []
    for a, b in zip(coarse["cases"], fine["cases"], strict=True):
        if a["case_id"] != b["case_id"]:
            raise ValueError("Unpaired case order")
        row = {"case_id": a["case_id"], "outcomes": [a["outcome"], b["outcome"]],
            "outcome_changed": a["outcome"] != b["outcome"],
            "active_end_s": [a["active_end_s"], b["active_end_s"]],
            "exposure_changed": a["active_end_s"] != b["active_end_s"],
            "peak_wrist_relative_difference": relative_difference(a["active"]["wrist_peak_n"], b["active"]["wrist_peak_n"]),
            "fixture_impulse_relative_difference": relative_difference(a["active"]["fixture_impulse_magnitude_ns"], b["active"]["fixture_impulse_magnitude_ns"])}
        row["load_outlier"] = row["peak_wrist_relative_difference"] > .2 or row["fixture_impulse_relative_difference"] > .2
        if coarse_trace is not None and fine_trace is not None:
            end = math.floor(min(a["active_end_s"], b["active_end_s"]) * coarse["physics_hz"] + 1e-8) / coarse["physics_hz"]
            i = a["index"]
            shared = []
            for run, trace in ((coarse, coarse_trace), (fine, fine_trace)):
                dt = 1 / run["physics_hz"]
                shared.append(_window_metrics(trace["native"], i, 0, round(end / dt), dt))
            row["common_preterminal_duration_s"] = end
            row["common_preterminal"] = shared
            row["common_preterminal_peak_relative_difference"] = relative_difference(shared[0]["wrist_peak_n"], shared[1]["wrist_peak_n"]) if end > 0 else None
            row["common_preterminal_impulse_relative_difference"] = relative_difference(shared[0]["fixture_impulse_magnitude_ns"], shared[1]["fixture_impulse_magnitude_ns"])
        rows.append(row)
    abort_change = abs(sum(c["outcome"] == "force_abort" for c in coarse["cases"]) - sum(c["outcome"] == "force_abort" for c in fine["cases"]))
    changed = sum(row["outcome_changed"] for row in rows)
    wrist_median = median(row["peak_wrist_relative_difference"] for row in rows)
    impulse_median = median(row["fixture_impulse_relative_difference"] for row in rows)
    margins = {"abort_count_difference_at_most_one": abort_change <= 1,
               "individual_outcome_changes_at_most_two": changed <= 2,
               "median_peak_wrist_relative_difference_at_most_20pct": wrist_median <= .2,
               "median_fixture_impulse_relative_difference_at_most_20pct": impulse_median <= .2}
    return {"physics_hz": [coarse["physics_hz"], fine["physics_hz"]], "margins": margins,
            "material_margins_pass": all(margins.values()), "abort_count_difference": abort_change,
            "individual_outcome_changes": changed, "median_peak_wrist_relative_difference": wrist_median,
            "median_fixture_impulse_relative_difference": impulse_median,
            "exposure_changed_cases": [row["case_id"] for row in rows if row["exposure_changed"]],
            "outliers": [row for row in rows if row["load_outlier"] or row["outcome_changed"]],
            "all_cases": rows,
            "interpretation": "Registered margins use each active prefix; these exposures can differ after abort. Common-preterminal and servo-interval diagnostics explain, but do not replace, the registered gate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing diagnostic evidence; choose a unique output")
    registration = json.loads((ROOT / "configs/contact_impact_registration_v1.json").read_text())
    verifications, summaries, traces = [], [], {}
    for run in args.runs:
        verification = verify_single(run)
        verifications.append(verification)
        if not all(verification["checks"].values()):
            continue
        report, trace = load_trace(run)
        summary = describe_run(report, trace)
        summaries.append(summary)
        traces[summary["physics_hz"]] = trace
    summaries.sort(key=lambda row: row["physics_hz"])
    if len({row["physics_hz"] for row in summaries}) != len(summaries):
        raise ValueError("Duplicate resolutions cannot be pooled")
    pairs = [compare_pair(a, b, traces[a["physics_hz"]], traces[b["physics_hz"]]) for a, b in zip(summaries, summaries[1:], strict=False)]
    initialization_checks = {}
    if summaries:
        base = traces[summaries[0]["physics_hz"]]
        for summary in summaries[1:]:
            other = traces[summary["physics_hz"]]
            for key in ("initial", "sensor_initial", "physical_initial", "sensor_draws", "cuda_rng_before_steps", "dead_zone"):
                initialization_checks[f"{summary['physics_hz']}hz_{key}"] = equal_tree(base[key], other[key])
    verification_pass = len(summaries) == len(args.runs) and all(all(v["checks"].values()) for v in verifications) and all(initialization_checks.values())
    pair_240_480 = next((p for p in pairs if p["physics_hz"] == [240, 480]), None)
    finest = pairs[-1] if pairs else None
    requires_960 = bool(verification_pass and pair_240_480 and not pair_240_480["material_margins_pass"] and 960 not in traces)
    result = {"schema": 1, "status": "verified_diagnostic" if verification_pass else "check_failed",
        "research_result": False, "registration_sha256": sha256(ROOT / "configs/contact_impact_registration_v1.json"),
        "registered_acceptance": registration["acceptance"], "verification": verifications,
        "cross_resolution_initialization_rng_checks": initialization_checks, "runs": summaries, "pairs": pairs,
        "decision": {"requires_960": requires_960,
            "finest_prefix_pair_pass": bool(verification_pass and finest and finest["material_margins_pass"]),
            "finest_pair_hz": finest["physics_hz"] if finest else None, "training_authorized_by_prefix": False},
        "relative_difference_definition": "abs(a-b)/max(abs(a),abs(b),1e-12); per-case active prefixes, median over all28 including zero-contact cases.",
        "scope_and_limitations": registration["scope_and_limitations"] + [
            "Contact matrices contain normal world-frame forces; wrist wrench is in child joint frame. Wrist impulse magnitudes and contact normal impulses are distinct quantities, not a checked momentum balance.",
            "Active-window native peaks are stopped at20N crossing. No uncensored peak convergence or force calibration is established by the registered descriptive gate.",
            "Servo intervals containing post-terminal samples are explicitly marked; commanded torque remains held until the next external servo tick.",
            "No outliers are removed. Different active exposure and zero-contact cases remain in registered medians and are reported individually.",
        ]}
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "decision": result["decision"],
        "failed_checks": [v.get("failed_checks", []) for v in verifications]}))
    return 0 if verification_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

