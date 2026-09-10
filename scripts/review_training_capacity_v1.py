"""Independently verify capacity artifacts and make a bounded environment-count decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.protocol import sha256  # noqa: E402
from scripts.profile_training_capacity_v1 import finite_tensor_tree, reloaded_actions_match  # noqa: E402
from scripts.run_training_capacity_v1 import report_checks, resource_summary, specification_for  # noqa: E402


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None


def adoption_decision(baseline, candidate):
    """Both useful samples and all charged work must improve by at least 10%."""
    base_rate = baseline["throughput"]["startup_inclusive"]
    candidate_rate = candidate["throughput"]["startup_inclusive"]
    active_gain = ratio(candidate_rate["active_samples_per_s"], base_rate["active_samples_per_s"])
    charged_gain = ratio(candidate_rate["charged_transitions_per_s"], base_rate["charged_transitions_per_s"])
    resources = candidate["resources"]
    checks = {
        "both_runs_independently_verified": baseline["status"] == candidate["status"] == "verified",
        "active_throughput_gain_at_least_10_percent": active_gain is not None and active_gain >= 1.1,
        "charged_throughput_gain_at_least_10_percent": charged_gain is not None and charged_gain >= 1.1,
        "candidate_no_resource_stop": candidate["resource_guard_stop"] is None,
        "candidate_gpu_below_guard": resources["peak_used_mib"] is not None and resources["peak_used_mib"] < 10240,
        "candidate_ram_above_reserve": resources["minimum_free_ram_mib"] is not None and resources["minimum_free_ram_mib"] > 4096,
        "candidate_no_invalid_jobs": candidate["invalid_initializations"] == 0 and candidate["invalid_jobs"] == 0,
    }
    return {"selected_num_envs": 2048 if all(checks.values()) else 1024,
            "status": "2048_earns_bounded_capacity_adoption" if all(checks.values()) else "retain_1024",
            "active_throughput_ratio_2048_over_1024": active_gain,
            "charged_throughput_ratio_2048_over_1024": charged_gain,
            "checks": checks,
            "primary_timing_denominator": "Measured launcher wall time, including startup, initialization, rollout, PPO and checkpoint/diagnostic export and reload; final launcher artifact hashing is outside its recorded timer.",
            "scope": "Provisional early-training engineering capacity choice. Two unequal, unpaired work budgets do not compare learned policy quality or establish sustained multi-hour stability."}


def verify_run(manifest_path):
    import torch

    from assembly_recovery.study_ppo import StudyPolicy

    manifest = json.loads(manifest_path.read_text())
    directory = manifest_path.parent
    prelaunch_path = directory / "prelaunch.json"
    prelaunch = json.loads(prelaunch_path.read_text())
    report_path = directory / "profile/report.json"
    report = json.loads(report_path.read_text())
    archive_path = directory / "source.zip"
    with zipfile.ZipFile(archive_path) as archive:
        registration = json.loads(archive.read("configs/training_capacity_v1.json"))
        archive_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    specification = specification_for(registration, manifest["specification"]["num_envs"])
    worker_dir = directory / "profile"
    checks = report_checks(report, specification, worker_dir)
    checks.update({
        "launcher_completed_without_failure": manifest.get("status") == "completed" and manifest.get("returncode") == 0
            and manifest.get("timed_out") is False and manifest.get("resource_guard_stop") is None,
        "all_launcher_checks_passed": bool(manifest.get("checks")) and all(manifest["checks"].values()),
        "prelaunch_hash_matches": sha256(prelaunch_path) == manifest.get("prelaunch_sha256") == report.get("manifest_sha256"),
        "exact_archived_source_matches_prelaunch": archive_hashes == prelaunch.get("source_hashes") == manifest.get("source_hashes"),
        "required_snapshot_files_present": all(name in archive_hashes for name in (
            "scripts/profile_training_capacity_v1.py", "scripts/run_training_capacity_v1.py", "configs/training_capacity_v1.json",
            "src/assembly_recovery/training_env.py", "src/assembly_recovery/unclipped_ppo.py", "configs/study.json")),
        "registration_matches_capture": archive_hashes.get("configs/training_capacity_v1.json") == manifest.get("registration_sha256"),
        "protocol_matches_capture": archive_hashes.get("configs/protocol_v4.json") == manifest.get("protocol_sha256"),
        "study_matches_capture": archive_hashes.get("configs/study.json") == manifest.get("study_sha256"),
        "no_input_checkpoint": manifest.get("input_checkpoint") is None,
        "specification_matches_registration": specification == manifest.get("specification") == report.get("specification"),
    })
    artifacts = manifest.get("artifacts", [])
    checks["all_manifest_artifact_hashes_match"] = bool(artifacts) and all(
        (ROOT / item["path"]).is_file() and (ROOT / item["path"]).stat().st_size == item["bytes"]
        and sha256(ROOT / item["path"]) == item["sha256"] for item in artifacts)
    required = {report_path.resolve(), archive_path.resolve(), prelaunch_path.resolve()}
    declared = {(ROOT / item["path"]).resolve() for item in artifacts}
    checks["report_source_and_inputs_in_manifest"] = required <= declared
    checkpoints = []
    for index, cohort in enumerate(report["cohorts"]):
        checkpoint_path = worker_dir / f"benchmark_cohort_{index + 1}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        spec = checkpoint["spec"]
        model = StudyPolicy(spec["actor_size"], spec["critic_size"], spec["action_size"], spec["widths"])
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=registration["learning_rate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        observation = {"policy": torch.zeros(2, spec["actor_size"]), "critic": torch.zeros(2, spec["critic_size"])}
        steps = [float(state["step"]) for state in optimizer.state.values()]
        expected_steps = sum(u.get("optimizer_steps", 0) for u in report["optimizer_updates"] if u["cohort"] <= index)
        checkpoint_checks = {
            "finite_model_and_optimizer": finite_tensor_tree(checkpoint),
            "capacity_scope": checkpoint.get("status") == "engineering_benchmark_only"
                and checkpoint.get("research_training_initializer") is False,
            "input_manifest_hash": checkpoint.get("manifest_sha256") == manifest["prelaunch_sha256"],
            "cohort_count": checkpoint.get("completed_cohorts") == index + 1,
            "genuine_optimizer_step_count": bool(steps) and all(step == expected_steps for step in steps),
            "architecture_unchanged": spec["actor_size"] == 24 and spec["critic_size"] == 61 and spec["action_size"] == 7
                and spec["widths"] == [512, 128, 64],
            "reloaded_actions_equal": reloaded_actions_match(model, checkpoint, observation, "cpu"),
            "cohort_charged_cost": checkpoint["cost"]["charged_control_equivalent_transitions"]
                == specification["expected_charged_transitions"] * (index + 1) / specification["cohorts"],
        }
        diagnostics = torch.load(worker_dir / f"diagnostics_{index}.pt", map_location="cpu", weights_only=True)
        checkpoint_checks["diagnostics_reloaded_and_finite"] = finite_tensor_tree(diagnostics)
        checkpoint_checks["diagnostic_shape_matches_report"] = list(diagnostics["controls"].shape) == cohort["checkpoint_validation"]["diagnostic_shape"]
        checkpoints.append({"cohort": index, "checkpoint_sha256": sha256(checkpoint_path),
                            "optimizer_steps": expected_steps, "checks": checkpoint_checks})
        del model, optimizer, checkpoint, diagnostics
    checks["independent_checkpoint_validation"] = all(all(c["checks"].values()) for c in checkpoints)
    resources = manifest["resources"]
    samples = json.loads((directory / "gpu_resources.json").read_text())["samples"]
    checks["resource_telemetry_measured"] = bool(samples)
    checks["resource_summary_matches_samples"] = bool(samples) and resources == resource_summary(samples)
    invalid_initializations = sum(not item["valid"] for cohort in report["cohorts"] for item in cohort["initialization"])
    invalid_jobs = sum(bool(job["forbidden_events"]) or job["outcome"] in {"incomplete", "probe_end", "nonfinite_state"}
                       for cohort in report["cohorts"] for job in cohort["jobs"])
    cost = report["cost"]
    active = cost["active_control_transitions"]
    charged = cost["charged_control_equivalent_transitions"]
    full_cohort_wall = sum(c["complete_cohort_wall_s"] for c in report["cohorts"])
    rollout_wall = sum(c["rollout_wall_s"] for c in report["cohorts"])
    optimizer_wall = sum(c["optimizer_wall_s"] for c in report["cohorts"])
    initialization_wall = sum(c["initialization_wall_s"] for c in report["cohorts"])
    export_wall = sum(c["export_and_reload_wall_s"] for c in report["cohorts"])
    elapsed = manifest["elapsed_wall_s"]
    last = report["cohorts"][-1]
    last_active = sum(last["active_count_by_control"])
    last_charged = specification["expected_charged_transitions"] / specification["cohorts"]
    throughput = {
        "startup_inclusive": {"wall_s": elapsed, "active_samples_per_s": ratio(active, elapsed),
                              "charged_transitions_per_s": ratio(charged, elapsed)},
        "all_cohorts_including_initialization_and_export": {"wall_s": full_cohort_wall,
            "active_samples_per_s": ratio(active, full_cohort_wall), "charged_transitions_per_s": ratio(charged, full_cohort_wall)},
        "sustained_second_cohort_including_initialization_and_export": {"wall_s": last["complete_cohort_wall_s"],
            "active_samples_per_s": ratio(last_active, last["complete_cohort_wall_s"]),
            "charged_transitions_per_s": ratio(last_charged, last["complete_cohort_wall_s"])},
        "rollout_and_optimizer_only": {"wall_s": rollout_wall + optimizer_wall,
            "active_samples_per_s": ratio(active, rollout_wall + optimizer_wall),
            "allocated_rollout_transitions_per_s": ratio(cost["rollout_control_transitions"], rollout_wall + optimizer_wall)},
    }
    checks["positive_finite_timing"] = all(math.isfinite(x) and x > 0 for x in (elapsed, full_cohort_wall, rollout_wall, optimizer_wall))
    checks["resources_within_guards"] = (resources["peak_used_mib"] is not None and resources["peak_used_mib"] < 10240
        and resources["minimum_free_ram_mib"] is not None and resources["minimum_free_ram_mib"] > 4096)
    active_rows = sum(sum(c["active_count_by_control"]) for c in report["cohorts"])
    checks["per_control_active_counts_equal_cost"] = active_rows == active
    trailing_slots = sum(c["scheduling"]["avoidable_trailing_control_slots"] for c in report["cohorts"])
    trailing_wall = sum(c["scheduling"]["fully_absorbing_boundary_intervals_wall_s"] for c in report["cohorts"])
    return {"status": "verified" if all(checks.values()) else "failed_verification", "checks": checks,
            "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "source_commit_at_start": manifest["source_commit_at_start"], "source_dirty_at_start": manifest["source_dirty_at_start"],
            "source_snapshot_sha256": sha256(archive_path), "registration_sha256": manifest["registration_sha256"],
            "behavior_source_sha256": {name: digest for name, digest in archive_hashes.items()
                if name.startswith("src/assembly_recovery/") or name in {
                    "scripts/profile_training_capacity_v1.py", "scripts/run_training_capacity_v1.py",
                    "configs/study.json", "configs/protocol_v4.json", "configs/training_capacity_v1.json"}},
            "specification": specification, "measured_cost": cost, "throughput": throughput,
            "time_components_s": {"launcher_total": elapsed, "app_and_environment_startup": report["app_and_environment_startup_wall_s"],
                "initialization": initialization_wall, "rollout": rollout_wall, "optimizer": optimizer_wall,
                "export_and_reload": export_wall, "cohort_other": full_cohort_wall - initialization_wall - rollout_wall - optimizer_wall - export_wall},
            "resources": {key: resources[key] for key in ("peak_used_mib", "minimum_free_ram_mib", "mean_gpu_utilization_percent", "sample_count")},
            "torch_peak_allocated_bytes": report["torch_peak_allocated_bytes"], "torch_peak_reserved_bytes": report["torch_peak_reserved_bytes"],
            "optimizer_steps": sum(u.get("optimizer_steps", 0) for u in report["optimizer_updates"]),
            "optimizer_active_samples": sum(u["active_samples"] for u in report["optimizer_updates"]),
            "active_rollout_fraction": ratio(active, cost["rollout_control_transitions"]),
            "absorbing_rollout_fraction": ratio(cost["absorbing_control_transitions"], cost["rollout_control_transitions"]),
            "scheduling": {"all_terminal_trailing_slots": trailing_slots,
                "all_terminal_trailing_rollout_fraction": ratio(trailing_slots, cost["rollout_control_transitions"]),
                "measured_trailing_wall_lower_bound_s": trailing_wall,
                "measured_trailing_wall_lower_bound_fraction": ratio(trailing_wall, full_cohort_wall),
                "scope": "Only entire intervals after an all-terminal PPO boundary have a measured avoidable wall-time bound. Partial-slot reuse was not enabled or validated."},
            "invalid_initializations": invalid_initializations, "invalid_jobs": invalid_jobs,
            "resource_guard_stop": manifest["resource_guard_stop"], "checkpoint_validation": checkpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence output already exists; preserve it and choose a new versioned path")
    result = {"schema": 1, "version": "training_capacity_review_v1", "research_result": False,
              "input_manifests": [str(args.baseline.resolve()), str(args.candidate.resolve())],
              "scope_and_limitations": [
                  "Bounded representative resource measurement with real optimization, not a curriculum or policy score comparison.",
                  "2048 performs twice the initialized jobs and planned charged work. Active samples and optimizer steps are measured, not matched.",
                  "Fresh-policy early cohorts do not establish mature-policy utilization or long-run stability.",
                  "Task timestep sensitivity remains open. This capacity choice does not pass physics, recovery or research-training gates.",
                  "Simulator observations have synthetic noise; no camera perception, pickup, hardware transfer or force-certified safety claim."]}
    try:
        baseline = verify_run(args.baseline)
        candidate = verify_run(args.candidate)
        if baseline["specification"]["num_envs"] != 1024 or candidate["specification"]["num_envs"] != 2048:
            raise ValueError("Review requires a 1024 baseline and 2048 candidate")
        if baseline["registration_sha256"] != candidate["registration_sha256"]:
            raise ValueError("Capacity runs did not use the same registered protocol")
        if baseline["behavior_source_sha256"] != candidate["behavior_source_sha256"]:
            raise ValueError("Behavior sources changed between the capacity runs")
        result.update(runs=[baseline, candidate], decision=adoption_decision(baseline, candidate))
        result["status"] = "verified" if baseline["status"] == candidate["status"] == "verified" else "failed_verification"
        result["total_charged_development_transitions"] = sum(r["measured_cost"]["charged_control_equivalent_transitions"] for r in result["runs"])
    except Exception as exc:
        result.update(status="failed_verification", error=f"{type(exc).__name__}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "decision": result.get("decision", {}).get("selected_num_envs")}))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
