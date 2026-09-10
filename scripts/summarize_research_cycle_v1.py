"""Summarize the executed recovery-teaching falsification cycle; no simulator imports."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assembly_recovery.protocol import sha256  # noqa: E402


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/research_cycle_decision_v1.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = "artifacts/assembly/research_cycle_20260910_r01/"
    aggregates = {
        "original_servo120": base + "contact_120_240_480_960_corrected_v3.json",
        "servo480": base + "servo_impact_comparison_v1.json",
        "servo480_reference20mm_s": base + "approach_slew_comparison_v1.json",
    }
    summaries = {key: read(path) for key, path in aggregates.items()}
    if not all(row["status"] == "verified_diagnostic" for row in summaries.values()):
        raise ValueError("Every executed condition must verify before adjudication")
    final = summaries["servo480_reference20mm_s"]
    if final["decision"]["earns_full_job_validation"]:
        raise ValueError("This negative-cycle summary cannot discard a passing final gate")
    locations = [
        ("original_servo120", 120, "contact_impact_v1_r01/120"),
        ("original_servo120", 240, "contact_impact_v2_r01/240"),
        ("original_servo120", 480, "contact_impact_v2_r01/480"),
        ("original_servo120", 960, "contact_impact_960_v2_r01/960"),
        ("servo480", 480, "servo_impact_v1_r01/480"),
        ("servo480", 960, "servo_impact_v1_r01/960"),
        ("servo480_reference20mm_s", 480, "approach_slew_v1_r01/480"),
        ("servo480_reference20mm_s", 960, "approach_slew_v1_r01/960"),
    ]
    runs, resources = [], []
    for condition, hz, name in locations:
        path = Path("artifacts/assembly") / name
        manifest, report = read(path / "manifest.json"), read(path / "validation/report.json")
        if manifest["returncode"] != 0 or manifest["timed_out"] or report["status"] != "completed":
            raise ValueError("Do not call a failed or timed-out simulator process complete")
        index = next(i for i, row in enumerate(summaries[condition]["runs"]) if row["physics_hz"] == hz)
        verified = summaries[condition]["verification"][index]
        if not all(verified["checks"].values()):
            raise ValueError("Corrected verification failed")
        if len(report["jobs"]) != 28 or len(report["recovery_witness"]) != 28:
            raise ValueError("Lost requested cases")
        active = [i for i, job in enumerate(report["jobs"]) if job["outcome"] == "probe_end"]
        stalled = [i for i in active if report["recovery_witness"][i]["witnessed_failure_step"] is not None]
        runs.append({
            "condition": condition, "native_physics_hz": hz,
            "external_servo_hz": 120 if condition == "original_servo120" else 480,
            "external_sensor_hz": 120, "policy_hz": 15,
            "applied_controller": "Unchanged scripted first attempt; no applied learned-policy actions",
            "prefix_summary": report["prefix_summary"], "active_contact_stalls": len(stalled),
            "active_contact_stall_case_ids": [report["cases"][i]["case_id"] for i in stalled],
            "original_launcher_status": manifest["status"],
            "simulator_process_completed": True, "corrected_cpu_verification_passed": True,
            "corrected_verification_checks": len(verified["checks"]),
            "cost": report["cost"], "wall_s_including_export_and_verification": manifest["elapsed_wall_s"],
            "source_commit_at_start": manifest["source_commit_at_start"],
            "source_dirty_at_start": manifest["source_dirty_at_start"],
            "path": path.as_posix(),
            "artifact_sha256": {p.as_posix(): sha256(ROOT / p) for p in (
                path / "manifest.json", path / "source.zip", path / "validation/report.json", path / "validation/trajectory.pt")},
        })
        resources.extend(read(path / "gpu_resources.json")["samples"])
    capacity = read("evidence/training_capacity_v2.json")
    training_manifests = [read(f"artifacts/assembly/capacity{n}_v1_r01/manifest.json") for n in (1024, 2048)]
    if any(m["status"] != "completed" or not all(m["checks"].values()) for m in training_manifests):
        raise ValueError("Capacity runs are not complete and verified")
    training_cost = sum(m["measured_cost"]["charged_control_equivalent_transitions"] for m in training_manifests)
    diagnostic_cost = sum(r["cost"]["charged_reference_transitions"] for r in runs)
    if training_cost != 2816256 or diagnostic_cost != 67396:
        raise ValueError("Unexpected executed-cycle cost")
    for m in training_manifests:
        resources.extend(m["resources"]["samples"])
    registration = read("configs/recovery_teaching_registration_v1.json")
    failures = [
        "evidence/contact_impact_verifier_failure_v1.json",
        "evidence/contact_impact_precision_verifier_failure_v1.json",
        "evidence/training_capacity_v1.json",
    ]
    checkpoint = Path("artifacts/assembly/uniform_unclipped_170_v4_r02/pilot/budget_checkpoint.pt")
    models = [p for n in (1024, 2048) for p in sorted(
        (ROOT / f"artifacts/assembly/capacity{n}_v1_r01/profile").glob("benchmark_cohort_*.pt"))]
    result = {
        "schema": 1, "status": "executed_cycle_closed_reject_premise", "recorded_at_utc": datetime.now(UTC).isoformat(),
        "research_question": registration["question"],
        "scope_and_limitations": [
            "One previously used development seed, eight scripted four-second prefix conditions; repeated cases are not independent deployment trials. No final-test use.",
            "Requests active at four seconds are censored, not failed complete30-second jobs. The scripted prefix, old ordinary learned model and untrained candidate have distinct labels.",
            "The active-window contact-impulse gate is terminal-censored and cannot alone establish physical nonconvergence. In the final pair, outcome-count and wrist-peak margins pass; contact-impulse and failure-support margins fail.",
            "No active witnessed contact stalls survive at480/960Hz in any tested controller condition. This specifically defeats the proposed handoff assay, not all learned recovery or FORGE.",
            "Coarse initialization is shared at every resolution. Native integration, initial gripper reactions and sampled control remain coupled; finest timestep is not assumed ground truth and no unique mechanism is claimed.",
            "No controller reset, part-pose write, attachment change, threshold relaxation, filtered-force substitution, difficult-case removal or learned-method superiority claim.",
        ],
        "decision": {
            "type": "reject_premise", "simulation_issue_resolved": False,
            "advance_fixed_prefix_teaching_campaign": False, "extend_to_gear": False,
            "candidate_method_effect": "Untested; zero candidate GPU training or policy evaluation",
            "reason": "The registered final correction failed stability and failure exposure. More decisively, all fine-resolution conditions leave zero active witnessed contact stalls for the learned handoff. Full-job validation and substantial method training were conditional on a passing gate.",
            "chosen_contribution": "Bounded empirical engineering result: the fixed-prefix recovery assay loses its target failure cohort under native refinement, and these two common control corrections do not restore it. A measured2048-environment capacity improvement is separately verified.",
            "not_rejected": "Recovery teaching in general, all Franka/FORGE tasks, or the preserved ordinary model's explicitly scoped120Hz development measurements",
        },
        "registration_sha256": {
            path: sha256(ROOT / path) for path in (
                "configs/recovery_teaching_registration_v1.json", "configs/contact_impact_registration_v1.json",
                "configs/contact_impact_continuation_v2.json", "configs/servo_impact_registration_v1.json",
                "configs/approach_slew_registration_v1.json", "configs/training_capacity_v1.json")},
        "physics_runs": runs,
        "verified_aggregates": {
            key: {"path": path, "sha256": sha256(ROOT / path), "decision": summaries[key]["decision"],
                  "pairs": [{k: pair[k] for k in (
                      "physics_hz", "margins", "abort_count_difference", "individual_outcome_changes",
                      "median_peak_wrist_relative_difference", "median_fixture_impulse_relative_difference")}
                      for pair in summaries[key]["pairs"]]}
            for key, path in aggregates.items()},
        "ordinary_training": {
            "historical_complete_run": {"charged_transitions": 10326272, "training_seed": 170,
                "development_completions": "81/84 at original120Hz", "checkpoint_path": checkpoint.as_posix(),
                "checkpoint_sha256": sha256(ROOT / checkpoint)},
            "new_work": "Two fresh engineering-only PPO profiles, two cohorts each at1024 and2048, both seed170. Real optimization, initialization, diagnostics and checkpoint export/reload ran.",
            "new_profile_charged_transitions": training_cost, "new_profile_optimizer_steps": sum(
                m["throughput"]["optimizer_steps"] for m in training_manifests),
            "new_recovery_reliability_evaluation": False,
        },
        "candidate_training": {"gpu_runs": 0, "charged_training_transitions": 0,
            "implementation": ["src/assembly_recovery/recovery_teaching_v1.py", "scripts/train_recovery_teaching_v1.py"],
            "verified_scope": "CPU masks and vectorized-prefix tests, including exclusion of off-policy actions and rewards. The original120Hz prototype was not launched or adopted as a corrected physical task."},
        "historical_controller_comparison": {
            "scope": "Two previously confirmed development seeds under original120Hz; not rerun as full jobs in this cycle.",
            "whole_job_completions_of_56": {"ordinary_learned": 47, "scripted_retry": 47, "scripted_continued_insertion": 11},
            "same_active_witnessed_stall_completions_of_37": {"ordinary_learned": 30, "scripted_retry": 31, "scripted_continued_insertion": 1},
            "source": "evidence/post_stall_confirmation_v2_verified.json",
            "adaptive_training_benefit_established": False, "learned_reliability_advantage_established": False,
            "original_withdrawal_competence_gate_passed": False,
        },
        "capacity": {"evidence": "evidence/training_capacity_v2.json",
            "sha256": sha256(ROOT / "evidence/training_capacity_v2.json"),
            "verified_status": capacity["status"],
            "adopt_2048_for_same_runtime": True, "4096_tested": False,
            "active_samples_per_s": [461.0531309452945, 595.4227079941361],
            "charged_references_per_s": [1664.2679979257636, 2306.340262284675],
            "scope": "Startup-inclusive engineering throughput at unequal work. Reprofile after any material physical/task change; no learning-quality comparison."},
        "cost": {"prior_tracked_reference_lower_bound": 24995215.25,
            "capacity_optimization_reference_transitions": training_cost,
            "physics_diagnostic_reference_transitions": diagnostic_cost,
            "cycle_charged_reference_transitions": training_cost + diagnostic_cost,
            "cumulative_tracked_reference_lower_bound": 24995215.25 + training_cost + diagnostic_cost,
            "completed_simulator_processes": 10, "simulator_process_failures": 0,
            "original_queue_cpu_verification_failures": 2, "separate_capacity_pair_cpu_check_failure": 1,
            "diagnostic_requests": 224, "capacity_training_requests": 6144,
            "candidate_training_transitions": 0, "final_test_transitions": 0,
            "summed_launcher_wall_s": sum(r["wall_s_including_export_and_verification"] for r in runs)
                + sum(m["elapsed_wall_s"] for m in training_manifests),
            "peak_gpu_mib": max(row["used_mib"] for row in resources),
            "minimum_free_ram_mib": min(row["free_ram_mib"] for row in resources if row["free_ram_mib"] is not None),
            "unit": "One reference transition equals eight executed native physics environment steps, including initialization, active/absorbing slots and all preserved failures. Corrected CPU reviews add zero simulator work."},
        "preserved_failures": [{"path": path, "sha256": sha256(ROOT / path)} for path in failures],
        "saved_benchmark_models": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p),
            "status": "engineering_benchmark_only; not a research initializer"} for p in models],
        "next_action": "Release the bounded technical result in the existing README/ROADMAP with the figure and verified evidence. Close this fixed-prefix campaign. Any future learned comparison requires a new, independently justified physical/load definition and a failure generator that actually leaves retained-grasp contact stalls; do not keep tuning this assay or open final tests.",
    }
    # Derive rates from manifests rather than rounded printed summaries.
    result["capacity"]["active_samples_per_s"] = [m["measured_cost"]["active_control_transitions"] / m["elapsed_wall_s"] for m in training_manifests]
    result["capacity"]["charged_references_per_s"] = [m["measured_cost"]["charged_control_equivalent_transitions"] / m["elapsed_wall_s"] for m in training_manifests]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps({"decision": result["decision"]["type"], "cost": result["cost"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
