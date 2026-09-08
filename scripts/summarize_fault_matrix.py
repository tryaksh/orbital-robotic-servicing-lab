"""Verify saved matrix pairs and retain every requested development case by bin."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.faults import development_cases  # noqa: E402
from assembly_recovery.protocol import sha256  # noqa: E402
from assembly_recovery.retry_controller import RetrySettings  # noqa: E402
from scripts.compare_recovery import recovery_witness, verify_run  # noqa: E402


def classify_job(job, contact_stall, witnessed_recovery, fixture_contact):
    if witnessed_recovery:
        return "witnessed_recovery"
    if job["success"]:
        return "completion_without_full_recovery_witness" if contact_stall else "completion_without_witnessed_contact_failure"
    if job["outcome"] == "deadline":
        return "contact_stall_deadline" if contact_stall else "contact_without_stall_deadline" if fixture_contact else "free_space_miss_deadline"
    return job["outcome"]


def summarize(plan, comparisons):
    study = plan["study"]
    expected_all = [c for seed in study["splits"]["development_seeds"] for c in development_cases(study, seed)]
    if plan["cases"] != expected_all or plan["requested_jobs"] != 2 * len(expected_all):
        raise ValueError("Plan case list or denominator differs from declared support")
    expected_settings = asdict(RetrySettings(**plan["retry_settings"]))
    rows, runs = [], []
    for seed, comparison_path in zip(study["splits"]["development_seeds"], comparisons, strict=True):
        comparison = json.loads(comparison_path.read_text())
        if comparison["status"] != "verified":
            raise ValueError("Matrix requires verified matched pairs")
        expected = development_cases(study, seed)
        arm_data = []
        for arm, run in zip(("retry", "continue"), comparison["runs"], strict=True):
            directory = ROOT / Path(run["manifest"]).parent
            manifest, report, controls, physics = verify_run(directory)
            if sha256(directory / "manifest.json") != run["manifest_sha256"]:
                raise ValueError("Comparison manifest changed")
            if report.get("fault_cases") != expected or len(report["jobs"]) != len(expected):
                raise ValueError("Requested case registry differs from the saved cohort")
            if manifest["source_hashes"]["configs/study.json"] != plan["study_sha256"]:
                raise ValueError("Study changed after matrix predeclaration")
            if (manifest["retry_config_sha256"] != plan["retry_settings_sha256"]
                    or any(c["settings"] != expected_settings for c in report["controllers"])
                    or report["simulated_seconds"] != plan["deadline_s"]
                    or report["held_part_gravity"] != study["job_protocol"]["primary_held_part_gravity"]
                    or report["velocity_mode"] != "corrected"):
                raise ValueError("Controller settings or horizon changed after predeclaration")
            arm_data.append((report, controls, physics))
            runs.append({"run_id": manifest["run_id"], "arm": arm, "seed": seed,
                         "manifest": run["manifest"], "manifest_sha256": run["manifest_sha256"],
                         "resources": report["resources"]})
        for i, case in enumerate(expected):
            row = {"case": case}
            for arm, (report, controls, physics) in zip(("retry", "continue"), arm_data, strict=True):
                job = report["jobs"][i]
                witness = recovery_witness(report, controls, physics, i)
                stall_step = job["first_stall_step"]
                window_steps = round(report["criteria"]["stall_window_s"] / report["criteria"]["physics_dt"])
                samples = physics[job["job_id"]]
                window = samples[max(0, stall_step - window_steps):stall_step] if stall_step else []
                contact_stall = bool(window) and sum(p["peg_fixture_contact_force_n"] > 0.1 for p in window) / len(window) >= 0.8
                withdrawal = (witness.get("withdrawn_part_rise_m", 0) >= 0.005
                              and (witness.get("withdrawal_final_base_clearance_m") or 0) > 0.001
                              and witness.get("withdrawal_longest_fixture_clear_s", 0) >= 0.2)
                fixture_contact = any(p["peg_fixture_contact_force_n"] > 0.1 for p in samples)
                classification = classify_job(job, contact_stall, witness["witnessed_complete_recovery"], fixture_contact)
                row[arm] = {"job_index": i, "initialization_valid": report["initialization_validity"][i]["valid"],
                            "outcome": job["outcome"], "classification": classification,
                            "upstream_success_seen": job["upstream_success_seen"],
                            "retained_grasp": job["grasp_contact_observed"] and job["outcome"] != "lost_grasp",
                            "fixture_contact_observed": fixture_contact,
                            "contact_failure_witnessed": contact_stall, "withdrawal_verified": withdrawal,
                            "witnessed_complete_recovery": witness["witnessed_complete_recovery"],
                            "elapsed_s": job["elapsed_s"], "peak_raw_wrist_force_n": job["peak_raw_wrist_force_n"],
                            "force_violation": job["peak_raw_wrist_force_n"] > report["criteria"]["force_budget_n"],
                            "discounted_job_return": report["returns"][i]["discounted_return"],
                            "terminal_partial_physics_steps_omitted": report["returns"][i]["terminal_partial_physics_steps_without_endpoint_reward"]}
            rows.append(row)
    bins = []
    for fault_bin in study["fault_support"]["bins"]:
        members = [r for r in rows if r["case"]["bin_id"] == fault_bin["id"]]
        result = {"bin": fault_bin, "requested_cases_per_arm": len(members)}
        for arm in ("retry", "continue"):
            arm_rows = [r[arm] for r in members]
            result[arm] = {"outcomes": dict(Counter(r["outcome"] for r in arm_rows)),
                           "classifications": dict(Counter(r["classification"] for r in arm_rows)),
                           **{key: sum(r[key] for r in arm_rows) for key in (
                               "initialization_valid", "retained_grasp", "contact_failure_witnessed", "withdrawal_verified",
                               "witnessed_complete_recovery", "force_violation", "upstream_success_seen")},
                           "mean_discounted_job_return": sum(r["discounted_job_return"] for r in arm_rows) / len(arm_rows)}
        recoveries = result["retry"]["witnessed_complete_recovery"]
        result["feasibility_status"] = "sampled_recovery_support_demonstrated" if recoveries else "unresolved_recovery_feasibility"
        result["entire_numeric_range_validated"] = False
        bins.append(result)
    return {"schema": 1, "research_result": False, "status": "verified_development_matrix", "plan": plan["run_id"],
            "study_sha256_at_predeclaration": plan["study_sha256"], "bins": bins, "runs": runs, "cases": rows,
            "requested_jobs": 2 * len(rows), "unique_development_cases": len(rows),
            "control_transitions_including_post_terminal_holds": sum(r["resources"]["control_transitions"] for r in runs),
            "scope_and_limitations": ["Scripted development feasibility matrix; not learned recovery, equal-cost training benefit, or final-test evidence.",
                                      "Four directions/replicates at one representative severity per bin per development seed; range interiors/boundaries and reserved combinations are not validated.",
                                      "Shared grasp offset, pose/force noise, mass, controller gains, smoothing, and dead-zone randomization remain; a nominal label means no controlled fault, not guaranteed insertion.",
                                      "All requested jobs, including invalid initialization, force abort, grasp loss and deadline, remain in denominators.",
                                      "Control-transition cost includes post-terminal holds but excludes reset IK/grasp initialization stepping; initialization wall time is in manifests. Training must instrument this separately.",
                                      "Returns use unchanged upstream reward with endpoint cutoff and omitted partial terminal interval; training terminal integration remains open.",
                                      "Geometry, synthetic observation interface, 30 s deadline, 20 N wrist abort budget and 0.5 s seating dwell are shared across each pair.",
                                      "Failure of this bounded script does not establish an infeasible fault. No additional controller tuning is authorized by this report."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.plan.read_text()), args.comparisons)
    result["plan_path"] = args.plan.as_posix()
    result["plan_sha256"] = sha256(args.plan)
    result["comparison_artifacts"] = [{"path": p.as_posix(), "sha256": sha256(p)} for p in args.comparisons]
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"requested_jobs": result["requested_jobs"], "bins": result["bins"]}, indent=2))


if __name__ == "__main__":
    main()
