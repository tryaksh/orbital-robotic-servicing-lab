"""Verify fixed-prefix failure exposure, full outcomes, matched randomness and unchanged retry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.evaluation import JobCriteria  # noqa: E402
from assembly_recovery.protocol import sha256  # noqa: E402
from scripts.probe_training import replay_physics  # noqa: E402
from scripts.verify_training_contract import verify_assets  # noqa: E402
from scripts.verify_uniform_pilot import metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence")
    checks, arms, traces = {}, {}, {}
    discounts = .995 ** torch.arange(450, dtype=torch.float64)
    for arm in ("learned", "retry", "continue"):
        path = args.run / arm
        manifest = verify_assets(path)
        r = json.loads((path / "diagnosis/report.json").read_text())
        d = torch.load(path / "diagnosis/trajectory.pt", weights_only=True, map_location="cpu")
        traces[arm] = d
        checks[arm + "_completed"] = manifest["status"] == r["status"] == "completed"
        checks[arm + "_valid_requests"] = len(r["jobs"]) == 28 and all(v["valid"] for v in r["initialization"])
        checks[arm + "_exact_cost"] = r["cost"]["charged_control_equivalent_transitions"] == 12834.5
        checks[arm + "_finite_arrays"] = all(bool(torch.isfinite(v).all()) for v in d.values() if isinstance(v, torch.Tensor))
        replay_physics(d["physics"], JobCriteria(**r["criteria"]), r["jobs"], "diagnosis", r["recovery_witness"])
        rows = []
        for i, (case, job, witness) in enumerate(zip(r["cases"], r["jobs"], r["recovery_witness"], strict=True)):
            terminal = round(job["elapsed_s"] * 120)
            included, notification = terminal // 8, (terminal + 7) // 8 - 1
            ledger = {k: float((v[:included, i].double() * discounts[:included]).sum()) for k, v in d["reward_terms"].items()}
            credit = float((d["completion_credit"][:, i].double() * discounts).sum())
            ret = float((d["job_reward"][:, i].double() * discounts).sum())
            expected_credit = (35 / 12) * .995 ** included * (1 - .995 ** (450 - included)) / (1 - .995) if job["success"] else 0.
            checks[f"{arm}_{i}_ledger"] = abs(sum(ledger.values()) + credit - ret) < .001
            checks[f"{arm}_{i}_return"] = abs(ret - r["discounted_job_returns"][i]) < 1e-5
            checks[f"{arm}_{i}_credit"] = abs(credit - expected_credit) < .002
            checks[f"{arm}_{i}_absorbing_zero"] = bool((d["job_reward"][notification + 1:, i] == 0).all())
            active = d["active"][:, i].bool()
            native = d["physics"][:terminal, i]
            rows.append({"case_id": case["case_id"], "bin_id": case["bin_id"], "outcome": job["outcome"],
                         "success": job["success"], "elapsed_s": job["elapsed_s"], "discounted_return": ret,
                         "upstream_discounted_return": sum(ledger.values()), "discounted_reward_terms": ledger,
                         "completion_credit_return": credit, "peak_raw_wrist_n": job["peak_raw_wrist_force_n"],
                         "final_target_xyz_m": d["target_xyz"][active, i][-1].tolist(),
                         "final_tool_xyz_m": d["tool_xyz"][active, i][-1].tolist(),
                         "minimum_held_base_height_m": float(native[:, 13].min() + .025),
                         "contact_fraction": float((native[:, 11] > .1).float().mean()),
                         "witnessed_recovery": witness["witnessed_complete_recovery"],
                         "prefix_terminal": terminal <= 480,
                         "witnessed_failure_by_handoff": witness["witnessed_failure_step"] is not None and witness["witnessed_failure_step"] <= 480,
                         "active_at_handoff": terminal > 480,
                         "recovery_after_prefix_failure": (terminal > 480 and witness["witnessed_failure_step"] is not None
                             and witness["witnessed_failure_step"] <= 480 and witness["witnessed_complete_recovery"]
                             and witness["witnessed_withdrawal_step"] > 480)})
        arms[arm] = {"manifest_sha256": sha256(path / "manifest.json"), "checkpoint_sha256": r["checkpoint_sha256"],
                     "cost": r["cost"], "metrics": metrics([r]), "cases": rows}
    reference = traces["learned"]
    checks["learned_actions_after_handoff_equal_frozen_means"] = torch.equal(reference["requested_action"][60:], reference["mean_action"][60:])
    registration_path = ROOT / "configs/failure_prefix_recovery_v1.json"
    registration = json.loads(registration_path.read_text())
    checks["exact_registered_checkpoint"] = arms["learned"]["checkpoint_sha256"] == registration["checkpoint_sha256"]
    for arm in ("retry", "continue"):
        data = traces[arm]
        checks[arm + "_same_initial_state"] = reference["initial"].keys() == data["initial"].keys() and all(
            torch.equal(value, data["initial"][name]) for name, value in reference["initial"].items())
        for name in ("sensor_draws", "cuda_rng_before_steps", "dead_zone"):
            checks[arm + "_same_" + name] = torch.equal(reference[name], data[name])
        for name in ("requested_action", "applied_action", "actor_before", "job_reward"):
            checks[arm + "_same_prefix_" + name] = torch.equal(reference[name][:60], data[name][:60])
        checks[arm + "_same_prefix_physics"] = torch.equal(reference["physics"][:480], data["physics"][:480])
        checks[arm + "_same_checkpoint"] = arms[arm]["checkpoint_sha256"] == arms["learned"]["checkpoint_sha256"]
    old_path = ROOT / "artifacts/assembly/partial_checkpoint_diagnosis_v3_r02/retry/diagnosis/trajectory.pt"
    old = torch.load(old_path, map_location="cpu", weights_only=True)
    checks["frozen_retry_positive_control"] = all(torch.equal(old[key], traces["retry"][key]) for key in (
        "physics", "requested_action", "applied_action", "actor_before", "sensor_draws", "job_reward", "completion_credit"))
    prefix_sets = []
    for item in arms.values():
        rows = item["cases"]
        eligible = [r for r in rows if r["witnessed_failure_by_handoff"] and r["active_at_handoff"]]
        prefix_sets.append([r["case_id"] for r in eligible])
        item["failure_exposure"] = {
            "all_requested_jobs": len(rows),
            "prefix_completions": sum(r["prefix_terminal"] and r["success"] for r in rows),
            "prefix_failed_jobs": sum(r["prefix_terminal"] and not r["success"] for r in rows),
            "active_at_handoff": sum(r["active_at_handoff"] for r in rows),
            "witnessed_prefix_failures_active_at_handoff": len(eligible),
            "completed_after_witnessed_prefix_failure": sum(r["success"] for r in eligible),
            "witnessed_recoveries_after_prefix_failure": sum(r["recovery_after_prefix_failure"] for r in eligible),
            "scope": "Conditional recovery counts accompany all-request outcomes; prefix failures and prefix scripted completions remain in the full-job denominator."}
    checks["identical_prefix_failure_cohort"] = prefix_sets[0] == prefix_sets[1] == prefix_sets[2]
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed", "research_result": False,
              "registration_sha256": sha256(registration_path), "checks": checks, "arms": arms,
              "charged_transitions": sum(v["cost"]["charged_control_equivalent_transitions"] for v in arms.values()),
              "scope_and_limitations": [
                  "Separate, fixed four-second scripted insertion prefix followed by learned mean, unchanged retry or continued insertion. The original 30-second clock includes the prefix.",
                  "Handoff time is fixed before launch and independent of evaluator contact, hidden state or policy outcomes. No within-job reset, pose write or attachment change.",
                  "Prefix completions were produced by the script. Only post-handoff outcomes after a witnessed prefix failure can establish learned recovery in this assay.",
                  "One development seed with 28 requested jobs per controller; conditional recovery samples are not all jobs or independent training seeds.",
                  "This assay does not replace or retrospectively pass the original 84-case competence gate and is not an equal-cost learning-method comparison.",
                  "No geometry, force abort, dwell, reward, trained parameters or final-test cases changed."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "outcomes": {k: {"metrics": v["metrics"]["all"], "failure_exposure": v["failure_exposure"]} for k, v in arms.items()},
                      "failed_checks": [k for k, v in checks.items() if not v]}, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
