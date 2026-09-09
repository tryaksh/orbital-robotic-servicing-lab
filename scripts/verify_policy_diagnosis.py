"""Rehash and compare every registered policy-diagnosis request."""
from __future__ import annotations

import argparse
import json
import statistics
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
    checks, results, traces, reports, rows = {}, {}, {}, {}, {}
    for arm in ("deterministic", "stochastic", "retry"):
        path = args.run / arm
        manifest = verify_assets(path)
        report = json.loads((path / "diagnosis/report.json").read_text())
        data = torch.load(path / "diagnosis/trajectory.pt", map_location="cpu", weights_only=True)
        reports[arm], traces[arm], rows[arm] = report, data, []
        checks[arm + "_complete"] = manifest["status"] == report["status"] == "completed"
        checks[arm + "_all_cases"] = len(report["jobs"]) == 28
        checks[arm + "_valid_initializations"] = all(v["valid"] for v in report["initialization"])
        checks[arm + "_exact_cost"] = report["cost"]["charged_control_equivalent_transitions"] == 12834.5
        checks[arm + "_sensor_draw_shape"] = list(data["sensor_draws"].shape) == [3600, 28, 10]
        checks[arm + "_finite_trace"] = all(bool(torch.isfinite(v).all()) for v in data.values() if isinstance(v, torch.Tensor))
        replay_physics(data["physics"], JobCriteria(**report["criteria"]), report["jobs"], "diagnosis", report["recovery_witness"])
        for i, (case, job) in enumerate(zip(report["cases"], report["jobs"], strict=True)):
            end = round(job["elapsed_s"] * 120)
            count = end // 8
            discount = 0.995 ** torch.arange(count, dtype=torch.float64)
            ledger = {k: float((v[:count, i].double() * discount).sum()) for k, v in data["reward_terms"].items()}
            ret = float((data["raw_reward"][:count, i].double() * discount).sum())
            error = float((sum(data["reward_terms"].values())[:, i] - data["raw_reward"][:, i]).abs().max())
            checks[f"{arm}_{i}_return"] = abs(ret - report["discounted_job_returns"][i]) < 1e-5 and error < 1e-5
            checks[f"{arm}_{i}_finite_terminal_mask"] = bool((data["job_reward"][count:, i] == 0).all())
            active = data["active"][:, i].bool()
            physics = data["physics"][:end, i]
            row = {"case_id": case["case_id"], "bin_id": case["bin_id"], "outcome": job["outcome"],
                   "success": job["success"], "elapsed_s": job["elapsed_s"], "discounted_return": ret,
                   "undiscounted_return": float(data["raw_reward"][:count, i].double().sum()),
                   "discounted_terms": ledger, "upstream_success_seen": job["upstream_success_seen"],
                   "raw_peak_wrist_n": job["peak_raw_wrist_force_n"],
                   "fixture_contact_fraction": float((physics[:, 11] > 0.1).float().mean()),
                   "minimum_base_height_m": float(physics[:, 13].min() + 0.025),
                   "final_base_height_m": float(physics[-1, 13] + 0.025),
                   "target_height_min_max_m": [float(data["target_xyz"][active, i, 2].min()), float(data["target_xyz"][active, i, 2].max())],
                   "final_target_xyz_m": data["target_xyz"][active, i][-1].tolist(),
                   "final_tool_xyz_m": data["tool_xyz"][active, i][-1].tolist(),
                   "tool_height_range_m": float(data["tool_xyz"][active, i, 2].max() - data["tool_xyz"][active, i, 2].min()),
                   "requested_action_clipped_fraction": float((data["requested_action"][active, i].abs() > 1).float().mean()),
                   "witness": report["recovery_witness"][i]}
            rows[arm].append(row)
        results[arm] = {"controller": arm, "manifest_sha256": sha256(path / "manifest.json"),
                        "checkpoint_sha256": report["checkpoint_sha256"], "cost": report["cost"],
                        "process_wall_s": manifest["elapsed_wall_s"], "metrics": metrics([report]),
                        "mean_discounted_terms": {k: statistics.mean(row["discounted_terms"][k] for row in rows[arm])
                                                   for k in data["reward_terms"]}}
    reference = traces["deterministic"]
    prior_path = ROOT / "artifacts/assembly/uniform_dev_10070_v2/pilot/diagnostics_0.pt"
    prior = torch.load(prior_path, map_location="cpu", weights_only=True)
    checks["instrumentation_preserves_original_deterministic_physics"] = torch.equal(reference["physics"], prior["physics"])
    checks["instrumentation_preserves_original_deterministic_actions"] = torch.equal(reference["applied_action"], prior["controls"][:, :, 10:17])
    for arm in ("stochastic", "retry"):
        data = traces[arm]
        checks[arm + "_same_initial_tensors"] = reference["initial"].keys() == data["initial"].keys() and all(
            torch.equal(v, data["initial"][k]) for k, v in reference["initial"].items())
        checks[arm + "_same_sensor_draws"] = torch.equal(reference["sensor_draws"], data["sensor_draws"])
        checks[arm + "_same_environment_rng"] = torch.equal(reference["cuda_rng_before_steps"], data["cuda_rng_before_steps"])
        checks[arm + "_same_dead_zone_randomization"] = torch.equal(reference["dead_zone"], data["dead_zone"])
        checks[arm + "_same_cases"] = reports[arm]["cases"] == reports["deterministic"]["cases"]
    comparisons = []
    for i in range(28):
        comparisons.append({"case_id": rows["retry"][i]["case_id"],
                            "outcomes": {arm: rows[arm][i]["outcome"] for arm in rows},
                            "returns": {arm: rows[arm][i]["discounted_return"] for arm in rows},
                            "retry_complete_deterministic_failed": rows["retry"][i]["success"] and not rows["deterministic"][i]["success"],
                            "deterministic_return_minus_retry": rows["deterministic"][i]["discounted_return"] - rows["retry"][i]["discounted_return"]})
    solvable = [r for r in comparisons if r["retry_complete_deterministic_failed"]]
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed", "research_result": False,
              "registration": "evidence/pilot_failure_diagnosis_v2.json", "checks": checks, "arms": results,
              "paired_comparisons": comparisons, "case_diagnostics": rows,
              "summary": {"charged_transitions": sum(r["cost"]["charged_control_equivalent_transitions"] for r in results.values()),
                          "retry_complete_deterministic_failed": len(solvable),
                          "failed_deterministic_outreturns_successful_retry": sum(r["deterministic_return_minus_retry"] > 0 for r in solvable),
                          "mean_return_advantage_of_failed_deterministic_over_successful_retry": statistics.mean(r["deterministic_return_minus_retry"] for r in solvable) if solvable else None},
              "scope_and_limitations": ["Frozen deterministic and independent stochastic checkpoint, unchanged script, all 28 seed-10070 cases.",
                  "Full finite-job returns and outcomes are paired using exact initial tensors, raw sensor draws and nuisance records.",
                  "One frozen network and one development seed cannot separate every exploration or optimization failure.",
                  "A failed behavior earning more return than a successful matched trajectory establishes a local incentive counterexample, not global optimality.",
                  "No training, final-test opening, reward correction or change to the original pilot result."]}
    with args.output.open("x") as file:
        file.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "summary": result["summary"],
                      "metrics": {k: v["metrics"]["all"] for k, v in results.items()},
                      "failed_checks": [k for k, v in checks.items() if not v]}, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
