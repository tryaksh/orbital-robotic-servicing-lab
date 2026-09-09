"""Verify v3 reward controls against immutable v2 physical trajectories."""
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
    old = ROOT / "artifacts/assembly/policy_diagnosis_v2_r01"
    checks, results, corrected, paired = {}, {}, {}, []
    for arm in ("deterministic", "stochastic", "retry"):
        m = verify_assets(args.run / arm)
        p = args.run / arm / "diagnosis"
        r = json.loads((p / "report.json").read_text())
        original = json.loads((old / arm / "diagnosis/report.json").read_text())
        d = torch.load(p / "trajectory.pt", weights_only=True, map_location="cpu")
        prior = torch.load(old / arm / "diagnosis/trajectory.pt", weights_only=True, map_location="cpu")
        checks[arm + "_process_complete"] = m["status"] == r["status"] == "completed"
        checks[arm + "_all_28_valid"] = len(r["jobs"]) == 28 and all(v["valid"] for v in r["initialization"])
        checks[arm + "_same_jobs"] = r["jobs"] == original["jobs"]
        checks[arm + "_same_witnesses"] = r["recovery_witness"] == original["recovery_witness"]
        checks[arm + "_same_cost"] = r["cost"] == original["cost"]
        for name, value in prior.items():
            if name == "job_reward":
                checks[arm + "_preserved_upstream_job_reward"] = torch.equal(value, d["upstream_job_reward"])
            elif isinstance(value, dict):
                checks[arm + "_same_" + name] = value.keys() == d[name].keys() and all(torch.equal(v, d[name][k]) for k, v in value.items())
            else:
                checks[arm + "_same_" + name] = torch.equal(value, d[name])
        replay_physics(d["physics"], JobCriteria(**r["criteria"]), r["jobs"], "diagnosis", r["recovery_witness"])
        discounts = .995 ** torch.arange(450, dtype=torch.float64)
        rows = []
        for i, job in enumerate(r["jobs"]):
            terminal = round(job["elapsed_s"] * 120)
            notification = (terminal + 7) // 8 - 1
            expected_credit = (35 / 12) * sum(.995 ** k for k in range(terminal // 8, 450)) if job["success"] else 0.
            observed_credit = float((d["completion_credit"][:, i].double() * discounts).sum())
            ret = float((d["job_reward"][:, i].double() * discounts).sum())
            checks[f"{arm}_{i}_credit_formula"] = abs(observed_credit - expected_credit) < .002
            checks[f"{arm}_{i}_full_return"] = abs(ret - r["discounted_job_returns"][i]) < 1e-5
            checks[f"{arm}_{i}_reward_ledger"] = torch.allclose(d["job_reward"][:, i], d["upstream_job_reward"][:, i] + d["completion_credit"][:, i], atol=1e-6, rtol=0)
            event = d["completion_credit"][:, i].clone()
            event[notification] = 0
            checks[f"{arm}_{i}_single_completion_event"] = bool((event == 0).all()) and (job["success"] or observed_credit == 0)
            rows.append({"case_id": r["cases"][i]["case_id"], "success": job["success"], "outcome": job["outcome"],
                         "original_return": original["discounted_job_returns"][i], "completion_credit_return": observed_credit,
                         "corrected_return": ret})
        corrected[arm] = rows
        results[arm] = {"manifest_sha256": sha256(args.run / arm / "manifest.json"), "cost": r["cost"], "metrics": metrics([r]), "cases": rows}
    for a, b in zip(corrected["retry"], corrected["deterministic"], strict=True):
        if a["success"] and not b["success"]:
            paired.append({"case_id": a["case_id"], "corrected_retry_minus_deterministic": a["corrected_return"] - b["corrected_return"],
                           "original_retry_minus_deterministic": a["original_return"] - b["original_return"]})
    checks["all_19_matched_successes_now_outrank_failures"] = len(paired) == 19 and all(p["corrected_retry_minus_deterministic"] > 0 for p in paired)
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed", "research_result": False,
              "registration_sha256": sha256(ROOT / "configs/completion_credit_v3.json"),
              "checks": checks, "arms": results, "paired_reward_comparison": paired,
              "summary": {"physical_traces_bitwise_unchanged": all(v for k, v in checks.items() if "_same_" in k),
                          "charged_transitions": sum(r["cost"]["charged_control_equivalent_transitions"] for r in results.values()),
                          "original_incentive_counterexamples": sum(p["original_retry_minus_deterministic"] < 0 for p in paired),
                          "corrected_incentive_counterexamples": sum(p["corrected_retry_minus_deterministic"] < 0 for p in paired)},
              "scope_and_limitations": ["Reward-only development control reruns; all old checkpoints and script actions unchanged.",
                  "Exact physical, sensor, action, reward-ledger and finite-job outcome parity against all v2 diagnostic cases.",
                  "The algebraic ideal-success credit removes these observed early-completion counterexamples; global reward alignment is not established.",
                  "No learned-policy improvement is inferred from relabeling rewards. A fresh learning run is required."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "summary": result["summary"], "failed_checks": [k for k,v in checks.items() if not v]}, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
