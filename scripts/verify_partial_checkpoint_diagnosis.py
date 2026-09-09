"""Verify all conditional extraction arms, including full returns and matched RNG."""
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
    for arm in ("deterministic", "stochastic", "clipped_mean", "retry"):
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
                         "witnessed_recovery": witness["witnessed_complete_recovery"]})
        arms[arm] = {"manifest_sha256": sha256(path / "manifest.json"), "checkpoint_sha256": r["checkpoint_sha256"],
                     "cost": r["cost"], "metrics": metrics([r]), "cases": rows}
    reference = traces["deterministic"]
    checks["deterministic_requests_equal_gaussian_means"] = torch.equal(reference["requested_action"], reference["mean_action"])
    registration = json.loads((ROOT / "configs/partial_checkpoint_diagnosis_v2.json").read_text())
    checks["exact_registered_partial_checkpoint"] = arms["deterministic"]["checkpoint_sha256"] == registration["checkpoint_sha256"]
    for arm in ("stochastic", "clipped_mean", "retry"):
        data = traces[arm]
        checks[arm + "_same_initial_state"] = reference["initial"].keys() == data["initial"].keys() and all(
            torch.equal(value, data["initial"][name]) for name, value in reference["initial"].items())
        for name in ("sensor_draws", "cuda_rng_before_steps", "dead_zone"):
            checks[arm + "_same_" + name] = torch.equal(reference[name], data[name])
        checks[arm + "_same_checkpoint"] = arms[arm]["checkpoint_sha256"] == arms["deterministic"]["checkpoint_sha256"]
    stochastic = traces["stochastic"]
    returns = torch.zeros_like(stochastic["job_reward"], dtype=torch.float64)
    carry = torch.zeros(28, dtype=torch.float64)
    for t in reversed(range(450)):
        carry = stochastic["job_reward"][t].double() + .995 * carry
        returns[t] = carry
    valid = stochastic["active"].bool()
    prediction = stochastic["critic_value"].double()[valid]
    target = returns[valid]
    error = prediction - target
    calibration = {"active_state_rows": int(valid.sum()), "mean_prediction": float(prediction.mean()),
                   "mean_return_to_go": float(target.mean()), "bias": float(error.mean()),
                   "root_mean_squared_error": float(error.square().mean().sqrt()),
                   "explained_variance": float(1 - error.var(unbiased=False) / target.var(unbiased=False)) if float(target.var(unbiased=False)) > 0 else None,
                   "initial_mean_prediction": float(stochastic["critic_value"][0].mean()),
                   "initial_mean_realized_return": float(returns[0].mean()),
                   "scope": "Frozen stochastic policy on 28 development cases; correlated state rows and noisy Monte Carlo targets, not independent samples or proof of the source of value error."}
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed", "research_result": False,
              "training_run_status": "failed_incomplete", "checkpoint_selection": "Last durable partial cohort 13; not a complete-budget policy",
              "registration_sha256": sha256(ROOT / "configs/partial_checkpoint_diagnosis_v2.json"), "checks": checks, "arms": arms, "stochastic_critic_calibration": calibration,
              "charged_transitions": sum(v["cost"]["charged_control_equivalent_transitions"] for v in arms.values()),
              "scope_and_limitations": ["Four distinct controllers on every registered seed-10070 development case, with exact initial state and random draws.",
                  "The v3 training run remains failed/incomplete. These partial-checkpoint diagnostic scores cannot replace a full-budget evaluation or pass its competence gate.",
                  "Expected clipped action is an extraction diagnostic, not a scripted motion planner, new training arm or expected simulator motion.",
                  "One development seed cannot establish robust competence; any changed extraction requires a new shared protocol and all-seed confirmation.",
                  "No final-test cases, geometry, force criterion, dwell, reward, reset semantics or trained parameters changed."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "metrics": {k: v["metrics"]["all"] for k, v in arms.items()},
                      "critic_calibration": calibration, "failed_checks": [k for k, v in checks.items() if not v]}, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
