"""Verify the preserved pilot failure and bounded initial-action correction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from verify_training_contract import verify_assets
from verify_training_paths import verified_run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from assembly_recovery.protocol import sha256  # noqa: E402
from assembly_recovery.study_ppo import StudyPolicy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence")
    base = ROOT / "artifacts/assembly"
    old = base / "initialization_failure_diagnostic_v1"
    new = base / "initialization_projection_diagnostic_v2"
    pilot = base / "uniform_pilot_170_v1"
    om, nm, pm = [verify_assets(p) for p in (old, new, pilot)]
    a, b = [torch.load(p / "probe/initialization.pt", map_location="cpu", weights_only=True) for p in (old, new)]
    ar, br = [json.loads((p / "probe/report.json").read_text()) for p in (old, new)]
    pr = json.loads((pilot / "pilot/report.json").read_text())
    checks = {"diagnostic_processes_completed": om["status"] == nm["status"] == "completed",
              "same_rng_checkpoint": om["checkpoint_sha256"] == nm["checkpoint_sha256"],
              "same_physical_state_and_noise": all(torch.equal(a[k], b[k]) for k in a if k != "initial_actions"),
              "exact_legal_projection": torch.equal(a["initial_actions"].clamp(-1, 1), b["initial_actions"]),
              "all_1024_corrected_initializations_valid": len(br["initialization"]) == 1024 and all(v["valid"] for v in br["initialization"]),
              "reproduced_same_failed_case": ar["invalid"][0]["index"] == 516 and len(ar["invalid"]) == 1,
              "failed_pilot_preserved_as_incomplete": pm["status"] != "completed" and pr["status"] == "failed" and not (pilot / "pilot/budget_checkpoint.pt").exists()}
    partials = []
    for path in sorted((pilot / "pilot").glob("partial_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        spec = saved["spec"]
        model = StudyPolicy(spec["actor_size"], spec["critic_size"], spec["action_size"], spec["widths"])
        model.load_state_dict(saved["model"], strict=True)
        finite = all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
        checks[path.stem + "_partial_and_finite"] = saved["status"] == "partial" and finite
        partials.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "status": saved["status"], "cost": saved["cost"], "finite": finite})
    rm, rr, ra = verified_run(base / "training_path_reference_r02")
    tm, tr, ta = verified_run(base / "training_path_tensor_r02")
    checks["corrected_pair_exact"] = rr["jobs"] == tr["jobs"] and all(torch.equal(ra[k], ta[k]) for k in ("part_pos", "part_quat", "commanded_action"))
    for old_name, new_data in (("training_path_reference_r01", ra), ("training_path_tensor_r01", ta)):
        _, _, old_data = verified_run(base / old_name)
        checks[old_name + "_physical_paths_unchanged"] = all(torch.equal(old_data[k], new_data[k]) for k in ("part_pos", "part_quat", "commanded_action"))
    cp = base / "tensor_contract_projection_32_r01"
    cm = verify_assets(cp)
    cr = json.loads((cp / "probe/report.json").read_text())
    checks["corrected_ppo_contract"] = cm["status"] == cr["status"] == "completed" and all(c["replay"]["exact_cpu_evaluator_parity"] and c["endpoint_reward_parity"] and c["terminal_observation_error"] == 0 for c in cr["cohorts"])
    checks["corrected_actor_audit"] = cr["information_probe"]["passed"] and cr["information_probe"]["privileged_input_actor_delta"] == 0
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed", "checks": checks, "research_result": False,
              "failed_pilot": {"run_id": pm["run_id"], "manifest_sha256": sha256(pilot / "manifest.json"), "status": pm["status"],
                               "error": pr["error"], "cost": pr["cost"], "declared_charged_target": pm["specification"]["expected_charged_control_equivalent_transitions"],
                               "requested_jobs": 5120, "completed_training_jobs": 2048, "invalid_initializations": 1,
                               "valid_but_canceled_with_failed_cohort": 1023, "later_unattempted_jobs": 2048,
                               "training_completions_not_evaluation": sum(j["success"] for c in pr["cohorts"] for j in c["jobs"]),
                               "witnessed_training_recoveries": sum(w["witnessed_complete_recovery"] for c in pr["cohorts"] for w in c["recovery_witness"]),
                               "partial_checkpoints": partials, "evaluated": False},
              "diagnosis": {"case_id": ar["invalid"][0]["case"]["case_id"], "fresh_diagnostic_unbounded_z_action": float(a["initial_actions"][516, 2]),
                            "projected_z_action": float(b["initial_actions"][516, 2]), "z_target_estimate_noise_m": ar["invalid"][0]["fixed_position_noise"][2],
                            "projected_slots": int((a["initial_actions"] != b["initial_actions"]).any(-1).sum()),
                            "explanation": "An unbounded Gaussian initial target-height error made the no-motion EMA history exceed the legal action box. Project finite initial action/previous-action history into that box after physical initialization. No pose, noise, geometry, force or reward change."},
              "runs": [{"run_id": m["run_id"], "manifest_sha256": sha256(p / "manifest.json")} for m, p in
                       ((om, old), (nm, new), (rm, base / rm["run_id"]), (tm, base / tm["run_id"]), (cm, cp))],
              "scope_and_limitations": [
                  "The failed pilot remains failed; its partial weights are not used to initialize the replacement pilot or called evaluated policies.",
                  "The RNG diagnostic uses a fresh simulator, not an exact restore of the failed pilot's physical trajectory. The before/after diagnostic pair itself has bitwise equal physical states/noise.",
                  "Projection is identity for already legal initial actions. The rerun reference/tensor paths verify the earlier representative paired physical paths are unchanged.",
                  "The correction applies to the shared project fault adapter and every future study arm. Upstream source remains untouched.",
                  "NaN/Inf history remains invalid and is rejected; the correction does not hide invalid physical/grasp/contact initialization.",
                  "Protocol v1 and its configuration remain preserved. A fresh v2 pilot must reach its own predeclared budget before evaluation."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
