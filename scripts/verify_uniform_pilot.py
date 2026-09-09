"""Verify and summarize a budget-selected uniform pilot and all development jobs."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from verify_training_contract import verify_assets

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.evaluation import JobCriteria  # noqa: E402
from assembly_recovery.protocol import sha256  # noqa: E402
from assembly_recovery.study_ppo import StudyPolicy  # noqa: E402
from scripts.probe_training import replay_physics  # noqa: E402


def metrics(cohorts):
    grouped = {}
    for cohort in cohorts:
        for case, job, witness, ret in zip(cohort["cases"], cohort["jobs"], cohort["recovery_witness"], cohort["discounted_job_returns"], strict=True):
            for key in ("all", case["bin_id"]):
                grouped.setdefault(key, []).append((job, witness, ret))
    result = {}
    for key, rows in grouped.items():
        result[key] = {"requests": len(rows), "completions": sum(j["success"] for j, _, _ in rows),
                       "outcomes": dict(Counter(j["outcome"] for j, _, _ in rows)),
                       "upstream_success_seen": sum(j["upstream_success_seen"] for j, _, _ in rows),
                       "witnessed_contact_failures": sum(w["witnessed_failure_step"] is not None for _, w, _ in rows),
                       "witnessed_recoveries": sum(w["witnessed_complete_recovery"] for _, w, _ in rows),
                       "mean_time_per_request_s": statistics.mean(j["elapsed_s"] for j, _, _ in rows),
                       "peak_raw_wrist_force_n": max(j["peak_raw_wrist_force_n"] for j, _, _ in rows),
                       "mean_discounted_return": statistics.mean(r for _, _, r in rows),
                       "forbidden_events": sum(len(j["forbidden_events"]) for j, _, _ in rows),
                       "active_control_transitions": sum(math.ceil(j["elapsed_s"] * 15 - 1e-8) for j, _, _ in rows),
                       "return_by_outcome": {outcome: {"requests": sum(j["outcome"] == outcome for j, _, _ in rows),
                           "mean": statistics.mean(ret for j, _, ret in rows if j["outcome"] == outcome)}
                           for outcome in sorted({j["outcome"] for j, _, _ in rows})}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence")
    tm = verify_assets(args.training)
    tr = json.loads((args.training / "pilot/report.json").read_text())
    checkpoint_path = args.training / "pilot/budget_checkpoint.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = ckpt["spec"]
    model = StudyPolicy(spec["actor_size"], spec["critic_size"], spec["action_size"], spec["widths"])
    model.load_state_dict(ckpt["model"], strict=True)
    checks = {"training_process_complete": tm["status"] == tr["status"] == "completed",
              "complete_budget_checkpoint": ckpt["status"] == "budget_complete",
              "finite_checkpoint": all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
              "charged_budget_exact": ckpt["cost"] == tr["cost"] and tr["cost"]["charged_control_equivalent_transitions"] == tm["specification"]["expected_charged_control_equivalent_transitions"],
              "predeclared_cohorts": len(tr["cohorts"]) == tm["specification"]["cohorts"],
              "twenty_update_boundaries": len(tr["optimizer_updates"]) == 20,
              "all_training_requests_recorded": sum(len(c["jobs"]) for c in tr["cohorts"]) == 5120,
              "nominal_share_exact": all(sum(c["bin_id"] == "nominal" for c in item["cases"]) * 4 == len(item["cases"]) for item in tr["cohorts"]),
              "all_training_initializations_valid": all(v["valid"] for c in tr["cohorts"] for v in c["initialization"])}
    evaluations, seeds, cohorts, diagnostic_rows = [], [], [], []
    for p in args.evaluation:
        m = verify_assets(p)
        r = json.loads((p / "pilot/report.json").read_text())
        seeds.append(r["seed"])
        checks[p.name + "_completed"] = m["status"] == r["status"] == "completed"
        checks[p.name + "_same_checkpoint"] = r["checkpoint_sha256"] == sha256(checkpoint_path) == m["input_checkpoint"]["sha256"]
        checks[p.name + "_same_protocol"] = m["protocol_sha256"] == tm["protocol_sha256"]
        frozen = json.loads((ROOT / json.loads((ROOT / "configs/study.json").read_text())["training"]["pilot_protocol"]).read_text())
        checks[p.name + "_same_behavior"] = all(m["source_hashes"][k] == tm["source_hashes"][k] == v for k, v in frozen["behavior_source_sha256"].items())
        checks[p.name + "_all_requests"] = len(r["cohorts"]) == 1 and len(r["cohorts"][0]["jobs"]) == 28
        checks[p.name + "_all_initializations_valid"] = all(v["valid"] for c in r["cohorts"] for v in c["initialization"])
        for c in r["cohorts"]:
            data = torch.load(p / f"pilot/diagnostics_{c['cohort']}.pt", map_location="cpu", weights_only=True)
            replay_physics(data["physics"], JobCriteria(), c["jobs"], f"evaluate-{r['seed']}-{c['cohort']}", c["recovery_witness"])
            for index, (case, job, witness) in enumerate(zip(c["cases"], c["jobs"], c["recovery_witness"], strict=True)):
                end = round(job["elapsed_s"] * 120)
                native = data["physics"][:end, index]
                controls = data["controls"][:, index]
                active = controls[:, 43].bool()
                start = witness["witnessed_failure_step"]
                maximum_rise = None if start is None else float((native[start - 1:, 12] - native[start - 1, 12]).max())
                diagnostic_rows.append({"case_id": case["case_id"], "outcome": job["outcome"],
                    "elapsed_s": job["elapsed_s"], "fixture_contact_seen": bool((native[:, 11] > 0.1).any()),
                    "witnessed_failure_step": start, "maximum_rise_after_contact_failure_m": maximum_rise,
                    "witnessed_withdrawal_step": witness["witnessed_withdrawal_step"],
                    "active_applied_action_z_min_max": [float(controls[active, 12].min()), float(controls[active, 12].max())],
                    "last_action_target_xyz_above_estimated_fixture_top_m": (controls[active][-1, 10:13] * 0.05).tolist(),
                    "last_held_base_height_above_fixture_origin_m": float(native[-1, 13] + 0.025),
                    "minimum_held_base_height_above_fixture_origin_m": float(native[:, 13].min() + 0.025),
                    "fixture_contact_fraction": float((native[:, 11] > 0.1).float().mean()),
                    "peak_force_n": job["peak_raw_wrist_force_n"], "upstream_success_seen": job["upstream_success_seen"]})
        cohorts.extend(r["cohorts"])
        evaluations.append({"run_id": m["run_id"], "manifest_sha256": sha256(p / "manifest.json"), "seed": r["seed"], "cost": r["cost"], "metrics": metrics(r["cohorts"])})
    checks["all_development_seeds_once"] = sorted(seeds) == [10070, 10071, 10072]
    resources = json.loads((args.training / "gpu_resources.json").read_text())
    intervals = tr["throughput_intervals"]
    result = {"schema": 1, "status": "verified" if all(checks.values()) else "check_failed", "research_result": False,
              "controller": tr["controller"], "checks": checks,
              "training": {"run_id": tm["run_id"], "manifest_sha256": sha256(args.training / "manifest.json"),
                           "checkpoint_sha256": sha256(checkpoint_path), "protocol_sha256": tm["protocol_sha256"],
                           "cost": tr["cost"], "process_wall_s": tm["elapsed_wall_s"],
                           "cohort_metrics": [metrics([c]) for c in tr["cohorts"]],
                           "rollout_rate_min_max": [min(x["transitions_per_s"] for x in intervals), max(x["transitions_per_s"] for x in intervals)],
                           "end_to_end_rollout_transitions_per_s": tr["cost"]["rollout_control_transitions"] / tm["elapsed_wall_s"],
                           "peak_gpu_mib": resources["peak_used_mib"], "minimum_free_ram_mib": resources["minimum_free_ram_mib"],
                           "optimizer_wall_s": sum(x["wall_s"] for x in tr["optimizer_updates"]),
                           "initialization_wall_s": sum(c["initialization_wall_s"] for c in tr["cohorts"]),
                           "diagnostic_export_wall_s": sum(c["diagnostic_export_wall_s"] for c in tr["cohorts"])},
              "development": {"metrics": metrics(cohorts), "runs": evaluations, "failure_diagnostics": diagnostic_rows},
              "scope_and_limitations": [
                  "One training seed, five changing-policy training cohorts and 84 fixed development requests; not a final-test estimate or a method comparison.",
                  "Training stochastic outcomes are not evaluation scores. No previous development weights initialized this pilot.",
                  "General contact witness is distinct from the historical scripted-phase witness. Prevention and unobserved failure do not count as witnessed recovery.",
                  "Large absorbing fractions consume simulator cost while supplying no new learning samples; throughput counts all slots.",
                  "The prior scripted matrix has different controller/RNG execution and is not a paired learning comparison.",
                  "Gear, fixed/adaptive curricula, multiple training seeds, held-out cases and learned-policy physics refinement remain unverified."]}
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "development": result["development"]["metrics"]}, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
