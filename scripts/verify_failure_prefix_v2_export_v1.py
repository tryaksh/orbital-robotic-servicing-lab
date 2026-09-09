"""Export correction: compare resolved frozen settings and normalize local paths."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.evaluation import JobCriteria  # noqa: E402
from assembly_recovery.faults import development_cases  # noqa: E402
from assembly_recovery.post_stall import post_stall_completion, summarize_endpoints  # noqa: E402
from assembly_recovery.protocol import sha256, write_json  # noqa: E402
from assembly_recovery.retry_controller import ActorRetryController, RetrySettings  # noqa: E402
from assembly_recovery.study_ppo import StudyPolicy  # noqa: E402
from scripts.probe_training import replay_physics  # noqa: E402
from scripts.verify_training_contract import verify_assets  # noqa: E402
from scripts.verify_uniform_pilot import metrics  # noqa: E402


def verify_script_actions(data, report):
    """Replay the frozen controller from its observations, with no physics calls."""
    controllers = [ActorRetryController(row, step_dt=report["controller_step_dt"],
                   position_bounds=report["controller_position_bounds"],
                   seated_height_m=report["controller_seated_height_m"],
                   retry=report["controller"] == "retry", settings=RetrySettings(**report["controller_settings"]))
                   for row in data["actor_before"][0].tolist()]
    steps = 60 if report["controller"] == "learned" else 450
    for step in range(steps):
        decisions = [ctrl.act(row, step * report["controller_step_dt"]) for ctrl, row in
                     zip(controllers, data["actor_before"][step].tolist(), strict=True)]
        expected = torch.tensor([d[0] for d in decisions], dtype=data["requested_action"].dtype)
        if not torch.equal(expected, data["requested_action"][step]):
            return False
        if [d[1]["phase"] for d in decisions] != report["script_phases"][step]:
            return False
    return True


def verify_rewards(data, report):
    discounts = .995 ** torch.arange(450, dtype=torch.float64)
    for i, job in enumerate(report["jobs"]):
        terminal = round(job["elapsed_s"] * 120)
        included, notification = terminal // 8, (terminal + 7) // 8 - 1
        ledger = sum(float((v[:included, i].double() * discounts[:included]).sum())
                     for v in data["reward_terms"].values())
        credit = float((data["completion_credit"][:, i].double() * discounts).sum())
        ret = float((data["job_reward"][:, i].double() * discounts).sum())
        expected_credit = (35 / 12) * .995 ** included * (1 - .995 ** (450 - included)) / (1 - .995) if job["success"] else 0.
        if (abs(ledger + credit - ret) >= .001 or abs(ret - report["discounted_job_returns"][i]) >= 1e-5
                or abs(credit - expected_credit) >= .002
                or not bool((data["job_reward"][notification + 1:, i] == 0).all())):
            return False
    return True


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return all(finite_tree(v) for v in value.values()) if isinstance(value, dict) else True


def paired_counts(left, right, *, conditional):
    if [r["case_id"] for r in left] != [r["case_id"] for r in right]:
        raise ValueError("Pair jobs by identical case identity and order")
    if conditional and [r["eligible_at_handoff"] for r in left] != [r["eligible_at_handoff"] for r in right]:
        raise ValueError("Conditional comparison requires the same pre-handoff cohort")
    pairs = [(a, b) for a, b in zip(left, right, strict=True) if not conditional or a["eligible_at_handoff"]]
    key = "post_stall_completion" if conditional else "success"
    return {"requests": len(pairs), "both_complete": sum(a[key] and b[key] for a, b in pairs),
            "learned_only": sum(a[key] and not b[key] for a, b in pairs),
            "control_only": sum(not a[key] and b[key] for a, b in pairs),
            "neither_complete": sum(not a[key] and not b[key] for a, b in pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.output.exists() or (args.run / "verification_resolved_settings_v1").exists():
        parser.error("Preserve existing evidence and verification artifacts")
    torch.set_num_threads(1)
    registration_path = ROOT / "configs/failure_prefix_confirmation_v2.json"
    registration = json.loads(registration_path.read_text())
    protocol = json.loads((ROOT / registration["physical_protocol"]["path"]).read_text())
    study = json.loads((ROOT / "configs/study.json").read_text())
    queue = json.loads((args.run / "queue.json").read_text())
    checkpoint = ROOT / registration["checkpoint_path"]
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = loaded["spec"]
    model = StudyPolicy(spec["actor_size"], spec["critic_size"], spec["action_size"], spec["widths"])
    model.load_state_dict(loaded["model"], strict=True)
    model.eval()
    checks = {"queue_completed": queue["status"] == "completed" and len(queue["runs"]) == 6,
              "registered_checkpoint": sha256(checkpoint) == registration["checkpoint_sha256"],
              "complete_budget_model": loaded["status"] == "budget_complete" and loaded["completed_cohorts"] == 22,
              "queue_registration": queue["registration_sha256"] == sha256(registration_path),
              "original_gate_still_failed": not json.loads((ROOT / "evidence/uniform_unclipped_v4.json").read_text())["competence_gate_passed"],
              "prior_evidence_unchanged": all(sha256(ROOT / p) == h for p, h in registration["prior_evidence_sha256"].items()),
              "registered_source_unchanged": all(sha256(ROOT / p) == h for p, h in registration["diagnostic_source_sha256"].items())}
    by_seed, raw, reports_by_arm, rows_by_arm, resources = {}, {}, {}, {}, []
    for seed in registration["seeds"]:
        traces, summaries, per_arm = {}, {}, {}
        for arm in registration["controllers"]:
            label = f"{seed}_{arm}"
            path = args.run / label
            manifest = verify_assets(path)
            report = json.loads((path / "diagnosis/report.json").read_text())
            data = torch.load(path / "diagnosis/trajectory.pt", weights_only=True, map_location="cpu")
            traces[arm] = data
            local = {
                "completed": manifest["status"] == report["status"] == "completed" and manifest["returncode"] == 0 and not manifest["timed_out"],
                "seed_cases": report["seed"] == manifest["seed"] == seed and report["cases"] == development_cases(study, seed),
                "all_requests": len(report["jobs"]) == len(report["initialization"]) == 28 and all(v["valid"] for v in report["initialization"]),
                "frozen_criteria": report["criteria"] == asdict(JobCriteria()),
                "checkpoint": report["checkpoint_sha256"] == manifest["input_checkpoint"]["sha256"] == registration["checkpoint_sha256"],
                "provenance": manifest["registration_sha256"] == sha256(registration_path)
                    and manifest["protocol_sha256"] == registration["physical_protocol"]["sha256"]
                    and all(manifest["source_hashes"][p] == h for p, h in {**protocol["behavior_source_sha256"], **registration["diagnostic_source_sha256"]}.items()),
                "preregistered": datetime.fromisoformat(registration["created_at_utc"]) < datetime.fromisoformat(manifest["started_at_utc"]),
                "exact_cost": report["cost"]["charged_control_equivalent_transitions"] == 12834.5,
                "finite_arrays": finite_tree(data),
                "native_shape": tuple(data["physics"].shape) == (3600, 28, 14),
                "unchanged_script_settings": report["controller_settings"] == asdict(RetrySettings(**json.loads((ROOT / "configs/retry_unload_realign_v1.json").read_text())))
                    and report["controller_step_dt"] == 1 / 15 and report["controller_position_bounds"] == [.05] * 3,
                "script_actions_and_phases": verify_script_actions(data, report),
                "reward_return_credit_absorbing": verify_rewards(data, report),
                "no_forbidden_events": all(not j["forbidden_events"] for j in report["jobs"]),
            }
            replay = replay_physics(data["physics"], JobCriteria(**report["criteria"]), report["jobs"], "diagnosis", report["recovery_witness"])
            local["native_physics_replay"] = replay["exact_cpu_evaluator_parity"] and replay["contact_witness_parity"]
            endpoints = [post_stall_completion(j, w, JobCriteria()) for j, w in zip(report["jobs"], report["recovery_witness"], strict=True)]
            exposure = summarize_endpoints(endpoints, report["jobs"])
            local["endpoint_replay"] = endpoints == report["post_stall_endpoint"] and exposure == report["post_stall_summary"]
            with torch.no_grad():
                means = model.distribution(model.actor_norm(data["actor_before"])).mean
            mean_error = float((means - data["mean_action"]).abs().max())
            local["frozen_mean_cpu_replay"] = mean_error <= registration["verification"]["cpu_mean_absolute_tolerance"]
            if arm == "learned":
                local["learned_actions"] = torch.equal(data["requested_action"][60:], data["mean_action"][60:])
                local["learned_phase_labels"] = all(row == ["learned_mean"] * 28 for row in report["script_phases"][60:])
            checks.update({label + "_" + k: v for k, v in local.items()})
            rows = []
            for i, (case, job, endpoint) in enumerate(zip(report["cases"], report["jobs"], endpoints, strict=True)):
                failure = endpoint["witnessed_failure_step"]
                end = round(job["elapsed_s"] * 120)
                rise = None if failure is None else float((data["physics"][failure - 1:end, i, 12] - data["physics"][failure - 1, i, 12]).max())
                rows.append({"case_id": case["case_id"], "bin_id": case["bin_id"], "seed": seed, "controller": arm,
                             **endpoint, "success": job["success"], "outcome": job["outcome"], "elapsed_s": job["elapsed_s"],
                             "maximum_rise_after_failure_m": rise, "peak_raw_wrist_force_n": job["peak_raw_wrist_force_n"]})
            per_arm[arm] = rows
            reports_by_arm.setdefault(arm, []).append(report)
            rows_by_arm.setdefault(arm, []).extend(rows)
            summaries[arm] = {"whole_job": metrics([report])["all"], "post_stall": exposure,
                              "post_stall_by_bin": {b: {"eligible": sum(r["eligible_at_handoff"] and r["bin_id"] == b for r in rows),
                                  "completed": sum(r["post_stall_completion"] and r["bin_id"] == b for r in rows)} for b in sorted({r["bin_id"] for r in rows})},
                              "manifest_sha256": sha256(path / "manifest.json"), "cpu_mean_max_absolute_error": mean_error}
            resource = json.loads((path / "gpu_resources.json").read_text())
            resources.append({"seed": seed, "controller": arm, "wall_s": manifest["elapsed_wall_s"], "cost": report["cost"],
                              "peak_gpu_mib": resource["peak_used_mib"], "minimum_free_ram_mib": min((s["free_ram_mib"] for s in resource["samples"] if s["free_ram_mib"] is not None), default=None)})
        reference = traces["learned"]
        for arm in ("retry", "continue"):
            data = traces[arm]
            prefix = f"{seed}_{arm}_paired_"
            checks[prefix + "initial_state"] = reference["initial"].keys() == data["initial"].keys() and all(torch.equal(v, data["initial"][k]) for k, v in reference["initial"].items())
            for name in ("sensor_draws", "cuda_rng_before_steps", "dead_zone"):
                checks[prefix + name] = torch.equal(reference[name], data[name])
            for name in ("requested_action", "applied_action", "actor_before", "critic_before", "job_reward"):
                checks[prefix + "prefix_" + name] = torch.equal(reference[name][:60], data[name][:60])
            checks[prefix + "native_prefix"] = torch.equal(reference["physics"][:480], data["physics"][:480])
            checks[prefix + "failure_cohort"] = [r["case_id"] for r in per_arm["learned"] if r["eligible_at_handoff"]] == [r["case_id"] for r in per_arm[arm] if r["eligible_at_handoff"]]
        by_seed[str(seed)] = {"arms": summaries, "paired": {arm: {"whole_job": paired_counts(per_arm["learned"], per_arm[arm], conditional=False),
            "post_stall": paired_counts(per_arm["learned"], per_arm[arm], conditional=True)} for arm in ("retry", "continue")}}
        raw[str(seed)] = per_arm
    aggregates = {}
    for arm, reports in reports_by_arm.items():
        jobs = [j for r in reports for j in r["jobs"]]
        grouped = metrics(reports)
        aggregates[arm] = {"whole_job": grouped["all"], "nominal": grouped["nominal"],
                           "post_stall": summarize_endpoints(rows_by_arm[arm], jobs)}
    charged = sum(r["cost"]["charged_control_equivalent_transitions"] for r in resources)
    checks["six_run_budget"] = charged == registration["expected_charged_transitions"] == 77007.
    thresholds = registration["readiness_for_physics_refinement"]
    per_seed_ready = {}
    for seed, item in by_seed.items():
        learned, continued = item["arms"]["learned"], item["arms"]["continue"]
        per_seed_ready[seed] = (
            learned["post_stall"]["post_stall_completions"] >= thresholds["minimum_completions_per_seed"]
            and sum(b != "nominal" and v["completed"] > 0 for b, v in learned["post_stall_by_bin"].items()) >= thresholds["minimum_fault_bins_per_seed"]
            and learned["post_stall"]["post_stall_completions"] > continued["post_stall"]["post_stall_completions"])
    verified = all(checks.values())
    raw_output = args.run / "verification_resolved_settings_v1"
    raw_output.mkdir(exist_ok=False)
    write_json(raw_output / "checks.json", checks)
    write_json(raw_output / "cases.json", raw)
    result = {"schema": 1, "status": "verified" if verified else "check_failed", "research_result": False,
              "registration_sha256": sha256(registration_path), "checkpoint_sha256": sha256(checkpoint),
              "verification": {"checks": len(checks), "failed_checks": [k for k, v in checks.items() if not v],
                  "artifacts": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in sorted(raw_output.iterdir())]},
              "confirmation_seeds": registration["seeds"], "by_seed": by_seed, "aggregate_confirmation_only": aggregates,
              "paired_confirmation_only": {arm: {"whole_job": paired_counts(rows_by_arm["learned"], rows_by_arm[arm], conditional=False),
                  "post_stall": paired_counts(rows_by_arm["learned"], rows_by_arm[arm], conditional=True)} for arm in ("retry", "continue")},
              "charged_transitions": charged, "resources": resources,
              "decision": {"eligible_for_bounded_learned_physics_refinement": verified and all(per_seed_ready.values()),
                  "per_seed_behavior_confirmation": per_seed_ready, "original_competence_gate_passed": False,
                  "adaptive_campaign_ready": False,
                  "reason": "Physics refinement is a separate validation step. Learned physics robustness and a registered shared uniform/adaptive/fixed-curriculum configuration remain required before a campaign."},
              "scope_and_limitations": registration["scope_and_limitations"]}
    with args.output.open("x", encoding="utf8", newline="\n") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "failed_checks": result["verification"]["failed_checks"],
                      "aggregate_confirmation_only": aggregates, "decision": result["decision"]}, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
