"""Verify a saved scripted pair and compute contact witnesses and full job returns."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly_recovery.evaluation import JobCriteria, JobEvaluator, PhysicsSample  # noqa: E402
from assembly_recovery.protocol import sha256  # noqa: E402
from assembly_recovery.retry_controller import ActorRetryController, RetrySettings  # noqa: E402
from assembly_recovery.reward_ledger import discounted_job_return  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def verify_run(directory):
    manifest = read_json(directory / "manifest.json")
    report = read_json(directory / "probe/report.json")
    if manifest["status"] != "completed" or report["status"] != "completed":
        raise ValueError("Failed/incomplete process cannot become a completed comparison")
    for artifact in manifest["artifacts"]:
        if sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Artifact hash mismatch: " + artifact["path"])
    with zipfile.ZipFile(directory / "source.zip") as archive:
        for name, digest in manifest["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Source archive mismatch: " + name)
    controls = [json.loads(line) for line in (directory / "probe/control_samples.jsonl").read_text().splitlines()]
    physics = {job["job_id"]: [] for job in report["jobs"]}
    for line in (directory / "probe/physics_samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        physics[row["job_id"]].append(row)
    criteria = JobCriteria(**report["criteria"])
    for i, job in enumerate(report["jobs"]):
        evaluator = JobEvaluator(job["job_id"], criteria)
        for row in physics[job["job_id"]]:
            sample = {k: v for k, v in row.items() if k not in {"job_id", "peg_fixture_contact_force_n"}}
            evaluator.observe(PhysicsSample(**sample))
        evaluator.finish("initialization_invalid" if job["outcome"] == "initialization_invalid" else "probe_end")
        if evaluator.result() != job:
            raise ValueError("Reloaded physics does not reproduce job")
        if job["forbidden_events"] or report["automatic_resets"] or report["initializations_during_jobs"]:
            raise ValueError("Forbidden recovery mutation")
        returns = discounted_job_return(controls, i, evaluator.last_step, 8)
        if returns != report["returns"][i]:
            raise ValueError("Reloaded return does not reproduce report")
        controller_record = report["controllers"][i]
        controller = ActorRetryController(report["initial_state"]["actor_observation"][i], step_dt=8 * criteria.physics_dt,
                                          position_bounds=[report["geometry"]["position_action_bound_m"]] * 3,
                                          seated_height_m=report["geometry"]["seated_fingertip_above_hole_top_m"],
                                          retry=report["controller"] == "scripted_retry",
                                          settings=RetrySettings(**controller_record["settings"]))
        for row in controls:
            action, state = controller.act(row["actor_observation_before"][i], (row["step"] - 1) * 8 * criteria.physics_dt)
            if max(abs(a - b) for a, b in zip(action, row["commanded_action"][i], strict=True)) > 1e-7:
                raise ValueError("Controller cannot be replayed from actor measurements alone")
            if state != row["controller_state"][i]:
                raise ValueError("Controller internal state replay mismatch")
        replayed = controller.report()
        # Old records predate optional settings; compare their recorded fields.
        replayed["settings"] = {k: replayed["settings"][k] for k in controller_record["settings"]}
        if replayed != controller_record:
            raise ValueError("Controller event replay mismatch")
    return manifest, report, controls, physics


def recovery_witness(report, controls, physics, index):
    job = report["jobs"][index]
    rows = physics[job["job_id"]]
    criteria = report["criteria"]
    dt = criteria["physics_dt"]
    trigger = report["controllers"][index]["trigger"]
    result = {"witnessed_complete_recovery": False, "trigger": trigger}
    if not trigger or job["first_stall_step"] is None:
        return result
    stall_step = job["first_stall_step"]
    window = rows[max(0, stall_step - round(criteria["stall_window_s"] / dt)):stall_step]
    occupancy = sum(row["peg_fixture_contact_force_n"] > 0.1 for row in window) / max(1, len(window))
    withdrawing = [row for row in controls if row["phase"][index] == "withdraw" and row["active_before_step"][index]]
    before = [row for row in controls if row["step"] * 8 * dt <= trigger["time_s"]]
    rise, clearance, clear_seconds = 0.0, None, 0.0
    if withdrawing and before:
        rise = max(row["part_pos"][index][2] for row in withdrawing) - before[-1]["part_pos"][index][2]
        clearance = withdrawing[-1]["part_pos"][index][2] - report["initial_fixed_pos"][index][2] - report["geometry"]["hole_height_m"]
        start, end = withdrawing[0]["step"] * 8, withdrawing[-1]["step"] * 8
        consecutive = 0
        for row in rows:
            if start <= row["step"] <= end:
                consecutive = consecutive + 1 if row["peg_fixture_contact_force_n"] <= 0.1 else 0
                clear_seconds = max(clear_seconds, consecutive * dt)
    witnessed = (job["success"] and job["recovered_after_stall"] and stall_step * dt <= trigger["time_s"]
                 and occupancy >= 0.8 and rise >= 0.005 and clearance is not None and clearance > 0.001
                 and clear_seconds >= 0.2 and not job["forbidden_events"])
    result.update(witnessed_complete_recovery=witnessed, evaluator_first_stall_s=stall_step * dt,
                  fixture_contact_fraction_in_stall_window=occupancy, withdrawn_part_rise_m=rise,
                  withdrawal_final_base_clearance_m=clearance, withdrawal_longest_fixture_clear_s=clear_seconds,
                  outcome=job["outcome"], retained_grasp_to_termination=job["outcome"] != "lost_grasp")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--continued", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.recovery = args.recovery.resolve()
    args.continued = args.continued.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Comparison output already exists; preserve old evidence")
    recovery = verify_run(args.recovery)
    continued = verify_run(args.continued)
    rm, rr, rc, rp = recovery
    cm, cr, cc, _ = continued
    if rr["controller"] != "scripted_retry" or cr["controller"] != "scripted_continue":
        raise ValueError("Expected scripted retry and continued insertion")
    identical_initial = rr["initial_state"] == cr["initial_state"]
    same_conditions = all(rr[key] == cr[key] for key in ("criteria", "geometry", "held_part_gravity", "velocity_mode", "simulated_seconds", "seed"))
    same_conditions &= (args.recovery / "probe/environment.yaml").read_bytes() == (args.continued / "probe/environment.yaml").read_bytes()
    behavior_files = ("scripts/validate_peg.py", "src/assembly_recovery/peg_env.py", "src/assembly_recovery/retry_controller.py",
                      "src/assembly_recovery/evaluation.py", "src/assembly_recovery/geometry.py", "src/assembly_recovery/reward_ledger.py", "configs/study.json")
    if rr.get("fault_cases") is not None or cr.get("fault_cases") is not None:
        same_conditions &= rr.get("fault_cases") == cr.get("fault_cases")
        behavior_files += ("src/assembly_recovery/faults.py", "src/assembly_recovery/fault_env.py")
    same_source = all(rm["source_hashes"][name] == cm["source_hashes"][name] for name in behavior_files)
    same_settings = [x["settings"] for x in rr["controllers"]] == [x["settings"] for x in cr["controllers"]]
    same_upstream = rm["upstream_source_sha256"] == cm["upstream_source_sha256"] and rm["upstream_commit"] == cm["upstream_commit"]
    pairs = []
    for i, (rjob, cjob) in enumerate(zip(rr["jobs"], cr["jobs"], strict=True)):
        trigger = rr["controllers"][i]["trigger"]
        prefix_steps = round(trigger["time_s"] / (8 * rr["criteria"]["physics_dt"])) if trigger else 0
        common_prefix = all(a["part_pos"][i] == b["part_pos"][i] and a["actor_observation_before"][i] == b["actor_observation_before"][i]
                            and a["commanded_action"][i] == b["commanded_action"][i]
                            for a, b in zip(rc[:prefix_steps], cc[:prefix_steps], strict=True))
        pairs.append({"job_index": i, "recovery": rjob, "continued": cjob,
                      "initial_and_pretrigger_trajectory_identical": identical_initial and common_prefix,
                      "identical_prefix_control_steps": prefix_steps,
                      "recovery_witness": recovery_witness(rr, rc, rp, i),
                      "recovery_return": rr["returns"][i], "continued_return": cr["returns"][i],
                      "discounted_return_difference_recovery_minus_continued": rr["returns"][i]["discounted_return"] - cr["returns"][i]["discounted_return"]})
    verified = identical_initial and same_conditions and same_source and same_settings and same_upstream
    verified &= all(row["initial_and_pretrigger_trajectory_identical"] for row in pairs)
    result = {"schema": 1, "status": "verified" if verified else "pairing_failed", "research_result": False,
              "scope_and_limitations": ["Development scripted feasibility and reward comparison; no learned policy, training advantage, or held-out result.",
                                        "Synthetic noisy simulator measurements are not camera perception; preset grasp is not learned pickup.",
                                        "Official FORGE reward is unchanged, including cohort-dependent success-prediction activation.",
                                        "Thirty-second common deadline; stop job returns at evaluator termination with later absorbing-zero rewards. Partial terminal control interval has no sampled reward and is omitted.",
                                        "Raw wrist force budget is an experimental abort threshold, not a hard physical force ceiling."],
              "checks": {"identical_initial_state": identical_initial, "same_conditions": same_conditions,
                         "same_behavior_source": same_source, "same_controller_settings": same_settings, "same_upstream": same_upstream,
                         "saved_source_and_artifact_hashes_verified": True, "physics_and_actor_controller_replayed": True},
              "runs": [{"run_id": m["run_id"], "manifest": str(d.relative_to(ROOT) / "manifest.json"),
                        "manifest_sha256": sha256(d / "manifest.json"), "source_commit_at_start": m["source_commit_at_start"],
                        "source_dirty_at_start": m["source_dirty_at_start"], "artifacts": m["artifacts"]}
                       for m, d in ((rm, args.recovery), (cm, args.continued))],
              "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "pairs": [{"job": p["job_index"], "retry": p["recovery"]["outcome"], "continued": p["continued"]["outcome"],
                                 "witnessed_recovery": p["recovery_witness"]["witnessed_complete_recovery"],
                                 "return_difference": p["discounted_return_difference_recovery_minus_continued"]} for p in pairs]}, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
