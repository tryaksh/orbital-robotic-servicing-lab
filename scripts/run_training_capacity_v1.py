"""Plan or launch one serial, bounded 1024/2048 representative capacity probe."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.protocol import (  # noqa: E402
    LAB_COMMIT,
    assess_completion,
    sha256,
    validate_run_id,
    validate_study,
    write_json,
)
from scripts.run_experiment import git, snapshot_source  # noqa: E402
from scripts.run_training_probe import monitor_gpu, run_resource_bounded  # noqa: E402


def specification_for(registration, num_envs):
    if num_envs not in registration["num_envs"]:
        raise ValueError("Capacity v1 permits only registered 1024/2048 environment counts")
    cohorts = registration["cohorts_per_size"]
    controls = registration["controls_per_cohort"]
    init = num_envs * cohorts * registration["expected_initialization_physics_steps_per_cohort"]
    return {"seed": registration["seed"], "num_envs": num_envs, "cohorts": cohorts,
            "controls_per_cohort": controls, "requested_jobs": num_envs * cohorts,
            "rollout_control_transitions": num_envs * cohorts * controls,
            "expected_initialization_physics_env_steps": init,
            "expected_charged_transitions": num_envs * cohorts * controls + init / registration["reference_physics_steps_per_transition"],
            "work_comparison": registration["work_comparison"]}


def report_checks(report, specification, worker_dir):
    """Inspect real artifacts, full-job accounting and PPO work without Isaac/GPU."""
    cohorts, cost = report.get("cohorts", []), report.get("cost", {})
    n = specification["num_envs"]
    active = cost.get("active_control_transitions", -1)
    absorbing = cost.get("absorbing_control_transitions", -1)
    updates = report.get("optimizer_updates", [])
    intervals = report.get("throughput_intervals", [])
    exports = [item for cohort in cohorts for item in cohort.get("exports", [])]
    return {
        "report_complete": report.get("status") == "completed",
        "scope_is_capacity_only": report.get("research_result") is False and report.get("version") == "training_capacity_v1",
        "all_cohorts": len(cohorts) == specification["cohorts"],
        "all_requested_jobs": all(len(c.get("jobs", [])) == n and len(c.get("initialization", [])) == n for c in cohorts),
        "valid_initializations": all(v.get("valid") is True for c in cohorts for v in c.get("initialization", [])),
        "one_terminal_per_job": all(c.get("done_counts") == [1] * n for c in cohorts),
        "jobs_finished_without_mutation": all(j.get("finished") is True and not j.get("forbidden_events") for c in cohorts for j in c.get("jobs", [])),
        "rollout_budget": cost.get("rollout_control_transitions") == specification["rollout_control_transitions"],
        "initialization_budget": cost.get("initialization_physics_env_steps") == specification["expected_initialization_physics_env_steps"],
        "charged_budget": cost.get("charged_control_equivalent_transitions") == specification["expected_charged_transitions"],
        "active_absorbing_accounting": active >= 0 and absorbing >= 0 and active + absorbing == specification["rollout_control_transitions"],
        "optimizer_uses_all_and_only_active_samples": sum(u.get("active_samples", 0) for u in updates) == active,
        "genuine_optimizer_work": bool(updates) and sum(u.get("optimizer_steps", 0) for u in updates) > 0,
        "interval_samples_accounted": sum(i.get("active_samples", 0) for i in intervals) == active
            and sum(i.get("absorbing_samples", 0) for i in intervals) == absorbing,
        "full_control_accounting": all(len(c.get("active_count_by_control", [])) == specification["controls_per_cohort"] for c in cohorts),
        "checkpoint_reload_verified": all(all(c.get("checkpoint_validation", {}).get(k) is True
            for k in ("model_and_optimizer_finite", "identical_loaded_actions", "optimizer_state_nonempty", "diagnostics_finite")) for c in cohorts),
        "diagnostic_horizon": all(c.get("checkpoint_validation", {}).get("diagnostic_shape", [None])[0] == specification["controls_per_cohort"] for c in cohorts),
        "all_exports_present_and_hashed": len(exports) == 2 * specification["cohorts"] and all(
            (worker_dir / item["path"]).is_file() and (worker_dir / item["path"]).stat().st_size == item["bytes"]
            and sha256(worker_dir / item["path"]) == item["sha256"] for item in exports),
    }


def charged_cost_bounds(report, specification):
    measured = report.get("cost", {}).get("charged_control_equivalent_transitions", 0)
    if report.get("status") == "completed":
        return {"lower": measured, "upper": measured, "scope": "Completed worker: exact full-cohort accounting."}
    # The inherited init counter increments before simulate, while rollout accounting
    # increments after a complete control. A native exception can straddle either.
    lower = max(0, measured - specification["num_envs"] / 8) if report.get("status") == "failed" else measured
    return {"lower": lower, "upper": None, "planned_charged_transitions": specification["expected_charged_transitions"],
            "scope": "Incomplete worker: durable completed work is the lower bound; allow one attempted initialization tick in a failed report. The inherited initialization loop has no verified hard step cap, so the upper bound is unknown. The expected 67-step initialization is a plan, not a failure bound; partial weights do not determine native work."}


def resource_summary(samples):
    return {"samples": samples, "sample_count": len(samples),
            "peak_used_mib": max((s["used_mib"] for s in samples), default=None),
            "mean_gpu_utilization_percent": sum(s["utilization_percent"] for s in samples) / len(samples) if samples else None,
            "minimum_free_ram_mib": min((s["free_ram_mib"] for s in samples if s["free_ram_mib"] is not None), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-envs", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--max-minutes", type=int, default=25)
    parser.add_argument("--baseline-run", type=Path, help="Completed 1024 capacity-v1 manifest required for a 2048 launch")
    parser.add_argument("--sim-python", type=Path, default=Path("C:/isaac-sim/python.bat"))
    args = parser.parse_args()
    run_id = validate_run_id(args.run_id)
    registration_path = ROOT / "configs/training_capacity_v1.json"
    registration = json.loads(registration_path.read_text())
    specification = specification_for(registration, args.num_envs)
    if not 1 <= args.max_minutes <= registration["wall_limit_minutes_per_size"]:
        parser.error("Capacity probes require a wall deadline of 1..25 minutes")
    study_path, protocol_path = ROOT / "configs/study.json", ROOT / "configs/protocol_v4.json"
    study = json.loads(study_path.read_text())
    missing = validate_study(study)
    if args.mode == "plan":
        print(json.dumps({"specification": specification,
            "total_pair_charged_transitions": sum(specification_for(registration, n)["expected_charged_transitions"] for n in registration["num_envs"]),
            "unmet_original_study_gates": missing, "physics_robustness_gate_passed": False,
            "scope_and_limitations": registration["scope_and_limitations"]}, indent=2))
        return 0
    if missing or specification["seed"] not in study["training"]["seeds"]:
        parser.error("Basic task gates or training/development seed separation failed")
    frozen = json.loads(protocol_path.read_text())
    if sha256(study_path) != frozen["study_sha256"]:
        parser.error("Frozen study changed; a new capacity protocol is required")
    for name, digest in frozen["behavior_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            parser.error("Frozen task/optimizer dependency changed: " + name)
    upstream = ROOT / ".deps/IsaacLab"
    upstream_status = git(upstream, "status", "--porcelain", "--untracked-files=no")
    if git(upstream, "rev-parse", "HEAD") != LAB_COMMIT or upstream_status:
        parser.error("Installed upstream must match its unchanged pinned revision")
    baseline = None
    if args.num_envs == 2048:
        if args.baseline_run is None:
            parser.error("2048 requires --baseline-run pointing to a completed 1024 capacity-v1 manifest")
        baseline = json.loads(args.baseline_run.read_text())
        resource = baseline.get("resources", {})
        if (baseline.get("status") != "completed" or baseline.get("version") != "training_capacity_v1"
                or baseline.get("specification", {}).get("num_envs") != 1024
                or not baseline.get("checks") or not all(baseline["checks"].values())
                or baseline.get("registration_sha256") != sha256(registration_path)
                or not resource.get("sample_count") or resource.get("peak_used_mib", 10240) >= 8192
                or (resource.get("minimum_free_ram_mib") is not None and resource["minimum_free_ram_mib"] < 8192)):
            parser.error("1024 baseline must pass artifacts and show GPU/RAM headroom before 2048")
        baseline = {"path": str(args.baseline_run.resolve()), "sha256": sha256(args.baseline_run), "resources": resource}
    output = ROOT / "artifacts/assembly" / run_id
    output.mkdir(parents=True, exist_ok=False)
    worker = output / "profile"
    lock_path = output.parent / "training_capacity_v1.lock"
    try:
        serial_lock = lock_path.open("x", encoding="utf8")
    except FileExistsError:
        parser.error("Another capacity probe owns the serial lock; inspect its recorded PID before resolving a stale lock")
    with serial_lock:
        serial_lock.write(json.dumps({"pid": os.getpid(), "run_id": run_id}))
    manifest = {"schema": 1, "version": "training_capacity_v1", "status": "starting", "run_id": run_id,
                "research_result": False, "specification": specification, "started_at_utc": datetime.now(UTC).isoformat(),
                "wall_time_limit_s": args.max_minutes * 60, "baseline_1024": baseline,
                "physics_robustness_gate_passed": False, "scope_and_limitations": registration["scope_and_limitations"]}
    samples, stop = [], threading.Event()
    monitor = None
    start = time.monotonic()
    try:
        source_hashes = snapshot_source(output / "source.zip")
        machine_lock = ROOT / "environment-lock.local.json"
        manifest.update(source_commit_at_start=git(ROOT, "rev-parse", "HEAD"),
            source_dirty_at_start=bool(git(ROOT, "status", "--porcelain")), source_hashes=source_hashes,
            upstream_commit=LAB_COMMIT, upstream_dirty_at_start=bool(upstream_status),
            upstream_source_sha256={p.relative_to(ROOT).as_posix(): sha256(p)
                for folder in ("factory", "forge")
                for p in (upstream / "source/isaaclab_tasks/isaaclab_tasks/direct" / folder).rglob("*.py")},
            registration_sha256=sha256(registration_path), study_sha256=sha256(study_path), protocol_sha256=sha256(protocol_path),
            environment_lock=json.loads(machine_lock.read_text()) if machine_lock.is_file() else None,
            environment_lock_sha256=sha256(machine_lock) if machine_lock.is_file() else None,
            environment_overrides={"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"}, input_checkpoint=None)
        # Separate immutable prelaunch manifest avoids a circular hash and preserves the exact worker inputs.
        prelaunch = output / "prelaunch.json"
        command = [str(args.sim_python), str(ROOT / "scripts/profile_training_capacity_v1.py"), "--headless",
                   "--output", str(worker), "--manifest", str(prelaunch)]
        manifest["command_before_manifest_hash"] = command
        write_json(prelaunch, manifest)
        command += ["--manifest-sha256", sha256(prelaunch)]
        manifest.update(command=command, prelaunch_sha256=sha256(prelaunch))
        write_json(output / "manifest.json", manifest)
        monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
        monitor.start()
        code, timeout, resource_stop = run_resource_bounded(command, output / "process.log", start + args.max_minutes * 60,
                                      {**os.environ, **manifest["environment_overrides"]}, samples)
        report_path = worker / "report.json"
        if not report_path.is_file():
            report_path = worker / "progress.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        checks = report_checks(report, specification, worker)
        checks["resource_samples_present"] = bool(samples)
        checks["resource_guard_not_triggered"] = resource_stop is None
        manifest.update(status=assess_completion(code, timeout, checks), checks=checks, returncode=code,
                        timed_out=timeout, resource_guard_stop=resource_stop, measured_cost=report.get("cost"))
        measured = report.get("cost", {}).get("charged_control_equivalent_transitions", 0)
        manifest["charged_cost_bounds"] = charged_cost_bounds(report, specification)
        completed_cohorts = report.get("cohorts", [])
        work_wall = sum(c.get("complete_cohort_wall_s", 0) for c in completed_cohorts)
        manifest["throughput"] = {"complete_cohort_work_wall_s": work_wall,
            "active_samples_per_complete_cohort_s": report.get("cost", {}).get("active_control_transitions", 0) / work_wall if work_wall and manifest["status"] == "completed" else None,
            "charged_transitions_per_complete_cohort_s": measured / work_wall if work_wall and manifest["status"] == "completed" else None,
            "optimizer_steps": sum(u.get("optimizer_steps", 0) for u in report.get("optimizer_updates", [])),
            "scope": "Cohort denominator includes physical initialization, rollout, PPO, complete export and reload. Launcher total additionally includes simulator startup and shutdown."}
    except BaseException as exc:
        manifest.update(status="launcher_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=8)
        manifest["elapsed_wall_s"] = time.monotonic() - start
        manifest["resources"] = resource_summary(samples)
        write_json(output / "gpu_resources.json", manifest["resources"])
        manifest["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                                 for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "process.log"}]
        try:
            write_json(output / "manifest.json", manifest)
        finally:
            lock_path.unlink()
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "wall_s": manifest["elapsed_wall_s"]}))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
