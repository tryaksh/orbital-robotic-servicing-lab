"""Predeclare and launch a gated uniform pilot or development checkpoint evaluation."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "train", "evaluate"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=170)
    parser.add_argument("--num-envs", type=int, choices=(32, 64, 128, 256, 512, 1024), default=1024)
    parser.add_argument("--cohorts", type=int, default=5)
    parser.add_argument("--max-minutes", type=int, default=45)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sim-python", type=Path, default=Path("C:/isaac-sim/python.bat"))
    args = parser.parse_args()
    study_path = ROOT / "configs/study.json"
    study = json.loads(study_path.read_text())
    missing = validate_study(study)
    if not 1 <= args.cohorts <= 5 or not 1 <= args.max_minutes <= 120:
        parser.error("The pilot permits one to five cohorts with a 1..120 minute wall limit")
    if args.mode == "evaluate":
        if args.seed not in study["splits"]["development_seeds"] or not args.checkpoint:
            parser.error("Pilot evaluation requires a declared development seed and checkpoint")
        args.num_envs, args.cohorts = 28, 1
    elif args.seed not in study["training"]["seeds"]:
        parser.error("Pilot training cannot use development or test seeds")
    if args.mode != "evaluate" and args.checkpoint:
        parser.error("Training starts fresh; a checkpoint is accepted only for evaluation")
    run_id = validate_run_id(args.run_id)
    specification = {"mode": args.mode, "seed": args.seed, "num_envs": args.num_envs, "cohorts": args.cohorts,
                     "rollout_control_transitions": args.num_envs * 450 * args.cohorts,
                     "expected_initialization_physics_env_steps": args.num_envs * 67 * args.cohorts,
                     "expected_charged_control_equivalent_transitions": args.num_envs * (450 + 67 / 8) * args.cohorts,
                     "budget_rule": "Fixed full-cohort rollout budget; all measured initialization physics charged in addition. Report deviations, never hide them.",
                     "checkpoint_selection": "Last predeclared cohort; intermediate weights remain partial.",
                     "reserved_evaluation_and_reporting_fraction": 0.25}
    if args.mode == "plan":
        print(json.dumps({"specification": specification, "unmet_gates": missing}, indent=2))
        return 0
    if missing:
        parser.error("Scientific gates still open: " + ", ".join(missing))
    protocol = ROOT / study["training"]["pilot_protocol"]
    if not protocol.is_file():
        parser.error("Versioned protocol is missing")
    frozen = json.loads(protocol.read_text())
    if sha256(study_path) != frozen["study_sha256"]:
        parser.error("Frozen study configuration changed")
    for name, digest in frozen["behavior_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            parser.error("Frozen behavior source changed: " + name)
    upstream = ROOT / ".deps/IsaacLab"
    if git(upstream, "rev-parse", "HEAD") != LAB_COMMIT or git(upstream, "status", "--porcelain", "--untracked-files=no"):
        parser.error("Pinned upstream source changed")
    output = ROOT / "artifacts/assembly" / run_id
    output.mkdir(parents=True, exist_ok=False)
    command = [str(args.sim_python), str(ROOT / "scripts/train_uniform.py"), "--headless", "--output", str(output / "pilot"),
               "--mode", args.mode, "--seed", str(args.seed), "--num-envs", str(args.num_envs), "--cohorts", str(args.cohorts)]
    command += ["--protocol", str(protocol), "--expected-study-sha256", sha256(study_path), "--expected-protocol-sha256", sha256(protocol),
                "--expected-charged-transitions", str(specification["expected_charged_control_equivalent_transitions"])]
    checkpoint = None
    if args.checkpoint:
        checkpoint = {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)}
        command += ["--checkpoint", checkpoint["path"], "--checkpoint-sha256", checkpoint["sha256"]]
    lock = ROOT / "environment-lock.local.json"
    manifest = {"schema": 1, "status": "starting", "run_id": run_id, "research_result": False,
                "started_at_utc": datetime.now(UTC).isoformat(), "specification": specification, "command": command,
                "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"),
                "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
                "source_hashes": snapshot_source(output / "source.zip"), "upstream_commit": LAB_COMMIT,
                "protocol_sha256": sha256(protocol), "study_sha256": sha256(study_path),
                "environment_lock": json.loads(lock.read_text()) if lock.is_file() else None,
                "environment_lock_sha256": sha256(lock) if lock.is_file() else None,
                "input_checkpoint": checkpoint, "wall_time_limit_s": args.max_minutes * 60,
                "environment_overrides": {"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
                "scope": "Bounded uniform pilot or development evaluation; no adaptive comparison or final-test result."}
    write_json(output / "manifest.json", manifest)
    samples, stop = [], threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
    monitor.start()
    start = time.monotonic()
    try:
        code, timeout, resource_stop = run_resource_bounded(command, output / "process.log", start + args.max_minutes * 60,
                                  {**os.environ, **manifest["environment_overrides"]}, samples)
        path = output / "pilot/report.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        checks = {"report_complete": report.get("status") == "completed",
                  "all_requested_cohorts": len(report.get("cohorts", [])) == args.cohorts,
                  "rollout_budget_reached": report.get("cost", {}).get("rollout_control_transitions") == specification["rollout_control_transitions"]}
        if args.mode == "train":
            checks["charged_budget_reached"] = report.get("cost", {}).get("charged_control_equivalent_transitions") == specification["expected_charged_control_equivalent_transitions"]
            checks["checkpoint_loaded"] = report.get("checkpoint_validation", {}).get("identical_loaded_actions") is True
            checks["checkpoint_finite"] = report.get("checkpoint_validation", {}).get("finite") is True
        else:
            checks["checkpoint_hash_matches"] = report.get("checkpoint_sha256") == checkpoint["sha256"]
        manifest.update(status=assess_completion(code, timeout, checks), checks=checks, returncode=code,
                        timed_out=timeout, resource_guard_stop=resource_stop)
    except BaseException as exc:
        manifest.update(status="launcher_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        monitor.join(timeout=8)
        write_json(output / "gpu_resources.json", {"samples": samples,
                   "peak_used_mib": max((x["used_mib"] for x in samples), default=None),
                   "minimum_free_ram_mib": min((x["free_ram_mib"] for x in samples if x["free_ram_mib"] is not None), default=None)})
        manifest["elapsed_wall_s"] = time.monotonic() - start
        manifest["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                                 for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "process.log"}]
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "wall_s": manifest["elapsed_wall_s"]}))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
