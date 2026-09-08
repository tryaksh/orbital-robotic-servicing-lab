"""Launch one bounded peg validation with a source snapshot and artifact checks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.protocol import LAB_COMMIT, assess_completion, sha256, validate_run_id, write_json  # noqa: E402
from scripts.run_experiment import git, run_bounded, snapshot_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=10070)
    parser.add_argument("--gravity", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--velocity", choices=("corrected", "upstream"), default="corrected")
    parser.add_argument("--controller", choices=("hold", "insert_withdraw", "release", "retry", "continue"), default="hold")
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--max-minutes", type=int, default=5)
    parser.add_argument("--sim-python", type=Path, default=Path("C:/isaac-sim/python.bat"))
    parser.add_argument("--retry-config", type=Path)
    parser.add_argument("--fault-plan", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-env", type=int, default=1)
    args = parser.parse_args()
    run_id = validate_run_id(args.run_id)
    if not 1 <= args.seconds <= 35 or not 1 <= args.max_minutes <= 30 or args.seed < 0:
        parser.error("Validation requires 1..35 simulated seconds, 1..30 wall minutes and a nonnegative seed")
    lab = ROOT / ".deps/IsaacLab"
    upstream = git(lab, "rev-parse", "HEAD")
    if upstream != LAB_COMMIT or git(lab, "status", "--porcelain", "--untracked-files=no"):
        parser.error("Pinned upstream source must be unchanged")
    if not args.sim_python.is_file():
        parser.error("Simulator interpreter not found")
    if args.fault_plan:
        plan = json.loads(args.fault_plan.read_text())
        from assembly_recovery.faults import validate_development_request

        try:
            validate_development_request(plan, study_sha256=sha256(ROOT / "configs/study.json"),
                                         settings_sha256=sha256(args.retry_config) if args.retry_config else None,
                                         seed=args.seed, controller=args.controller, seconds=args.seconds,
                                         gravity=args.gravity, velocity=args.velocity)
        except ValueError as exc:
            parser.error(str(exc))
    output = ROOT / "artifacts/assembly" / run_id
    output.mkdir(parents=True, exist_ok=False)
    command = [str(args.sim_python), str(ROOT / "scripts/validate_peg.py"), "--headless", "--output", str(output / "probe"),
               "--seed", str(args.seed), "--gravity", args.gravity, "--velocity", args.velocity,
               "--controller", args.controller, "--seconds", str(args.seconds)]
    if args.retry_config:
        config_copy = output / "retry_settings.json"
        config_copy.write_bytes(args.retry_config.read_bytes())
        command += ["--retry-config", str(config_copy)]
    if args.fault_plan:
        plan_copy = output / "fault_plan.json"
        plan_copy.write_bytes(args.fault_plan.read_bytes())
        command += ["--fault-plan", str(plan_copy)]
    if args.video:
        command += ["--video", "--enable_cameras", "--video-env", str(args.video_env)]
    manifest = {"schema": 1, "run_id": run_id, "status": "starting", "research_result": False,
                "started_at_utc": datetime.now(UTC).isoformat(), "command": command,
                "wall_time_limit_s": args.max_minutes * 60,
                "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"),
                "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
                "source_hashes": snapshot_source(output / "source.zip"), "upstream_commit": upstream,
                "checkpoint": None, "seed": args.seed,
                "environment_lock": json.loads((ROOT / "environment-lock.local.json").read_text()) if (ROOT / "environment-lock.local.json").is_file() else None,
                "environment_lock_sha256": sha256(ROOT / "environment-lock.local.json") if (ROOT / "environment-lock.local.json").is_file() else None,
                "fault_plan_sha256": sha256(output / "fault_plan.json") if args.fault_plan else None,
                "retry_config_sha256": sha256(output / "retry_settings.json") if args.retry_config else None,
                "upstream_source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                                           for folder in ("factory", "forge")
                                           for p in (lab / "source/isaaclab_tasks/isaaclab_tasks/direct" / folder).rglob("*.py")},
                "environment_overrides": {"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
                "scope": "Scripted development feasibility validation; no learned policy or research effect size."}
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        environment = {**os.environ, **manifest["environment_overrides"]}
        code, timeout = run_bounded(command, output / "process.log", started + args.max_minutes * 60, environment)
        report = output / "probe/report.json"
        trace = output / "probe/physics_samples.jsonl"
        checks = {"probe_completed": report.is_file() and json.loads(report.read_text()).get("status") == "completed",
                  "physics_trace_saved": trace.is_file() and trace.stat().st_size > 0}
        if args.video:
            checks["video_saved"] = (output / "probe/trajectory.mp4").is_file() and (output / "probe/trajectory.mp4").stat().st_size > 0
        manifest.update(returncode=code, timed_out=timeout, checks=checks,
                        status=assess_completion(code, timeout, checks))
        manifest["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                                 for p in (output / "source.zip", report, trace, output / "probe/control_samples.jsonl",
                                           output / "probe/environment.yaml") if p.is_file()]
        manifest["artifacts"] += [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                                  for p in (output / "probe/trajectory.mp4", output / "retry_settings.json", output / "fault_plan.json") if p.is_file()]
    except BaseException as exc:
        manifest.update(status="launcher_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_wall_s"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "wall_s": manifest["elapsed_wall_s"]}))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
