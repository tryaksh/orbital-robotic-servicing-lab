"""One bounded serial queue; compatibility verification gates finer physics."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import LAB_COMMIT, assess_completion, sha256, validate_run_id, write_json  # noqa: E402
from scripts.run_experiment import git, snapshot_source  # noqa: E402
from scripts.run_training_probe import monitor_gpu, run_resource_bounded  # noqa: E402


def preflight_v1(registration):
    if git(ROOT, "branch", "--show-current") != "research/assembly-recovery-training":
        raise ValueError("Use the authorized research branch")
    if git(ROOT, "status", "--porcelain"):
        raise ValueError("Commit the exact preregistration/source before launch")
    upstream = ROOT / ".deps/IsaacLab"
    if git(upstream, "rev-parse", "HEAD") != LAB_COMMIT or git(upstream, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Pinned upstream source changed")
    protocol_path = ROOT / registration["physical_protocol"]["path"]
    if sha256(protocol_path) != registration["physical_protocol"]["sha256"]:
        raise ValueError("Frozen physical protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if sha256(ROOT / "configs/study.json") != protocol["study_sha256"]:
        raise ValueError("Frozen study changed")
    for name, digest in {**protocol["behavior_source_sha256"], **registration["source_sha256"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    for name, digest in registration["installed_source_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("Installed simulator source changed: " + name)
    for name, digest in registration["preserved_evidence_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Preserved evidence changed: " + name)
    checkpoint = ROOT / registration["checkpoint_path"]
    training = json.loads((checkpoint.parent.parent / "manifest.json").read_text())
    if training["status"] != "completed" or sha256(checkpoint) != registration["checkpoint_sha256"]:
        raise ValueError("Expected the complete registered budget model")
    if registration["seed"] != 10071 or registration["training_launches_allowed"] != 0:
        raise ValueError("This queue permits only the registered development physics evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    registration_path = ROOT / "configs/learned_physics_validation_v3.json"
    registration = json.loads(registration_path.read_text())
    if args.plan:
        print(json.dumps({k: registration[k] for k in ("run_order", "expected_charged_reference_transitions", "wall_limit_s")}, indent=2))
        return 0
    preflight_v1(registration)
    output = ROOT / "artifacts/assembly" / validate_run_id(args.run_id)
    output.mkdir(parents=True, exist_ok=False)
    queue = {"schema": 1, "status": "running", "runs": [], "registration_sha256": sha256(registration_path),
             "launcher_pid": os.getpid(), "started_at_utc": datetime.now(UTC).isoformat(),
             "wall_limit_s": registration["wall_limit_s"], "run_order": registration["run_order"]}
    write_json(output / "queue.json", queue)
    deadline = time.monotonic() + registration["wall_limit_s"]
    for row in registration["run_order"]:
        preflight_v1(registration)
        if time.monotonic() >= deadline:
            break
        label = row["label"]
        run = output / label
        run.mkdir(exist_ok=False)
        checkpoint = ROOT / registration["checkpoint_path"]
        command = ["C:/isaac-sim/python.bat", str(ROOT / "scripts/validate_learned_physics_v3.py"), "--headless",
                   "--output", str(run / "validation"), "--mode", row["mode"], "--refinement", str(row["refinement"]),
                   "--seed", str(registration["seed"]), "--checkpoint", str(checkpoint), "--checkpoint-sha256", sha256(checkpoint)]
        lock = ROOT / "environment-lock.local.json"
        upstream = ROOT / ".deps/IsaacLab"
        manifest = {"schema": 1, "status": "starting", "run_id": args.run_id + "/" + label,
            "research_result": False, "seed": registration["seed"], "mode": row["mode"], "refinement": row["refinement"],
            "started_at_utc": datetime.now(UTC).isoformat(), "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"),
            "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")), "source_hashes": snapshot_source(run / "source.zip"),
            "upstream_commit": LAB_COMMIT, "installed_source_sha256": registration["installed_source_sha256"], "upstream_source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                for folder in ("factory", "forge") for p in (upstream / "source/isaaclab_tasks/isaaclab_tasks/direct" / folder).rglob("*.py")},
            "protocol_sha256": registration["physical_protocol"]["sha256"], "study_sha256": sha256(ROOT / "configs/study.json"),
            "registration_sha256": sha256(registration_path), "environment_lock": json.loads(lock.read_text()),
            "environment_lock_sha256": sha256(lock), "command": command,
            "input_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "environment_overrides": {"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
            "expected_charged_reference_transitions": row["expected_charged_reference_transitions"],
            "wall_limit_s": row["wall_limit_s"], "scope": registration["scope_and_limitations"]}
        write_json(run / "manifest.json", manifest)
        samples, stop = [], threading.Event()
        monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
        monitor.start()
        start = time.monotonic()
        try:
            code, timeout, resource_stop = run_resource_bounded(command, run / "process.log", min(deadline, start + row["wall_limit_s"]),
                {**os.environ, **manifest["environment_overrides"]}, samples)
            report_path = run / "validation/report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            checks = {"completed_report": report.get("status") == "completed",
                      "all_requests": len(report.get("jobs", [])) == 28,
                      "trajectory_saved": (run / "validation/trajectory.pt").is_file(),
                      "charged_budget": report.get("cost", {}).get("charged_reference_transitions") == row["expected_charged_reference_transitions"]}
            manifest.update(status=assess_completion(code, timeout, checks), returncode=code, timed_out=timeout,
                            resource_guard_stop=resource_stop, checks=checks)
            progress_path = run / "validation/progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            manifest["cost_observed"] = report.get("cost", progress.get("cost"))
            manifest["cost_complete"] = bool(report.get("cost")) and not timeout
            manifest["unresolved_cost_upper_bound"] = None if manifest["cost_complete"] else (256 + 3600 * row["refinement"]) * 28 / 8
        except BaseException as exc:
            manifest.update(status="launcher_failed", error=f"{type(exc).__name__}: {exc}",
                            unresolved_cost_upper_bound=(256 + 3600 * row["refinement"]) * 28 / 8)
            raise
        finally:
            stop.set()
            monitor.join(timeout=8)
            write_json(run / "gpu_resources.json", {"samples": samples, "peak_used_mib": max((s["used_mib"] for s in samples), default=None)})
            manifest["elapsed_wall_s"] = time.monotonic() - start
            manifest["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(run.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "process.log"}]
            write_json(run / "manifest.json", manifest)
        entry = {"label": label, "simulator_status": manifest["status"], "manifest_sha256": sha256(run / "manifest.json")}
        if manifest["status"] == "completed":
            verification_command = [sys.executable, str(ROOT / "scripts/verify_learned_physics_v3.py"),
                "--single", "--run", str(run), "--output", str(run / "verification.json")]
            with (run / "verification.log").open("w", encoding="utf8") as log:
                try:
                    verified = subprocess.run(verification_command, stdout=log, stderr=subprocess.STDOUT,
                        timeout=max(1, min(120, deadline - time.monotonic())), check=False)
                    entry["verification_returncode"] = verified.returncode
                except subprocess.TimeoutExpired:
                    entry["verification_returncode"] = "timeout"
            entry["status"] = "verified" if entry["verification_returncode"] == 0 else "verification_failed"
            if (run / "verification.json").exists():
                entry["verification_sha256"] = sha256(run / "verification.json")
        else:
            entry["status"] = manifest["status"]
        queue["runs"].append(entry)
        write_json(output / "queue.json", queue)
        print(json.dumps(entry), flush=True)
        # This includes exact archived 120 Hz parity before the fine runs.
        if entry["status"] != "verified":
            break
    queue["status"] = "completed" if len(queue["runs"]) == 6 and all(r["status"] == "verified" for r in queue["runs"]) else "failed_or_incomplete"
    queue["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_json(output / "queue.json", queue)
    return 0 if queue["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
