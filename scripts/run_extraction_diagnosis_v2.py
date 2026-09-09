"""Run the three registered diagnostic arms serially with exact provenance."""
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
from assembly_recovery.protocol import LAB_COMMIT, assess_completion, sha256, validate_run_id, write_json  # noqa: E402
from scripts.run_experiment import git, snapshot_source  # noqa: E402
from scripts.run_training_probe import monitor_gpu, run_resource_bounded  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output = ROOT / "artifacts/assembly" / validate_run_id(args.run_id)
    protocol = ROOT / "configs/protocol_v3.json"
    frozen = json.loads(protocol.read_text())
    if sha256(ROOT / "configs/study.json") != frozen["study_sha256"]:
        parser.error("Frozen study changed")
    for name, digest in frozen["behavior_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            parser.error("Frozen behavior source changed: " + name)
    upstream = ROOT / ".deps/IsaacLab"
    if git(upstream, "rev-parse", "HEAD") != LAB_COMMIT or git(upstream, "status", "--porcelain", "--untracked-files=no"):
        parser.error("Pinned upstream changed")
    checkpoint = ROOT / "artifacts/assembly/uniform_completion_170_v3_r02/pilot/budget_checkpoint.pt"
    registration = ROOT / "configs/extraction_diagnosis_v2.json"
    registered = json.loads(registration.read_text())
    for name, digest in registered["diagnostic_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            parser.error("Frozen diagnostic source changed: " + name)
    evaluation = json.loads((ROOT / "evidence/uniform_completion_v3_r02.json").read_text())
    if evaluation["status"] != "verified" or evaluation["competence_gate_passed"]:
        parser.error("This follow-up is conditional on a verified failed final-budget competence gate")
    if sha256(checkpoint) != evaluation["training"]["checkpoint_sha256"]:
        parser.error("Final evaluated checkpoint mismatch")
    output.mkdir(parents=True, exist_ok=False)
    queue = {"status": "running", "runs": [], "registration_sha256": sha256(registration),
             "expected_charged_transitions": 51338.0, "wall_limit_s": 2400}
    write_json(output / "queue.json", queue)
    deadline = time.monotonic() + queue["wall_limit_s"]
    for controller in ("deterministic", "stochastic", "clipped_mean", "retry"):
        run = output / controller
        run.mkdir(exist_ok=False)
        command = ["C:/isaac-sim/python.bat", str(ROOT / "scripts/diagnose_extraction_v2.py"), "--headless",
                   "--output", str(run / "diagnosis"), "--controller", controller,
                   "--checkpoint", str(checkpoint), "--checkpoint-sha256", sha256(checkpoint)]
        lock = ROOT / "environment-lock.local.json"
        manifest = {"schema": 1, "status": "starting", "run_id": args.run_id + "/" + controller,
                    "research_result": False, "started_at_utc": datetime.now(UTC).isoformat(),
                    "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"),
                    "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
                    "source_hashes": snapshot_source(run / "source.zip"), "upstream_commit": LAB_COMMIT,
                    "upstream_source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                        for folder in ("factory", "forge")
                        for p in (upstream / "source/isaaclab_tasks/isaaclab_tasks/direct" / folder).rglob("*.py")},
                    "protocol_sha256": sha256(protocol), "study_sha256": sha256(ROOT / "configs/study.json"),
                    "registration_sha256": sha256(registration), "reward_registration_sha256": sha256(ROOT / "configs/completion_credit_v3.json"), "environment_lock": json.loads(lock.read_text()),
                    "environment_lock_sha256": sha256(lock), "command": command,
                    "input_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
                    "environment_overrides": {"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
                    "expected_charged_transitions": 12834.5, "wall_limit_s": 600,
                    "scope": "Conditional final-v3-budget extraction diagnosis; four distinct controllers, unchanged physical task and v3 reward."}
        write_json(run / "manifest.json", manifest)
        samples, stop = [], threading.Event()
        monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
        monitor.start()
        start = time.monotonic()
        try:
            code, timeout, resource_stop = run_resource_bounded(command, run / "process.log", min(deadline, start + 600),
                {**os.environ, **manifest["environment_overrides"]}, samples)
            report_path = run / "diagnosis/report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            checks = {"completed_report": report.get("status") == "completed", "all_requests": len(report.get("jobs", [])) == 28,
                      "charged_budget": report.get("cost", {}).get("charged_control_equivalent_transitions") == 12834.5,
                      "checkpoint_hash": report.get("checkpoint_sha256") == manifest["input_checkpoint"]["sha256"],
                      "trajectory_saved": (run / "diagnosis/trajectory.pt").is_file()}
            manifest.update(status=assess_completion(code, timeout, checks), returncode=code, timed_out=timeout,
                            resource_guard_stop=resource_stop, checks=checks)
        except BaseException as exc:
            manifest.update(status="launcher_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            stop.set()
            monitor.join(timeout=8)
            write_json(run / "gpu_resources.json", {"samples": samples, "peak_used_mib": max((s["used_mib"] for s in samples), default=None)})
            manifest["elapsed_wall_s"] = time.monotonic() - start
            manifest["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(run.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "process.log"}]
            write_json(run / "manifest.json", manifest)
        queue["runs"].append({"controller": controller, "status": manifest["status"], "manifest_sha256": sha256(run / "manifest.json")})
        write_json(output / "queue.json", queue)
        print(json.dumps(queue["runs"][-1]), flush=True)
        if manifest["status"] != "completed":
            break
    queue["status"] = "completed" if len(queue["runs"]) == 4 and all(r["status"] == "completed" for r in queue["runs"]) else "failed_or_incomplete"
    write_json(output / "queue.json", queue)
    return 0 if queue["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
