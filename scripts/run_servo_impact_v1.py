"""Conditional2x2 servo/native integration diagnostic; shared raw-force job rule."""
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
from scripts.run_contact_impact_v1 import preflight as frozen_preflight  # noqa: E402
from scripts.run_experiment import git, snapshot_source  # noqa: E402
from scripts.run_training_probe import monitor_gpu, run_resource_bounded  # noqa: E402


def preflight():
    frozen = frozen_preflight()
    continuation = json.loads((ROOT / "configs/contact_impact_continuation_v2.json").read_text())
    for key in ("preserved_source_sha256", "corrected_verifier_source_sha256"):
        for name, digest in continuation[key].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("Continuation source changed: " + name)
    prior = continuation["verified_prior_run"]
    for name, digest in prior["artifact_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Preserved120Hz evidence changed: " + name)
    correction = prior["corrected_verification"]
    if sha256(ROOT / correction["path"]) != correction["sha256"]:
        raise ValueError("Corrected120Hz verification changed")
    from scripts.review_contact_impact_v2 import verify_single
    checked = verify_single(ROOT / prior["path"])
    if not all(checked["checks"].values()):
        raise ValueError("The retained120Hz reference does not pass corrected verification")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--refinements", type=int, nargs="+", default=[4, 8], choices=(4, 8))
    parser.add_argument("--original-evidence", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    registration_path = ROOT / "configs/contact_impact_registration_v1.json"
    registration = json.loads(registration_path.read_text())
    continuation_path = ROOT / "configs/contact_impact_continuation_v2.json"
    continuation = json.loads(continuation_path.read_text())
    servo_path = ROOT / "configs/servo_impact_registration_v1.json"
    servo = json.loads(servo_path.read_text())
    if sha256(ROOT / servo["trigger_evidence"]) != servo["trigger_evidence_sha256"]:
        raise ValueError("Precontact cause-test evidence changed")
    if len(set(args.refinements)) != len(args.refinements):
        raise ValueError("Duplicate resolutions")
    if not args.plan:
        if not args.original_evidence:
            raise ValueError("Original480/960Hz evidence is required")
        original = json.loads(args.original_evidence.read_text())
        decision = original.get("decision", {})
        if (original.get("status") != "verified_diagnostic"
                or decision.get("finest_pair_hz") != [480, 960]
                or decision.get("finest_prefix_pair_pass") is not False):
            raise ValueError("Conditional servo test requires verified original finest-pair sensitivity")
    if args.plan:
        print(json.dumps({"refinements": args.refinements, "registration": registration}, indent=2))
        return 0
    frozen = preflight()
    output = ROOT / "artifacts/assembly" / validate_run_id(args.run_id)
    output.mkdir(parents=True, exist_ok=False)
    queue = {"schema": 1, "status": "running", "runs": [], "launcher_pid": os.getpid(),
             "started_at_utc": datetime.now(UTC).isoformat(), "registration_sha256": sha256(registration_path),
             "continuation_registration_sha256": sha256(continuation_path),
             "servo_registration_sha256": sha256(servo_path)}
    write_json(output / "queue.json", queue)
    deadline = time.monotonic() + registration["queue_wall_limit_s"]
    for refinement in args.refinements:
        preflight()
        run = output / str(120 * refinement)
        run.mkdir(exist_ok=False)
        command = ["C:/isaac-sim/python.bat", str(ROOT / "scripts/probe_servo_impact_v1.py"),
                   "--headless", "--output", str(run / "validation"), "--refinement", str(refinement),
                   "--seed", "10071", "--checkpoint", str(ROOT / frozen["checkpoint_path"]),
                   "--checkpoint-sha256", frozen["checkpoint_sha256"]]
        lock = ROOT / "environment-lock.local.json"
        source_hashes = snapshot_source(run / "source.zip")
        manifest = {"schema": 1, "status": "starting", "research_result": False,
                    "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"),
                    "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
                    "source_hashes": source_hashes, "source_archive_sha256": sha256(run / "source.zip"),
                    "upstream_commit": LAB_COMMIT, "installed_source_sha256": frozen["installed_source_sha256"],
                    "environment_lock": json.loads(lock.read_text()), "environment_lock_sha256": sha256(lock),
                    "registration": registration, "registration_sha256": sha256(registration_path),
                    "continuation_registration": continuation, "continuation_registration_sha256": sha256(continuation_path),
                    "servo_registration": servo, "servo_registration_sha256": sha256(servo_path),
                    "original_refinement_evidence": {"path": str(args.original_evidence.resolve()), "sha256": sha256(args.original_evidence)},
                    "checkpoint_sha256": frozen["checkpoint_sha256"], "seed": 10071,
                    "refinement": refinement, "command": command, "started_at_utc": datetime.now(UTC).isoformat(),
                    "environment_overrides": {"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
                    "scope": registration["scope_and_limitations"]}
        write_json(run / "manifest.json", manifest)
        samples, stop = [], threading.Event()
        monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
        monitor.start()
        start = time.monotonic()
        expected = (67 + 480 * refinement) * 28 / 8
        upper = (256 + 480 * refinement) * 28 / 8
        try:
            code, timeout, resource_stop = run_resource_bounded(command, run / "process.log",
                min(deadline, start + registration["per_run_wall_limit_s"]),
                {**os.environ, **manifest["environment_overrides"]}, samples)
            report_path = run / "validation/report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            progress_path = run / "validation/progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            checks = {"completed_report": report.get("status") == "completed",
                      "all_requests": len(report.get("jobs", [])) == 28,
                      "trajectory_saved": (run / "validation/trajectory.pt").is_file(),
                      "charged_budget": report.get("cost", {}).get("charged_reference_transitions") == expected}
            manifest.update(status=assess_completion(code, timeout, checks), returncode=code, timed_out=timeout,
                            resource_guard_stop=resource_stop, checks=checks,
                            cost_observed=report.get("cost", progress.get("cost")),
                            cost_complete=bool(report.get("cost")) and not timeout and not resource_stop,
                            unresolved_cost_upper_bound=upper)
            if manifest["cost_complete"]:
                manifest["unresolved_cost_upper_bound"] = None
            write_json(run / "manifest.json", manifest)
            if manifest["status"] == "completed":
                from scripts.review_servo_impact_v1 import verify_single
                verification = verify_single(run)
                write_json(run / "verification.json", verification)
                manifest["checks"].update(verification["checks"])
                if not all(verification["checks"].values()):
                    manifest["status"] = "verification_failed"
        except BaseException as exc:
            manifest.update(status="launcher_failed", error=f"{type(exc).__name__}: {exc}",
                            unresolved_cost_upper_bound=upper)
        finally:
            stop.set()
            monitor.join(timeout=8)
            write_json(run / "gpu_resources.json", {"samples": samples,
                "peak_used_mib": max((s["used_mib"] for s in samples), default=None)})
            manifest["elapsed_wall_s"] = time.monotonic() - start
            manifest["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(run.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "process.log"}]
            write_json(run / "manifest.json", manifest)
        queue["runs"].append({"refinement": refinement, "status": manifest["status"],
                              "manifest_sha256": sha256(run / "manifest.json"),
                              "cost": manifest.get("cost_observed"),
                              "unresolved_cost_upper_bound": manifest.get("unresolved_cost_upper_bound")})
        write_json(output / "queue.json", queue)
        print(json.dumps(queue["runs"][-1]), flush=True)
        if manifest["status"] != "completed":
            break
    queue["status"] = "completed" if len(queue["runs"]) == len(args.refinements) and all(
        r["status"] == "completed" for r in queue["runs"]) else "failed_or_incomplete"
    queue["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_json(output / "queue.json", queue)
    return 0 if queue["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
