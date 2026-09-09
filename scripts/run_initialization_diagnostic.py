"""Launch a short, provenance-recorded initialization diagnosis without learning."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import LAB_COMMIT, assess_completion, sha256, validate_run_id, write_json  # noqa: E402
from scripts.run_experiment import git, run_bounded, snapshot_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if git(ROOT / ".deps/IsaacLab", "rev-parse", "HEAD") != LAB_COMMIT or git(ROOT / ".deps/IsaacLab", "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Pinned upstream changed")
    output = ROOT / "artifacts/assembly" / validate_run_id(args.run_id)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / "artifacts/assembly/uniform_pilot_170_v1/pilot/partial_cohort_2.pt"
    command = ["C:/isaac-sim/python.bat", str(ROOT / "scripts/probe_initialization_failure.py"), "--headless",
               "--output", str(output / "probe"), "--checkpoint", str(checkpoint), "--checkpoint-sha256", sha256(checkpoint)]
    m = {"run_id": output.name, "status": "starting", "research_result": False, "command": command,
         "source_commit_at_start": git(ROOT, "rev-parse", "HEAD"), "source_dirty_at_start": bool(git(ROOT, "status", "--porcelain")),
         "source_hashes": snapshot_source(output / "source.zip"), "upstream_commit": LAB_COMMIT,
         "checkpoint_sha256": sha256(checkpoint), "seed": 170, "rng_from_partial_checkpoint": True,
         "environment_lock": json.loads((ROOT / "environment-lock.local.json").read_text()),
         "environment_overrides": {"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1"},
         "requested_initializations": 1024, "predeclared_rollout_control_transitions": 0,
         "scope": "One initialization, no policy rollout or optimization; diagnose a failed pilot, not a successful resume."}
    write_json(output / "manifest.json", m)
    start = time.monotonic()
    try:
        code, timeout = run_bounded(command, output / "process.log", start + 300, {**os.environ, **m["environment_overrides"]})
        p = output / "probe/report.json"
        r = json.loads(p.read_text()) if p.exists() else {}
        checks = {"report_completed": r.get("status") == "completed", "initialization_recorded": len(r.get("initialization", [])) == 1024,
                  "state_saved": (output / "probe/initialization.pt").is_file(), "zero_policy_rollout": r.get("cost", {}).get("rollout_control_transitions") == 0}
        m.update(returncode=code, timed_out=timeout, checks=checks, status=assess_completion(code, timeout, checks))
    finally:
        m["elapsed_wall_s"] = time.monotonic() - start
        m["artifacts"] = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "process.log"}]
        write_json(output / "manifest.json", m)
    print(json.dumps({"run_id": m["run_id"], "status": m["status"]}))
    return 0 if m["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
