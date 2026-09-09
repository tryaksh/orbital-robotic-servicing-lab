"""Evaluate the final registered completion-credit uniform policy on the three declared development seeds."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import sha256, validate_run_id, write_json  # noqa: E402
from scripts.run_experiment import run_bounded  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--version", choices=("v4",), default="v4")
    args = parser.parse_args()
    training = ROOT / "artifacts/assembly" / validate_run_id(args.training_run)
    manifest = json.loads((training / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Only evaluate a completed pilot")
    checkpoint = training / "pilot/budget_checkpoint.pt"
    output = ROOT / "artifacts/assembly" / f"uniform_development_queue_{args.version}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "checkpoint_sha256": sha256(checkpoint), "runs": [],
              "scope": "Three serial development evaluations. No final seeds or checkpoint selection by score.", "wall_limit_s": 1800}
    deadline = time.monotonic() + report["wall_limit_s"]
    write_json(output / "queue.json", report)
    for seed in (10070, 10071, 10072):
        run_id = f"uniform_dev_{seed}_{args.version}"
        command = [sys.executable, str(ROOT / "scripts/run_unclipped.py"), "evaluate", "--run-id", run_id,
                   "--seed", str(seed), "--checkpoint", str(checkpoint), "--max-minutes", "10"]
        code, timeout = run_bounded(command, output / f"{run_id}.log", deadline, os.environ.copy())
        report["runs"].append({"run_id": run_id, "returncode": code, "timed_out": timeout})
        write_json(output / "queue.json", report)
        print(json.dumps(report["runs"][-1]), flush=True)
        if timeout or code:
            break
    report["status"] = "completed" if len(report["runs"]) == 3 and all(r["returncode"] == 0 for r in report["runs"]) else "failed_or_incomplete"
    write_json(output / "queue.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
