"""Bounded serial post-correction path pairing and PPO contract checks."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from assembly_recovery.protocol import write_json  # noqa: E402
from scripts.run_experiment import run_bounded  # noqa: E402


def main():
    output = ROOT / "artifacts/assembly/projection_checks_queue_v2"
    output.mkdir(parents=True, exist_ok=False)
    specs = [("training_path_reference_r02", ["--path-backend", "reference", "--controller", "retry"]),
             ("training_path_tensor_r02", ["--path-backend", "tensor", "--controller", "retry"]),
             ("tensor_contract_projection_32_r01", ["--num-envs", "32", "--audit", "--optimize"])]
    report = {"status": "running", "specs": specs, "runs": [], "scope": "Shared initial-action correction verification; no final tests."}
    deadline = time.monotonic() + 1200
    write_json(output / "queue.json", report)
    for name, extra in specs:
        command = [sys.executable, str(ROOT / "scripts/run_training_probe.py"), "--run-id", name, "--max-minutes", "8", *extra]
        code, timeout = run_bounded(command, output / f"{name}.log", deadline, os.environ.copy())
        report["runs"].append({"run_id": name, "returncode": code, "timed_out": timeout})
        write_json(output / "queue.json", report)
        print(json.dumps(report["runs"][-1]), flush=True)
        if timeout or code:
            break
    report["status"] = "completed" if len(report["runs"]) == 3 and all(r["returncode"] == 0 for r in report["runs"]) else "failed_or_incomplete"
    write_json(output / "queue.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
