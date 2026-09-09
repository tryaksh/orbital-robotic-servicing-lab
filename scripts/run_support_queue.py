"""One bounded serial queue for the remaining predeclared support probes."""
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
    output = ROOT / "artifacts/assembly/training_support_queue_v1"
    output.mkdir(parents=True, exist_ok=False)
    specs = [("low", 10070, "continue"), ("interior", 10071, "retry"), ("interior", 10071, "continue"),
             ("high", 10072, "retry"), ("high", 10072, "continue")]
    report = {"status": "running", "scope": "Serial development support checks, no learning or final tests.",
              "wall_limit_s": 1800, "specs": specs, "runs": []}
    deadline = time.monotonic() + report["wall_limit_s"]
    write_json(output / "queue.json", report)
    for grid, seed, controller in specs:
        run_id = f"training_support_{grid}_{controller}_r01"
        command = [sys.executable, str(ROOT / "scripts/run_training_probe.py"), "--run-id", run_id,
                   "--path-backend", "tensor", "--controller", controller, "--grid", grid, "--seed", str(seed),
                   "--max-minutes", "10"]
        code, timeout = run_bounded(command, output / f"{run_id}.log", deadline, os.environ.copy())
        report["runs"].append({"run_id": run_id, "returncode": code, "timed_out": timeout})
        write_json(output / "queue.json", report)
        print(json.dumps(report["runs"][-1]), flush=True)
        if timeout:
            break
    report["status"] = "completed" if len(report["runs"]) == len(specs) and all(r["returncode"] == 0 for r in report["runs"]) else "failed_or_incomplete"
    write_json(output / "queue.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
