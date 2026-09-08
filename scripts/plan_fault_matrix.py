"""Predeclare one small development-only matrix before launching any of its jobs."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from assembly_recovery.faults import development_cases  # noqa: E402
from assembly_recovery.protocol import sha256, validate_run_id, validate_study  # noqa: E402
from scripts.run_experiment import git  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--retry-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_id = validate_run_id(args.run_id)
    study = json.loads((ROOT / "configs/study.json").read_text())
    validate_study(study)
    cases = [case for seed in study["splits"]["development_seeds"] for case in development_cases(study, seed)]
    result = {"schema": 1, "run_id": run_id, "created_at_utc": datetime.now(UTC).isoformat(),
              "source_commit": git(ROOT, "rev-parse", "HEAD"), "source_dirty": bool(git(ROOT, "status", "--porcelain")),
              "study": study, "study_sha256": sha256(ROOT / "configs/study.json"), "cases": cases,
              "retry_settings": json.loads(args.retry_config.read_text()), "retry_settings_sha256": sha256(args.retry_config),
              "arms": ["scripted_actor_retry", "scripted_continued_insertion"],
              "requested_jobs": len(cases) * 2, "control_transitions": len(cases) * 2 * 450,
              "deadline_s": 30, "max_wall_minutes_per_run": 15, "concurrent_gpu_jobs": 1,
              "decision": "Evaluate all declared cases with realignment variant 2, without further controller tuning or final-test access.",
              "scope": "Representative development points and shared randomization; no claim that whole severity intervals are recoverable."}
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_id": run_id, "requested_jobs": result["requested_jobs"],
                      "control_transitions": result["control_transitions"], "study_sha256": result["study_sha256"]}))


if __name__ == "__main__":
    main()
