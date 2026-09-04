"""Preserve a targeted insertion-training intervention and its losing control."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training-task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _clean_source(report: dict[str, Any], label: str) -> dict[str, Any]:
    source = report.get("source_revision") or {}
    if not source.get("commit") or source.get("dirty") is not False:
        raise ValueError(f"{label} report has no clean source revision")
    return source


def _insert_hash(report: dict[str, Any], label: str) -> str:
    checkpoint = report.get("checkpoint_sha256")
    value = checkpoint.get("insert") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} report has no insertion checkpoint SHA-256")
    return value.upper()


def _workflow_arm(report: dict[str, Any], label: str) -> dict[str, Any]:
    condition = report.get("evaluation_condition") or {}
    if condition.get("protocol") != "insertion_condition_v2" or condition.get("station") != 0:
        raise ValueError(f"{label} is not a station-zero conditioned insertion report")
    chain = report.get("chain") or {}
    metrics = chain.get("terminal_metrics") or {}
    return {
        "source_revision": _clean_source(report, label),
        "seed": int(report["seed"]),
        "checkpoint_sha256": _insert_hash(report, label),
        "initial_state_sha256": condition.get("initial_state_sha256"),
        "load_path": condition.get("load_path"),
        "episodes": int(chain["episodes"]),
        "successes": int(chain["successes"]),
        "success_rate": float(chain["success_rate"]),
        "terminal_metrics": metrics,
    }


def _training_arm(report: dict[str, Any]) -> dict[str, Any]:
    terminal = report.get("terminal_metrics") or {}
    summary = terminal.get("terminal_metrics") or {}
    return {
        "source_revision": _clean_source(report, "training-task"),
        "seed": int(report["seed"]) if report.get("seed") is not None else None,
        "seed_reported_in_json": report.get("seed") is not None,
        "checkpoint_sha256": _insert_hash(report, "training-task"),
        "task": report.get("task"),
        "episodes": int(report["episodes_completed"]),
        "successes": int(report["termination_counts"]["insertion_success"]),
        "success_rate": float(report["success_rate"]),
        "terminal_metrics": summary,
    }


def build_report(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    training_report: dict[str, Any],
) -> dict[str, Any]:
    """Build the intervention record after fail-closed pairing checks."""

    baseline = _workflow_arm(baseline_report, "baseline")
    candidate = _workflow_arm(candidate_report, "candidate")
    training = _training_arm(training_report)
    if baseline["seed"] != candidate["seed"]:
        raise ValueError("conditioned controller arms use different seeds")
    if baseline["initial_state_sha256"] != candidate["initial_state_sha256"]:
        raise ValueError("conditioned controller arms use different initial states")
    if baseline["load_path"] != candidate["load_path"]:
        raise ValueError("conditioned controller arms use different load paths")
    if candidate["checkpoint_sha256"] != training["checkpoint_sha256"]:
        raise ValueError("candidate and training-task reports use different checkpoints")

    old_axial = float(baseline["terminal_metrics"]["axial_error_m"]["p50"])
    new_axial = float(candidate["terminal_metrics"]["axial_error_m"]["p50"])
    old_lateral = float(baseline["terminal_metrics"]["lateral_error_m"]["p50"])
    new_lateral = float(candidate["terminal_metrics"]["lateral_error_m"]["p50"])
    old_attitude = float(baseline["terminal_metrics"]["orientation_error_rad"]["p50"])
    new_attitude = float(candidate["terminal_metrics"]["orientation_error_rad"]["p50"])
    promoted = candidate["successes"] > baseline["successes"] and training["successes"] > 0
    return {
        "schema_version": 1,
        "title": "Rack-mouth-only insertion training intervention",
        "evidence_type": "simulation_only_insertion_distribution_intervention",
        "generated_utc": datetime.now(UTC).isoformat(),
        "intervention": {
            "parent_checkpoint_sha256": baseline["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
            "epochs_resumed": 400,
            "changed_variable": "reset station fixed at rack mouth (station 0)",
            "unchanged": [
                "geometry",
                "observations",
                "actions",
                "rewards",
                "success criteria",
                "phase budget",
                "fixed-to-compliant load path",
            ],
        },
        "identical_state_comparison": {
            "seed": baseline["seed"],
            "initial_state_sha256": baseline["initial_state_sha256"],
            "load_path": baseline["load_path"],
            "arms": {"v24": baseline, "v25_handoff_only": candidate},
            "median_change": {
                "axial_error_m": new_axial - old_axial,
                "lateral_error_m": new_lateral - old_lateral,
                "orientation_error_rad": new_attitude - old_attitude,
            },
        },
        "candidate_on_its_training_task": training,
        "decision": {
            "promoted": promoted,
            "reason": (
                "candidate produced no successful episode on the identical handoff or its own training task"
                if not promoted
                else "candidate improved the identical handoff and produced successes on its training task"
            ),
            "next": "retain this losing arm; expand from v24's successful late stations with a success-gated reverse curriculum",
        },
    }


def main() -> None:
    args = _parser().parse_args()
    inputs = {name: getattr(args, name) for name in ("baseline", "candidate", "training_task")}
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in inputs.items()}
    report = build_report(loaded["baseline"], loaded["candidate"], loaded["training_task"])
    report["raw_reports"] = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in inputs.items()
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preserved evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
