"""Compare learned and guarded insertion on exactly paired initial conditions.

The inputs are raw ``.npz`` episode tables written by
``run_workflow_demo.py``. Every condition/seed must have both controller arms,
and their recorded initial-state digest must match. Missing arms, mismatched
states, dirty source trees, or mixed learned checkpoints fail closed. The JSON
keeps both arms even when one loses; it is a characterization report, not a
promotion certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from zero_g_blade_swap.evaluation import (
    TERMINAL_METRIC_FIELDS,
    round_floats,
    summarize_terminal_episodes,
)
from zero_g_blade_swap.provenance import git_source_revision

PROTOCOL = "insertion_condition_v2"
SCHEMA_VERSION = 3
CONTROLLERS = ("guarded", "policy")
ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Paired raw episode .npz files.")
    parser.add_argument("--output", type=Path, required=True, help="Versioned JSON evidence path.")
    parser.add_argument(
        "--expected_policy_sha256",
        default=None,
        help="Optional required SHA-256 of the learned insertion checkpoint.",
    )
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and validate the raw schema, condition identity, and provenance."""

    runs: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            fields = tuple(str(name) for name in data["fields"])
            missing = [name for name in TERMINAL_METRIC_FIELDS if name not in fields]
            if missing:
                raise ValueError(f"{path} is missing required fields {missing}")
            rows = np.asarray(data["rows"], dtype=np.float64)
            metadata = json.loads(str(data["metadata"].item()))
        condition = metadata.get("evaluation_condition") or {}
        if condition.get("protocol") != PROTOCOL:
            raise ValueError(f"{path} does not declare protocol {PROTOCOL}")
        if not condition.get("initial_state_sha256"):
            raise ValueError(f"{path} has no initial-state digest")
        load_path = condition.get("load_path")
        if not isinstance(load_path, dict) or not load_path.get("source"):
            raise ValueError(f"{path} has no explicit load-path condition")
        controller = metadata.get("controller")
        if controller not in CONTROLLERS:
            raise ValueError(f"{path} controller must be one of {CONTROLLERS}, got {controller!r}")
        source = metadata.get("source_revision") or {}
        if not source.get("available") or not source.get("commit"):
            raise ValueError(f"{path} has no usable source revision")
        if source.get("dirty") is not False:
            raise ValueError(f"{path} was generated from a dirty tracked worktree")
        if rows.ndim != 2 or rows.shape[1] != len(fields) or rows.shape[0] == 0:
            raise ValueError(f"{path} has invalid or empty rows shape {rows.shape}")
        runs.append(
            {
                "path": path.resolve(),
                "file_sha256": _file_sha256(path),
                "fields": fields,
                "rows": rows,
                "metadata": metadata,
            }
        )
    if not runs:
        raise ValueError("at least one run is required")
    return runs


def _condition_key(run: dict[str, Any]) -> tuple[str, int | None, int]:
    metadata = run["metadata"]
    condition = metadata["evaluation_condition"]
    station = condition.get("station")
    return str(condition["kind"]), None if station is None else int(station), int(metadata["seed"])


def _counts(run: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_terminal_episodes(run["rows"], run["fields"], include_successful_metrics=False)
    counts = {
        key: summary[key]
        for key in (
            "episodes",
            "successes",
            "success_rate",
            "success_rate_wilson_95",
            "termination_reasons",
            "instability_terminations",
            "safety_abort_terminations",
            "non_finite_metric_episodes",
            "terminal_metrics",
        )
    }
    field_index = {name: index for index, name in enumerate(run["fields"])}
    for field, output in (
        ("latch_engaged_in_episode", "episodes_with_latch_engaged"),
        ("latch_compliant_in_episode", "episodes_with_latch_compliant"),
    ):
        if field not in field_index:
            raise ValueError(f"{run['path']} is missing required load-path field {field}")
        counts[output] = int((run["rows"][:, field_index[field]] > 0.5).sum())
    return counts


def _pooled_counts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    successes = 0
    episodes = 0
    for run in runs:
        counts = _counts(run)
        successes += int(counts["successes"])
        episodes += int(counts["episodes"])
    # Build a minimal compatible table so the shared Wilson implementation and
    # success semantics remain the single source of truth.
    rows = np.zeros((episodes, len(TERMINAL_METRIC_FIELDS)), dtype=np.float64)
    rows[:successes, TERMINAL_METRIC_FIELDS.index("success")] = 1.0
    return {
        key: value
        for key, value in summarize_terminal_episodes(
            rows, TERMINAL_METRIC_FIELDS, metric_fields=(), include_successful_metrics=False
        ).items()
        if key in ("episodes", "successes", "success_rate", "success_rate_wilson_95")
    }


def build_report(
    runs: list[dict[str, Any]], expected_policy_sha256: str | None = None
) -> dict[str, Any]:
    """Build a fail-closed, paired comparison while retaining every arm."""

    pairs: dict[tuple[str, int | None, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        key = _condition_key(run)
        controller = str(run["metadata"]["controller"])
        if controller in pairs.setdefault(key, {}):
            raise ValueError(f"duplicate {controller} arm for condition {key}")
        pairs[key][controller] = run

    incomplete = {key: sorted(set(CONTROLLERS) - set(arms)) for key, arms in pairs.items() if set(arms) != set(CONTROLLERS)}
    if incomplete:
        raise ValueError(f"every condition needs both controller arms; missing {incomplete}")

    source_commits = {run["metadata"]["source_revision"]["commit"] for run in runs}
    if len(source_commits) != 1:
        raise ValueError(f"paired runs came from different source commits: {sorted(source_commits)}")
    policy_hashes = {
        str(run["metadata"].get("checkpoints", {}).get("insert", "")).upper()
        for run in runs
        if run["metadata"]["controller"] == "policy"
    }
    if "" in policy_hashes or len(policy_hashes) != 1:
        raise ValueError(f"policy runs do not name exactly one insertion checkpoint: {sorted(policy_hashes)}")
    policy_hash = next(iter(policy_hashes))
    if expected_policy_sha256 and policy_hash != expected_policy_sha256.upper():
        raise ValueError(f"learned checkpoint is {policy_hash}, expected {expected_policy_sha256.upper()}")

    paired_rows = []
    for key in sorted(pairs, key=lambda item: (item[0], -1 if item[1] is None else item[1], item[2])):
        arms = pairs[key]
        hashes = {
            arms[controller]["metadata"]["evaluation_condition"]["initial_state_sha256"]
            for controller in CONTROLLERS
        }
        if len(hashes) != 1:
            raise ValueError(f"controller arms for {key} did not start from the same state: {sorted(hashes)}")
        load_paths = {
            json.dumps(
                arms[controller]["metadata"]["evaluation_condition"]["load_path"],
                sort_keys=True,
                separators=(",", ":"),
            )
            for controller in CONTROLLERS
        }
        if len(load_paths) != 1:
            raise ValueError(f"controller arms for {key} did not use the same load path")
        load_path_json = next(iter(load_paths))
        counts = {controller: _counts(arms[controller]) for controller in CONTROLLERS}
        guarded_rate = float(counts["guarded"]["success_rate"])
        policy_rate = float(counts["policy"]["success_rate"])
        losing_arm = "tie" if policy_rate == guarded_rate else "policy" if policy_rate < guarded_rate else "guarded"
        paired_rows.append(
            {
                "condition": {"kind": key[0], "station": key[1], "seed": key[2]},
                "initial_state_sha256": next(iter(hashes)),
                "load_path": json.loads(load_path_json),
                "load_path_sha256": hashlib.sha256(load_path_json.encode()).hexdigest(),
                "arms": {
                    controller: {
                        "raw_file": str(arms[controller]["path"]),
                        "raw_file_sha256": arms[controller]["file_sha256"],
                        **counts[controller],
                    }
                    for controller in CONTROLLERS
                },
                "policy_minus_guarded_success_rate": policy_rate - guarded_rate,
                "losing_arm": losing_arm,
            }
        )

    by_controller = {
        controller: _pooled_counts(
            [run for run in runs if run["metadata"]["controller"] == controller]
        )
        for controller in CONTROLLERS
    }
    guarded_rate = float(by_controller["guarded"]["success_rate"])
    policy_rate = float(by_controller["policy"]["success_rate"])
    policy_not_worse_everywhere = all(
        row["policy_minus_guarded_success_rate"] >= 0.0 for row in paired_rows
    ) and policy_rate >= guarded_rate
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "paired_conditioned_insertion_controller_comparison",
        "protocol": {
            "name": PROTOCOL,
            "paired_on": [
                "condition kind",
                "reset station when applicable",
                "seed",
                "initial-state SHA-256",
                "load-path SHA-256",
            ],
            "same_success_predicate": True,
            "same_settling_check": True,
            "same_phase_budget": True,
            "certification": False,
        },
        "source_revision": {"commit": next(iter(source_commits)), "dirty": False},
        "learned_policy": {"name": "v24", "checkpoint_sha256": policy_hash},
        "overall": {
            "by_controller": by_controller,
            "policy_minus_guarded_success_rate": policy_rate - guarded_rate,
        },
        "paired_conditions": paired_rows,
        "decision": {
            "policy_not_worse_on_every_paired_condition_and_pooled": policy_not_worse_everywhere,
            "recommended_controller": "policy" if policy_not_worse_everywhere else "guarded",
            "rule_frozen_before_results": (
                "replace guarded only if policy is not worse at every paired condition and in the pooled rate"
            ),
        },
        "scope_and_limitations": [
            "Simulation evidence only; no hardware contact or load qualification.",
            "Reset-station runs isolate insertion and are not end-to-end chain certificates.",
            "Reset-station runs reproduce the delayed fixed-to-compliant load path configured by the v24 task.",
            "Chain-handoff pairing is valid only when each batch contains at most one episode per environment.",
            "Both winning and losing controller arms are retained in paired_conditions.",
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    aggregation_source = git_source_revision(ROOT)
    if not aggregation_source.get("available") or not aggregation_source.get("commit"):
        raise SystemExit("cannot identify the aggregation source revision")
    if aggregation_source.get("dirty") is not False:
        raise SystemExit("refusing to aggregate evidence from a dirty tracked worktree")
    report = build_report(load_runs(args.runs), args.expected_policy_sha256)
    report["aggregation_source_revision"] = aggregation_source
    report["generated_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(report['paired_conditions'])} paired conditions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
