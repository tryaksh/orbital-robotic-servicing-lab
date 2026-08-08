"""Pool per-run terminal episode metrics into one held-out evaluation report.

``play.py --episode_metrics`` writes one ``.npz`` of raw per-episode rows per
run.  This script merges those runs, recomputes exact pooled percentiles and
Wilson intervals, applies the promotion gate, and writes a single compact JSON
report.  It never touches Isaac Sim, so it can be run and tested anywhere.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from zero_g_blade_swap.evaluation import (
    TERMINAL_METRIC_FIELDS,
    concatenate_rows,
    group_rows,
    round_floats,
    summarize_terminal_episodes,
    wilson_interval,
)

STAGE_NAMES = {0: "near_stage_0", 1: "medium_stage_1", 2: "full_stage_2"}
STAGE_METRIC_FIELDS = (
    "axial_error_m",
    "lateral_error_m",
    "orientation_error_rad",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
    "cycle_time_s",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        help="Per-run .npz files, directories, or glob patterns written by play.py --episode_metrics.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report.")
    parser.add_argument("--title", required=True, help="Short human-readable name for the evaluated claim.")
    parser.add_argument(
        "--minimum_stage_success_rate",
        type=float,
        default=0.95,
        help="Promotion gate applied to every curriculum stage and to the pooled result.",
    )
    parser.add_argument(
        "--scope",
        nargs="*",
        default=[],
        help="Explicit limitation lines copied into the report.",
    )
    return parser


def _resolve_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.npz")))
        elif candidate.exists():
            paths.append(candidate)
        else:
            matches = sorted(Path().glob(pattern))
            if not matches:
                raise FileNotFoundError(f"no episode metric files matched {pattern!r}")
            paths.extend(matches)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("no episode metric files were resolved")
    return unique


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Read each run's rows and metadata, checking the recorded field order."""

    runs: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            fields = tuple(str(name) for name in data["fields"])
            if fields != TERMINAL_METRIC_FIELDS:
                raise ValueError(f"{path} recorded fields {fields}, expected {TERMINAL_METRIC_FIELDS}")
            runs.append(
                {
                    "path": path,
                    "rows": np.asarray(data["rows"], dtype=np.float64),
                    "metadata": json.loads(str(data["metadata"].item())),
                }
            )
    return runs


def _counts(rows: np.ndarray) -> dict[str, Any]:
    episodes = int(rows.shape[0])
    successes = int((rows[:, TERMINAL_METRIC_FIELDS.index("success")] > 0.5).sum())
    low, high = wilson_interval(successes, episodes)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "success_rate_wilson_95": {"low": low, "high": high},
    }


def build_report(runs: list[dict[str, Any]], title: str, minimum_stage_success_rate: float) -> dict[str, Any]:
    """Pool every run and apply the documented promotion gate."""

    rows = concatenate_rows(run["rows"] for run in runs)
    if rows.shape[0] == 0:
        raise ValueError("no episodes were recorded across the supplied runs")

    checkpoints = {run["metadata"].get("checkpoint_sha256") for run in runs}
    if len(checkpoints) != 1:
        raise ValueError(f"runs came from different checkpoints: {sorted(map(str, checkpoints))}")
    tasks = sorted({str(run["metadata"].get("task")) for run in runs})
    seeds = sorted({int(run["metadata"]["seed"]) for run in runs})
    levels = sorted({run["metadata"].get("robustness_level") for run in runs})

    overall = summarize_terminal_episodes(rows, include_successful_metrics=False)
    if overall["successes"] < overall["episodes"]:
        # Only meaningful when failures exist; otherwise it repeats the pooled block.
        overall["successful_episode_metrics"] = summarize_terminal_episodes(rows)["successful_episode_metrics"]
    by_stage = {
        STAGE_NAMES.get(stage, f"stage_{stage}"): {
            **_counts(block),
            **{
                key: value
                for key, value in summarize_terminal_episodes(
                    block,
                    metric_fields=STAGE_METRIC_FIELDS,
                    include_successful_metrics=False,
                ).items()
                if key in ("termination_reasons", "instability_terminations", "terminal_metrics")
            },
        }
        for stage, block in group_rows(rows, "curriculum_stage").items()
    }

    by_seed: dict[str, Any] = {}
    by_stage_and_seed: list[dict[str, Any]] = []
    for run in runs:
        seed = str(int(run["metadata"]["seed"]))
        by_seed.setdefault(seed, []).append(run["rows"])
        for stage, block in group_rows(run["rows"], "curriculum_stage").items():
            by_stage_and_seed.append(
                {
                    "stage": STAGE_NAMES.get(stage, f"stage_{stage}"),
                    "seed": int(run["metadata"]["seed"]),
                    **_counts(block),
                }
            )
    by_seed = {seed: _counts(concatenate_rows(blocks)) for seed, blocks in sorted(by_seed.items())}

    stage_rates = [entry["success_rate"] for entry in by_stage.values() if entry["success_rate"] is not None]
    instability = int(overall["instability_terminations"])
    non_finite = int(overall["non_finite_metric_episodes"])
    gate = {
        "minimum_stage_success_rate": minimum_stage_success_rate,
        "worst_stage_success_rate": min(stage_rates) if stage_rates else None,
        "pooled_success_rate": overall["success_rate"],
        "every_stage_meets_minimum": bool(stage_rates) and all(rate >= minimum_stage_success_rate for rate in stage_rates),
        "pooled_meets_minimum": overall["success_rate"] is not None
        and overall["success_rate"] >= minimum_stage_success_rate,
        "zero_instability_terminations": instability == 0,
        "zero_non_finite_metric_episodes": non_finite == 0,
    }
    gate["passed"] = bool(
        gate["every_stage_meets_minimum"]
        and gate["pooled_meets_minimum"]
        and gate["zero_instability_terminations"]
        and gate["zero_non_finite_metric_episodes"]
    )

    return {
        "title": title,
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "evidence_type": "simulation_only",
        "policy": {
            "algorithm": "PPO (RL-Games), deterministic evaluation",
            "checkpoint_sha256": next(iter(checkpoints)),
            "checkpoint": Path(str(runs[0]["metadata"].get("checkpoint", ""))).name,
            "grasp_model": "physx_fixed_joint_already_secured_abstraction",
        },
        "protocol": {
            "tasks": tasks,
            "robustness_levels": levels,
            "held_out_evaluation_seeds": seeds,
            "curriculum_stages": sorted(by_stage),
            "runs": len(runs),
            "gravity": "zero",
            "policy_rate_hz": 30,
            "physics_rate_hz": 120,
            "metric_capture": "per-episode terminal state recorded before Isaac Lab's automatic reset",
            "environments_per_run": sorted({run["metadata"].get("num_envs") for run in runs}),
            # Point at the single source of truth instead of restating thresholds
            # that could silently drift away from the task definition.
            "success_definition": (
                "zero_g_blade_swap.tasks.blade_swap.mdp.insertion:"
                "insertion_success_conditions via contact_insertion_success_mask"
            ),
        },
        "overall": overall,
        "by_curriculum_stage": by_stage,
        "by_evaluation_seed": by_seed,
        "by_stage_and_seed": sorted(by_stage_and_seed, key=lambda entry: (entry["stage"], entry["seed"])),
        "gate": gate,
    }


def main() -> int:
    args = _parser().parse_args()
    runs = load_runs(_resolve_inputs(args.episodes))
    report = build_report(runs, args.title, args.minimum_stage_success_rate)
    if args.scope:
        report["scope_and_limitations"] = list(args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
    gate = report["gate"]
    print(f"[INFO] Pooled episodes: {report['overall']['episodes']}")
    print(f"[INFO] Pooled success rate: {report['overall']['success_rate']}")
    print(f"[INFO] Worst stage success rate: {gate['worst_stage_success_rate']}")
    print(f"[INFO] Gate passed: {gate['passed']}")
    print(f"[INFO] Wrote {args.output}")
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
