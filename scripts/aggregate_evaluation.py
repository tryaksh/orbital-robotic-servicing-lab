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
    BLADE_MASS_FIELD,
    TERMINAL_METRIC_FIELDS,
    align_rows,
    bucket_success_rates,
    concatenate_rows,
    group_rows,
    round_floats,
    summarize_terminal_episodes,
    wilson_interval,
)

STAGE_NAMES = {0: "near_stage_0", 1: "medium_stage_1", 2: "full_stage_2"}
# The project's documented promotion rule for randomized physics buckets.
MASS_BUCKET_MINIMUM = 0.80
# Level-2 training range, used only when a run recorded no range at all.
DEFAULT_MASS_RANGE = (5.0, 15.0)
STAGE_METRIC_FIELDS = (
    "axial_error_m",
    "lateral_error_m",
    "orientation_error_rad",
    "blade_linear_velocity_mps",
    "blade_angular_velocity_radps",
    "cycle_time_s",
    "peak_contact_force_n",
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
    """Read each run's rows, field names, and metadata.

    Runs may carry extra optional columns, so the core fields are required but
    the full field list is kept per run and aligned by name when pooling.
    """

    runs: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            fields = tuple(str(name) for name in data["fields"])
            missing = [name for name in TERMINAL_METRIC_FIELDS if name not in fields]
            if missing:
                raise ValueError(f"{path} is missing required fields {missing}")
            runs.append(
                {
                    "path": path,
                    "fields": fields,
                    "rows": np.asarray(data["rows"], dtype=np.float64),
                    "metadata": json.loads(str(data["metadata"].item())),
                }
            )
    return runs


def pooled_fields(runs: list[dict[str, Any]]) -> tuple[str, ...]:
    """Core fields first, then any optional column that some run recorded."""

    extra = sorted({name for run in runs for name in run["fields"]} - set(TERMINAL_METRIC_FIELDS))
    return (*TERMINAL_METRIC_FIELDS, *extra)


def _stress_label(stress: dict[str, Any]) -> str:
    """Short, sortable name for one point in the stress sweep."""

    scale = float(stress.get("pose_noise_scale", 1.0) or 1.0)
    mass = stress.get("blade_mass_range_kg")
    bias = stress.get("belief_bias_mm")
    # The pose-belief sweep varies only this axis, so lead the label with it:
    # sorted output then reads as the curve it is meant to produce.
    threshold = stress.get("force_threshold_n")
    label = f"belief_bias_{bias:05.2f}mm_" if bias is not None else ""
    # A commanded force budget is a swept axis in its own right, and pooling two
    # budgets into one row would average away the modulation being measured.
    label += f"force_threshold_{threshold:05.1f}N_" if threshold is not None else ""
    label += f"pose_noise_x{scale:05.2f}"
    if mass:
        label += f"_mass_{mass[0]:g}-{mass[1]:g}kg"
    if stress.get("lead_in_present") is False:
        label += "_no_lead_in"
    return label


def _pooled_mass_range(runs: list[dict[str, Any]]) -> tuple[float, float]:
    """Widest blade mass range any run sampled, so no run's tail is clipped."""

    ranges = []
    for run in runs:
        stress = run["metadata"].get("stress") or {}
        configured = stress.get("blade_mass_range_kg") or stress.get("trained_blade_mass_kg")
        if configured:
            ranges.append((float(configured[0]), float(configured[1])))
    if not ranges:
        return DEFAULT_MASS_RANGE
    return min(item[0] for item in ranges), max(item[1] for item in ranges)


def _counts(rows: np.ndarray, fields: tuple[str, ...] = TERMINAL_METRIC_FIELDS) -> dict[str, Any]:
    episodes = int(rows.shape[0])
    successes = int((rows[:, fields.index("success")] > 0.5).sum())
    low, high = wilson_interval(successes, episodes)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "success_rate_wilson_95": {"low": low, "high": high},
    }


def build_report(runs: list[dict[str, Any]], title: str, minimum_stage_success_rate: float) -> dict[str, Any]:
    """Pool every run and apply the documented promotion gate."""

    fields = pooled_fields(runs)
    aligned = [align_rows(run["rows"], run["fields"], fields) for run in runs]
    rows = concatenate_rows(aligned, fields)
    if rows.shape[0] == 0:
        raise ValueError("no episodes were recorded across the supplied runs")

    checkpoints = {run["metadata"].get("checkpoint_sha256") for run in runs}
    if len(checkpoints) != 1:
        raise ValueError(f"runs came from different checkpoints: {sorted(map(str, checkpoints))}")
    source_revisions = [run["metadata"].get("source_revision") for run in runs]
    sources_present = [source for source in source_revisions if isinstance(source, dict)]
    if sources_present and len(sources_present) != len(runs):
        raise ValueError("runs mix recorded and missing source revisions")
    source_revision = None
    if sources_present:
        unusable = [
            source
            for source in sources_present
            if not source.get("available") or not source.get("commit") or source.get("dirty") is not False
        ]
        if unusable:
            raise ValueError("runs must come from available, clean source revisions")
        source_commits = {str(source["commit"]) for source in sources_present}
        if len(source_commits) != 1:
            raise ValueError(f"runs came from different source commits: {sorted(source_commits)}")
        source_revision = dict(sources_present[0])
    tasks = sorted({str(run["metadata"].get("task")) for run in runs})
    seeds = sorted({int(run["metadata"]["seed"]) for run in runs})
    levels = sorted({run["metadata"].get("robustness_level") for run in runs})

    overall = summarize_terminal_episodes(rows, fields, include_successful_metrics=False)
    if overall["successes"] < overall["episodes"]:
        # Only meaningful when failures exist; otherwise it repeats the pooled block.
        overall["successful_episode_metrics"] = summarize_terminal_episodes(rows, fields)[
            "successful_episode_metrics"
        ]
    by_stage = {
        STAGE_NAMES.get(stage, f"stage_{stage}"): {
            **_counts(block, fields),
            **{
                key: value
                for key, value in summarize_terminal_episodes(
                    block,
                    fields,
                    metric_fields=STAGE_METRIC_FIELDS,
                    include_successful_metrics=False,
                ).items()
                if key in ("termination_reasons", "instability_terminations", "terminal_metrics")
            },
        }
        for stage, block in group_rows(rows, "curriculum_stage", fields).items()
    }

    by_seed: dict[str, Any] = {}
    by_stage_and_seed: list[dict[str, Any]] = []
    for run, table in zip(runs, aligned, strict=True):
        seed = str(int(run["metadata"]["seed"]))
        by_seed.setdefault(seed, []).append(table)
        for stage, block in group_rows(table, "curriculum_stage", fields).items():
            by_stage_and_seed.append(
                {
                    "stage": STAGE_NAMES.get(stage, f"stage_{stage}"),
                    "seed": int(run["metadata"]["seed"]),
                    **_counts(block, fields),
                }
            )
    by_seed = {seed: _counts(concatenate_rows(blocks, fields), fields) for seed, blocks in sorted(by_seed.items())}

    mass_range = _pooled_mass_range(runs)
    mass_buckets = (
        bucket_success_rates(rows, BLADE_MASS_FIELD, mass_range[0], mass_range[1], fields)
        if BLADE_MASS_FIELD in fields
        else None
    )

    # A run evaluated outside the training distribution characterizes the
    # capability envelope; it is not certification and must never be pooled
    # into one without saying so.
    stresses = [run["metadata"].get("stress") or {} for run in runs]
    out_of_distribution = any(bool(item.get("out_of_distribution")) for item in stresses)
    by_stress: dict[str, Any] = {}
    for table, stress in zip(aligned, stresses, strict=True):
        by_stress.setdefault(_stress_label(stress), []).append(table)
    by_stress = {
        label: _counts(concatenate_rows(blocks, fields), fields) for label, blocks in sorted(by_stress.items())
    }

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
    if mass_buckets is not None:
        # Level 2 onward must also hold up across the randomized mass range,
        # not just on average, so a heavy-blade weakness cannot hide in a pool.
        worst_bucket = mass_buckets["minimum_observed_success_rate"]
        gate["blade_mass_range_kg"] = list(mass_range)
        gate["minimum_mass_bucket_success_rate"] = MASS_BUCKET_MINIMUM
        gate["worst_mass_bucket_success_rate"] = worst_bucket
        gate["every_mass_bucket_meets_minimum"] = worst_bucket is not None and worst_bucket >= MASS_BUCKET_MINIMUM
    gate["passed"] = bool(
        gate["every_stage_meets_minimum"]
        and gate["pooled_meets_minimum"]
        and gate["zero_instability_terminations"]
        and gate["zero_non_finite_metric_episodes"]
        and gate.get("every_mass_bucket_meets_minimum", True)
    )
    if out_of_distribution:
        # Deliberately degrading a policy is the point of a sweep, so a failed
        # gate here is a measurement, not a regression.
        gate["applies"] = False
        gate["note"] = "capability envelope sweep; the gate certifies in-distribution runs only"

    report = {
        "title": title,
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "evidence_type": "simulation_capability_envelope" if out_of_distribution else "simulation_only",
        "out_of_distribution": out_of_distribution,
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
            # Force policies are only comparable when the abort limit that
            # truncates their force distribution is the same.
            "belief_bias_mm": sorted(
                {
                    run["metadata"]["stress"]["belief_bias_mm"]
                    for run in runs
                    if (run["metadata"].get("stress") or {}).get("belief_bias_mm") is not None
                }
            )
            or None,
            "trained_belief_bias_ceiling_mm": sorted(
                {
                    run["metadata"]["stress"]["trained_belief_bias_ceiling_mm"]
                    for run in runs
                    if (run["metadata"].get("stress") or {}).get("trained_belief_bias_ceiling_mm") is not None
                }
            )
            or None,
            "contact_force_limit_n": sorted(
                {
                    run["metadata"]["contact_force_limit_n"]
                    for run in runs
                    if run["metadata"].get("contact_force_limit_n") is not None
                }
            )
            or None,
            # Point at the single source of truth instead of restating thresholds
            # that could silently drift away from the task definition.
            "success_definition": (
                "zero_g_blade_swap.tasks.blade_swap.mdp.insertion:"
                "insertion_success_conditions via contact_insertion_success_mask"
            ),
        },
        "overall": overall,
        **({"by_stress_setting": by_stress} if len(by_stress) > 1 else {}),
        **({"by_blade_mass_bucket": mass_buckets} if mass_buckets is not None else {}),
        "by_curriculum_stage": by_stage,
        "by_evaluation_seed": by_seed,
        "by_stage_and_seed": sorted(by_stage_and_seed, key=lambda entry: (entry["stage"], entry["seed"])),
        "gate": gate,
    }
    if source_revision is not None:
        report["source_revision"] = source_revision
    return report


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
