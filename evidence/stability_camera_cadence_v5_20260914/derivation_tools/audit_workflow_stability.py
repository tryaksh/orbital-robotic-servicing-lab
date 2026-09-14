"""Supplement bounded-cohort NPZ scores with the existing physical mission checks.

This reads six completed runs and preserves their original NPZ scores and 95%
gate aggregates. It does not rewrite evidence, relax thresholds, or certify
camera/video behavior. Bounded harvest rows are ordered by environment index;
report rows are joined by their explicit, unique ``env`` keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from zero_g_blade_swap.service import verification
from zero_g_blade_swap.service.verification import ORIENTATION_LIMIT_RAD, POSITION_LIMIT_M, RACK_HOLD_S

SEEDS = (4070, 5070, 6070)
ARMS = ("legacy_rail", "refined_fixed_base")
ENVIRONMENTS = 8
TASK = "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"


def file_binding(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def number(value, *, minimum=0.0, maximum=math.inf) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and minimum <= value <= maximum


def environment_rows(section: dict, label: str, count: int) -> dict[int, dict]:
    values = section.get("observed_per_environment")
    if not isinstance(values, list):
        raise ValueError(f"{label}: missing environment records")
    result = {}
    for row in values:
        env = row.get("env") if isinstance(row, dict) else None
        if type(env) is not int or not 0 <= env < count or env in result:
            raise ValueError(f"{label}: invalid or duplicate environment key {env!r}")
        result[env] = row
    if set(result) != set(range(count)):
        raise ValueError(f"{label}: missing environment records")
    return result


def audit_environment(npz_success: bool, transit: dict, carried: dict, rack: dict, held: dict, release: dict) -> dict:
    """The physical per-environment requirements already used by verify_mission."""
    checks = {
        "npz_terminal_success": npz_success,
        "robot_carrier": transit.get("carrier") == "six_axis_robot",
        "entered_transit": carried.get("entered_transit") is True,
        "retained_throughout": carried.get("retained_throughout") is True,
        "transit_samples": number(carried.get("samples"), minimum=1),
        "tool_travel": number(carried.get("tool_travel_m"), minimum=0.001),
        "module_travel": number(carried.get("module_travel_m"), minimum=0.001),
        "transit_position": number(carried.get("max_position_drift_m"), maximum=POSITION_LIMIT_M),
        "transit_orientation": number(carried.get("max_orientation_drift_rad"), maximum=ORIENTATION_LIMIT_RAD),
        "rack_enabled": rack.get("enabled") is True,
        "rack_load_path": (
            rack.get("world_constraint") is False and rack.get("module_pose_write") is False
            and rack.get("joint_body0") == "Rack" and rack.get("joint_body1") == "SpareBlade"
        ),
        "rack_engaged_after_seating": held.get("engaged_after_measured_seating") is True,
        "rack_recheck_observed": held.get("full_rack_only_recheck_observed") is True,
        "rack_hold_duration": number(held.get("rack_only_interval_s"), minimum=RACK_HOLD_S),
        "rack_position": number(held.get("max_rack_to_module_position_drift_m"), maximum=POSITION_LIMIT_M),
        "rack_orientation": number(held.get("max_rack_to_module_orientation_drift_rad"), maximum=ORIENTATION_LIMIT_RAD),
        "latch_released": release.get("released_after_seating") is True,
        "hand_released": release.get("hand_opened_after_settling_verification") is True,
    }
    return {
        "npz_success": npz_success, "strict_physical_success": all(checks.values()),
        "checks": checks, "failed_checks": [key for key, passed in checks.items() if not passed],
    }


def audit_cohort(report: dict, success: np.ndarray, *, seed: int, count: int = ENVIRONMENTS) -> list[dict]:
    if (report.get("seed") != seed or report.get("task") != TASK
            or report.get("workflow") != "relocate" or report.get("num_envs") != count):
        raise ValueError("Report does not describe the expected bounded cohort")
    if success.shape != (count,) or not np.isin(success, [0.0, 1.0]).all():
        raise ValueError("NPZ must contain one binary success value per environment")
    sections = {}
    tables = {}
    for key in ("robot_carried_transit", "destination_rack_retention", "capture_interface"):
        section = report.get(key)
        if not isinstance(section, dict):
            raise ValueError(f"{key}: missing section")
        sections[key] = section
        tables[key] = environment_rows(section, key, count)
    rows = []
    for env in range(count):
        result = audit_environment(
            bool(success[env]), sections["robot_carried_transit"], tables["robot_carried_transit"][env],
            sections["destination_rack_retention"], tables["destination_rack_retention"][env],
            tables["capture_interface"][env],
        )
        rows.append({"seed": seed, "env": env, **result})
    return rows


def load_run(root: Path, arm: str, seed: int, commit: str) -> tuple[list[dict], list[dict], str]:
    directory = root / arm / f"seed{seed}"
    report_path, metrics_path = directory / "workflow_report.json", directory / "episodes.npz"
    log_path = directory / "execution.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    horizons = re.findall(r"-> episode ([0-9.]+) s", log)
    preflights = re.findall(r"\[PLAN\] visual occupancy preflight passed=(\d+)/(\d+)", log)
    if len(horizons) != 1 or float(horizons[0]) <= 1900.0 / 30.0:
        raise ValueError(f"{directory}: episode timeout could invalidate bounded-row identity")
    if len(preflights) != 1 or int(preflights[0][1]) != ENVIRONMENTS:
        raise ValueError(f"{directory}: missing or repeated preflight; cannot join potentially reset episodes")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(metrics_path, allow_pickle=False) as archive:
        fields = tuple(str(value) for value in archive["fields"])
        rows = np.asarray(archive["rows"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"].item()))
    if rows.shape != (ENVIRONMENTS, len(fields)) or len(set(fields)) != len(fields):
        raise ValueError(f"{metrics_path}: expected eight rows and unique fields")
    if metadata.get("seed") != seed or metadata.get("task") != TASK:
        raise ValueError(f"{metrics_path}: wrong seed or task")
    for payload in (report, metadata):
        revision = payload.get("source_revision", {})
        if revision.get("commit") != commit or revision.get("dirty") is not False:
            raise ValueError(f"{directory}: source is missing, dirty, or different from the comparison")
    policy_set = report.get("policy_set_sha256")
    recorded_set = metadata.get("checkpoint_sha256")
    if not isinstance(policy_set, str) or not isinstance(recorded_set, str) or policy_set.lower() != recorded_set.lower():
        raise ValueError(f"{directory}: report and rows disagree on the policy set")
    audited = audit_cohort(report, rows[:, fields.index("success")], seed=seed)
    return audited, [file_binding(path, root) for path in (report_path, metrics_path, log_path)], policy_set.lower()


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    npz_passes = sum(row["npz_success"] for row in rows)
    strict_passes = sum(row["strict_physical_success"] for row in rows)
    return {
        "episodes": count, "npz_successes": npz_passes, "npz_success_rate": npz_passes / count,
        "strict_physical_successes": strict_passes, "strict_physical_success_rate": strict_passes / count,
        "npz_passes_rejected_by_physical_audit": [
            {"seed": row["seed"], "env": row["env"], "failed_checks": row["failed_checks"]}
            for row in rows if row["npz_success"] and not row["strict_physical_success"]
        ],
    }


def paired_changes(baseline: list[dict], refined: list[dict]) -> dict:
    if [(row["seed"], row["env"]) for row in baseline] != [(row["seed"], row["env"]) for row in refined]:
        raise ValueError("Paired rows do not identify the same ordered initial conditions")
    result = {}
    for field, label in (("npz_success", "npz"), ("strict_physical_success", "strict_physical")):
        for before, after, change in ((True, False, "pass_to_fail"), (False, True, "fail_to_pass")):
            result[f"{label}_{change}"] = [
                {"seed": old["seed"], "env": old["env"], "baseline_failed_checks": old["failed_checks"],
                 "refined_failed_checks": new["failed_checks"]}
                for old, new in zip(baseline, refined, strict=True)
                if old[field] is before and new[field] is after
            ]
    return result


def build_audit(root: Path) -> dict:
    manifest_path = root / "comparison.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("seeds") != list(SEEDS) or manifest.get("environments_per_seed") != ENVIRONMENTS
            or manifest.get("steps") != 1900 or manifest.get("video") is not False
            or manifest.get("lighting") != "randomized"):
        raise ValueError("Comparison manifest does not describe the fixed bounded protocol")
    commands = manifest.get("runs")
    if not isinstance(commands, list) or len(commands) != 6:
        raise ValueError("Comparison must contain all six uniquely identified commands")
    seen = set()
    for run in commands:
        key = (run.get("arm"), run.get("seed"))
        if key not in {(arm, seed) for arm in ARMS for seed in SEEDS} or key in seen:
            raise ValueError("Comparison command has an invalid or duplicate arm/seed")
        seen.add(key)
        argv = run.get("argv", [])
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("Comparison command is malformed")
        if any(flag in argv for flag in ("--episodes", "--video", "--stable_lighting")):
            raise ValueError("Comparison command changes bounded collection or rendering protocol")
        for flag, value in (("--num_envs", "8"), ("--steps", "1900"), ("--seed", str(run["seed"]))):
            if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag) + 1] != value:
                raise ValueError(f"Comparison command has wrong {flag}")
    commit = manifest["source_commit"]
    inputs = [file_binding(manifest_path, root)]
    by_arm = {arm: [] for arm in ARMS}
    policy_sets = set()
    for seed in SEEDS:
        for arm in ARMS:
            rows, bindings, policy_set = load_run(root, arm, seed, commit)
            by_arm[arm].extend(rows)
            inputs.extend(bindings)
            policy_sets.add(policy_set)
    if len(policy_sets) != 1:
        raise ValueError("Paired arms have different checkpoint sets")
    aggregates = {}
    for arm in ARMS:
        path = root / arm / "aggregate.json"
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        if aggregate["overall"]["episodes"] != 24:
            raise ValueError(f"{path}: original aggregate does not contain 24 episodes")
        aggregates[arm] = {"input": file_binding(path, root), "original_gate": aggregate["gate"]}
        inputs.append(file_binding(path, root))
    by_seed = []
    for seed in SEEDS:
        selected = {arm: [row for row in rows if row["seed"] == seed] for arm, rows in by_arm.items()}
        by_seed.append({
            "seed": seed, "arms": {arm: summarize(rows) for arm, rows in selected.items()},
            "paired_changes": paired_changes(*(selected[arm] for arm in ARMS)),
        })
    return {
        "evidence_type": "supplemental_bounded_cohort_physical_audit",
        "source_commit": commit, "policy_set_sha256": next(iter(policy_sets)),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mission_verifier_sha256": hashlib.sha256(Path(verification.__file__).read_bytes()).hexdigest(),
        "thresholds_from_mission_verifier": {
            "position_limit_m": POSITION_LIMIT_M, "orientation_limit_rad": ORIENTATION_LIMIT_RAD,
            "rack_hold_s": RACK_HOLD_S,
        },
        "scope_and_limitations": [
            "Read-only supplemental physical audit; all original NPZ scores and 95% gate aggregates remain unchanged.",
            "NPZ row i maps to environment i only for these bounded cohorts, harvested with arange and no episode-reset collection.",
            "Hashed logs must show an episode timeout beyond the run horizon and exactly one initial eight-environment preflight. Repeated preflight rejects possible episode mixing; repeated PhysX joint warnings do not identify resets.",
            "Explicit reset counters were not recorded. A reset too late to trigger another preflight clears NPZ frozen success and cannot create a false strict success, but its precise timing cannot be recovered here.",
            "The audit adds existing robot-retention, rack-only-hold and support-release mission checks to recorded NPZ terminal success.",
            "It does not independently revalidate camera detections, video, or all seven seating predicates; terminal seating comes from NPZ success.",
            "Three seeds and eight environments per seed, simulation only. Controller and transfer-mechanism changes are combined.",
        ],
        "inputs": inputs, "original_aggregates": aggregates,
        "arms": {arm: summarize(rows) for arm, rows in by_arm.items()},
        "by_seed": by_seed, "paired_changes": paired_changes(*(by_arm[arm] for arm in ARMS)),
        "episode_results": by_arm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing evidence is never overwritten")
    args = parser.parse_args()
    result = build_audit(args.comparison_dir.resolve())
    # Build before opening the output so an incomplete comparison leaves no
    # partial audit that could be mistaken for a complete result.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
