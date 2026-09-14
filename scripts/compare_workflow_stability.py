"""Run preserved and refined controllers on the same 24 seeded initial conditions.

Both arms run from one clean source revision, with eight environments per seed,
1900 control steps, randomized lighting and no video. This uses bounded cohorts,
not timeout-reset episode collection. All raw reports, rows, logs and commands
remain in a new directory, including failed physical outcomes. A completed
comparison is not a passed reliability gate; each aggregate retains the 95% gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SEEDS = (4070, 5070, 6070)
ENVIRONMENTS = 8
STEPS = 1900
TASK = "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
ARM_FLAGS = {
    "legacy_rail": (
        "--robot_rail_on_relocation", "--transit_motion_profile", "legacy",
        "--extraction_finish", "policy", "--guarded_insert_solver", "differential_ik",
    ),
    "refined_fixed_base": (
        "--transit_motion_profile", "quintic", "--transit_joint_trim",
        "--extraction_finish", "guarded", "--guarded_insert_solver", "absolute_ik",
    ),
}


def workflow_argv(project: Path, launcher: Path, output: Path, seed: int, arm: str) -> list[str]:
    policies = project / "policies" / "servicing_v2"
    return [
        str(launcher), str(project / "scripts" / "run_workflow_demo.py"),
        "--headless", "--workflow", "relocate", "--task", TASK, "--curriculum_stage", "0",
        "--grasp_checkpoint", str(policies / "capture_v7m130.pth"),
        "--extract_checkpoint", str(policies / "extract_v19noised.pth"),
        "--insert_checkpoint", str(policies / "insert_v13m130.pth"),
        "--perception_backend", "fiducial_pnp", "--module_velocity_source", "kinematics",
        "--fiducial_guard_bounds", "lead_in", "--insert_controller", "guarded",
        "--latch_on_release", "--latch_joint_mode", "fixed",
        "--latch_rated_force_n", "20000", "--latch_rated_torque_nm", "1000",
        "--latch_position_stiffness_n_per_m", "40000", "--latch_rotation_stiffness_nm_per_rad", "20000",
        "--mating_mode", "compliant", "--mating_force_cap_n", "1000",
        "--destination_channel_relief_m", "0.0046125",
        "--release_sequence", "simultaneous", "--rack_retention",
        "--num_envs", str(ENVIRONMENTS), "--steps", str(STEPS), "--seed", str(seed),
        *ARM_FLAGS[arm],
        "--report", str(output / "workflow_report.json"),
        "--episode_metrics", str(output / "episodes.npz"),
    ]


def read_cohort(path: Path, seed: int, commit: str) -> dict:
    """Validate the artifact itself; a process exit line is not episode evidence."""
    with np.load(path, allow_pickle=False) as archive:
        fields = tuple(str(value) for value in archive["fields"])
        rows = np.asarray(archive["rows"])
        metadata = json.loads(str(archive["metadata"].item()))
    if rows.shape != (ENVIRONMENTS, len(fields)) or len(set(fields)) != len(fields):
        raise ValueError(f"{path} must contain exactly eight rows with unique metric fields")
    if metadata.get("seed") != seed or metadata.get("task") != TASK:
        raise ValueError(f"{path} has the wrong seed or task")
    revision = metadata.get("source_revision", {})
    if revision.get("commit") != commit or revision.get("dirty") is not False:
        raise ValueError(f"{path} does not bind the expected clean source revision")
    success = rows[:, fields.index("success")]
    if not np.isin(success, [0.0, 1.0]).all():
        raise ValueError(f"{path} has invalid success indicators")
    return {
        "episodes": len(rows), "successes": int(success.sum()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "policy_set_sha256": metadata.get("checkpoint_sha256"),
    }


def clean_commit(project: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project, capture_output=True, text=True, check=True,
    )
    if status.stdout.strip():
        raise ValueError("Commit source changes before running a paired comparison")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--isaac_python", type=Path, default=Path("C:/isaac-sim/python.bat"))
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--dry_run", action="store_true", help="Write exact launch commands without starting Isaac")
    args = parser.parse_args()
    project, output, launcher = args.project_root.resolve(), args.output.resolve(), args.isaac_python.resolve()
    commit = clean_commit(project)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    manifest = {
        "source_commit": commit, "seeds": SEEDS, "environments_per_seed": ENVIRONMENTS,
        "steps": STEPS, "lighting": "randomized", "video": False,
        "scope": "Matched seeds and environment indices; 24 initial conditions per arm. No success criterion changes.",
        "runs": records,
    }
    manifest_path = output / "comparison.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    process_options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    for seed in SEEDS:
        for arm in ARM_FLAGS:
            if clean_commit(project) != commit:
                raise ValueError("Source revision changed during the comparison")
            run_dir = output / arm / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            argv = workflow_argv(project, launcher, run_dir, seed, arm)
            record = {"arm": arm, "seed": seed, "argv": argv}
            records.append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            if args.dry_run:
                continue
            print(f"Running {arm}, seed {seed}, eight environments", flush=True)
            with (run_dir / "execution.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    argv, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    check=False, **process_options,
                )
            record["exit_code"] = completed.returncode
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            record["cohort"] = read_cohort(run_dir / "episodes.npz", seed, commit)
            report = json.loads((run_dir / "workflow_report.json").read_text(encoding="utf-8"))
            if report.get("num_envs") != ENVIRONMENTS or report.get("visual_randomization") != "on":
                raise ValueError("Report does not match the fixed comparison protocol")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Measured {record['cohort']['successes']}/8 successful episodes", flush=True)
    if args.dry_run:
        print(f"Six launch commands written to {manifest_path}")
        return 0
    policy_sets = {record["cohort"]["policy_set_sha256"] for record in records}
    if len(policy_sets) != 1 or None in policy_sets:
        raise ValueError("Comparison arms did not execute the same checkpoint set")
    for arm in ARM_FLAGS:
        argv = [
            sys.executable, str(project / "scripts" / "aggregate_evaluation.py"), "--episodes",
            *(str(output / arm / f"seed{seed}" / "episodes.npz") for seed in SEEDS),
            "--output", str(output / arm / "aggregate.json"),
            "--title", f"Workflow stability comparison: {arm}, 24 seeded episodes",
            "--scope", manifest["scope"],
            "Simulation only, frozen policy weights; eight environments per seed with randomized lighting and no video.",
            "Strict supported settling, robot release and rack-only recheck remain unchanged. The 95% gate is unchanged.",
            "Controller recipe and transport mechanism change together; this comparison does not isolate either contribution.",
        ]
        with (output / arm / "aggregate.log").open("x", encoding="utf-8") as log:
            aggregate = subprocess.run(
                argv, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT,
                check=False, **process_options,
            )
        # The aggregator returns 2 for a valid measured result below its gate.
        if aggregate.returncode not in (0, 2) or not (output / arm / "aggregate.json").is_file():
            raise RuntimeError(f"Aggregation failed for {arm}; see the retained log")
    print(f"Comparison and both unchanged-gate aggregates saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
