"""Plan the collision-aware UR10e branch change used by relocation.

The endpoints come from the passed kinematic workcell sweep. Lula RRT uses the
installed UR10e collision-sphere model and joint limits to find a self-collision
free configuration-space path between them. The module is already behind the
rack flares for this segment; rack-contact avoidance is enforced separately by
the workflow's clear-waypoint gate.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

START = (-0.998173, -2.363000, 2.648252, 2.856311, -0.572631, -1.570782)
TARGET = (-1.429480, -0.109389, -2.147392, -4.026341, 0.141319, 1.570785)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evidence/relocation_rrt_path.json"),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


args = _parser().parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.kit.app  # noqa: E402

extension_manager = omni.kit.app.get_app().get_extension_manager()
if not extension_manager.set_extension_enabled_immediate(
    "isaacsim.robot_motion.motion_generation", True
):
    raise RuntimeError("Could not enable the installed Isaac Sim motion-generation extension")

from isaacsim.robot_motion.motion_generation.lula import RRT  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    motion_root = Path("C:/isaac-sim/exts/isaacsim.robot_motion.motion_generation")
    ur10e = motion_root / "motion_policy_configs/universal_robots/ur10e"
    planner = RRT(
        robot_description_path=str(ur10e / "rmpflow/ur10e_robot_description.yaml"),
        urdf_path=str(ur10e / "ur10e.urdf"),
        rrt_config_path=str(root / "configs/ur10e_relocation_rrt.yaml"),
        end_effector_frame_name="tool0",
    )
    planner.set_random_seed(4070)
    planner.set_cspace_target(np.asarray(TARGET, dtype=np.float64))
    path = planner.compute_path(
        np.asarray(START, dtype=np.float64),
        np.empty((0,), dtype=np.float64),
    )
    if path is None or len(path) < 2:
        raise RuntimeError("Lula RRT found no path between the two passed workcell solutions")
    report = {
        "status": "passed",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "planner": "Isaac Sim Lula RRT",
        "seed": 4070,
        "robot": "UR10e",
        "robot_description": str(ur10e / "rmpflow/ur10e_robot_description.yaml"),
        "urdf": str(ur10e / "ur10e.urdf"),
        "planner_config": "configs/ur10e_relocation_rrt.yaml",
        "endpoint_source": "evidence/workcell_reach_solution.json",
        "scope": "self-collision and joint limits; executed only behind the rack flares",
        "start_joint_position_rad": list(START),
        "target_joint_position_rad": list(TARGET),
        "waypoint_count": int(len(path)),
        "waypoints_rad": np.asarray(path, dtype=np.float64).tolist(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[RRT] wrote {args.report}: {len(path)} sparse waypoints", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
