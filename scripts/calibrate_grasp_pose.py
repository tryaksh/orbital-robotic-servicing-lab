"""Solve the arm joint angles that put the finger pads on the blade handle.

Moving the grasp task's tool frame onto the pads leaves the old reset poses
pointing the fingers short of the handle, because those joint angles were
derived for a frame 74 mm further out. Rather than hand-tune six joint angles
per curriculum stage, this drives the corrected tool frame onto the handle with
the task's own differential IK controller and reports where it converged.

Nothing here trains anything. It is a kinematic calibration: run it, read the
joint angles, and paste them into ``CONTACT_INSERTION_STAGE_ARM_JOINT_POS``.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-Insertion-Contact-v0")
    parser.add_argument("--steps", type=int, default=400, help="Servo iterations per curriculum stage.")
    parser.add_argument(
        "--tolerance_m",
        type=float,
        default=0.002,
        help="Report failure if the tool frame cannot be driven this close to the handle centre.",
    )
    parser.add_argument("--report", type=Path, default=Path("artifacts/grasp_pose_calibration.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply, quat_inv

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.evaluation import round_floats
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import attached_blade_pose_world
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world

STAGES = (0, 1, 2)


def _handle_centre(task) -> torch.Tensor:
    position, orientation = attached_blade_pose_world(task)
    offset = position.new_tensor(task.cfg.scene.spare_blade.spawn.handle_offset).expand(position.shape[0], -1)
    return position + quat_apply(orientation, offset)


def main() -> dict[str, object]:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=len(STAGES))
        env_cfg.configure_robustness(0)
        # A servo run must not be interrupted by the task's own episode logic.
        # The timeout is the important one: at 12 s the episode resets the arm
        # to its start pose, which silently undoes the whole solve.
        # insertion_success stays: the curriculum term reads it by name.
        env_cfg.terminations.insertion_failed = None
        env_cfg.rewards.failure = None
        env_cfg.episode_length_s = 1.0e6
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        robot = task.scene["robot"]
        arm_ids, arm_names = robot.find_joints(
            ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            preserve_order=True,
        )
        # One environment per curriculum stage, so all three solve at once.
        stage_tensor = torch.tensor(STAGES, dtype=torch.long, device=task.device)
        task._insertion_curriculum_stage = stage_tensor.clone()
        env.reset()
        task._insertion_curriculum_stage = stage_tensor.clone()

        action = torch.zeros((len(STAGES), task.action_manager.total_action_dim), device=task.device)
        # The differential IK adds its delta in the robot *root* frame, and the
        # action term multiplies by a per-axis scale first, so a world-frame
        # error has to be rotated and descaled before it can be commanded.
        scale = torch.tensor(task.cfg.actions.arm.scale[:3], device=task.device).clamp_min(1.0e-6)
        with torch.inference_mode():
            for step in range(args.steps):
                tool, _ = end_effector_pose_world(task)
                error_w = _handle_centre(task) - tool
                error_b = quat_apply(quat_inv(robot.data.root_quat_w), error_w)
                action[:, 0:3] = (error_b / scale).clamp(-1.0, 1.0)
                env.step(action)
                if step % 50 == 0:
                    distances = torch.linalg.vector_norm(error_w, dim=-1) * 1000.0
                    print(f"[CALIB] step {step:4d}: residual mm {[round(float(v), 1) for v in distances]}")
            tool, _ = end_effector_pose_world(task)
            residual = torch.linalg.vector_norm(_handle_centre(task) - tool, dim=-1)
            joints = robot.data.joint_pos[:, arm_ids].clone()

        rows = []
        for index, stage in enumerate(STAGES):
            rows.append(
                {
                    "curriculum_stage": stage,
                    "residual_tool_to_handle_m": float(residual[index]),
                    "converged": bool(float(residual[index]) <= args.tolerance_m),
                    "arm_joint_pos_rad": [round(float(value), 7) for value in joints[index]],
                }
            )
            state = "converged" if rows[-1]["converged"] else "DID NOT CONVERGE"
            print(f"[CALIB] stage {stage}: residual {float(residual[index]) * 1000:.2f} mm ({state})")
            print(f"[CALIB]   {tuple(rows[-1]['arm_joint_pos_rad'])},")

        return {
            "status": "passed",
            "title": "Grasp-pose kinematic calibration",
            "evidence_type": "simulation_kinematic_calibration",
            "task": args.task,
            "tool_offset_pos": list(task.cfg.tool_offset_pos),
            "arm_joint_names": list(arm_names),
            "tolerance_m": args.tolerance_m,
            "servo_steps": args.steps,
            "all_stages_converged": all(row["converged"] for row in rows),
            "stages": rows,
        }
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    report: dict[str, object]
    try:
        report = main()
    except BaseException as exc:
        traceback.print_exc()
        report = {"status": "failed", "task": args.task, "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(round_floats(report), indent=2) + "\n", encoding="utf-8")
        print(f"[CALIB] wrote {args.report}")
        simulation_app.close()
