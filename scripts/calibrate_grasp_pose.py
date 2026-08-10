"""Solve the arm joint angles that put the finger pads on the blade's interface.

Moving a grasp task's tool frame onto the pads leaves the old reset poses
pointing the fingers somewhere else, because those joint angles were derived for
a different frame. Rather than hand-tune six joint angles per curriculum stage,
this drives the corrected tool frame onto the target with the task's own
differential IK controller and reports where it converged.

Two modes:

* **Position only**, the default. Servos the tool frame onto the handle centre
  and leaves orientation wherever the reset pose put it. This is what the
  top-down contact grasp needs.
* **Six degrees of freedom**, when a target orientation is available. A head-on
  capture points the tool along the extraction axis instead of down, so the
  orientation is most of the problem and cannot be left to the reset pose.
  The target is read from the task's ``tool_target_rot`` or given explicitly.

Because a 90-degree reorientation is a long way for a differential servo to
travel, the six-degree-of-freedom mode starts several seeds in parallel, one per
environment, offsetting ``wrist_1_joint`` by a different quarter turn in each.
Whichever seed converges is the answer; the others cost nothing but GPU width.

Nothing here trains anything. It is a kinematic calibration: run it, read the
joint angles, and paste them into the task's stage poses.

Two traps in this script were found the hard way and are fixed. The IK delta is
applied in the robot *root* frame, not the world frame, and the episode timeout
must be disabled or the arm resets to its start pose mid-solve.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


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
    parser.add_argument(
        "--tolerance_rad",
        type=float,
        default=0.010,
        help="Orientation tolerance, used only when a target orientation is given.",
    )
    parser.add_argument(
        "--target_rot",
        type=float,
        nargs=4,
        default=None,
        help="World-frame target orientation as scalar-first quaternion. Defaults to the task's tool_target_rot.",
    )
    parser.add_argument(
        "--seed_wrist_1_offsets",
        type=float,
        nargs="+",
        default=None,
        help="Quarter turns added to wrist_1_joint to seed the solve. Defaults to none, or four quadrants in 6-DoF.",
    )
    parser.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--pin_blade",
        action="store_true",
        help=(
            "Hold the blade at its reset pose for the whole solve. In zero gravity a contact task's blade is a "
            "free body, so the swinging arm shoves it and every curriculum stage converges to the same answer."
        ),
    )
    parser.add_argument(
        "--finger_joint",
        type=float,
        default=None,
        help=(
            "Hold the fingers at this command for the whole solve. A capture task closes the gripper on reset, "
            "which makes the pads foul the interface they are supposed to be driven around."
        ),
    )
    parser.add_argument("--report", type=Path, default=Path("artifacts/grasp_pose_calibration.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be at least 1")
if not args.stages:
    parser.error("--stages must name at least one curriculum stage")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv, quat_mul

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.evaluation import round_floats
from zero_g_blade_swap.tasks.blade_swap.mdp.actions import (
    ROBOTIQ_2F85_COUPLING_SIGNS,
    ROBOTIQ_2F85_JOINT_NAMES,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import attached_blade_pose_world, reset_insertion_blade
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world


def _handle_centre(task) -> torch.Tensor:
    position, orientation = attached_blade_pose_world(task)
    offset = position.new_tensor(task.cfg.scene.spare_blade.spawn.handle_offset).expand(position.shape[0], -1)
    return position + quat_apply(orientation, offset)


def main() -> dict[str, object]:
    env = None
    try:
        stages = list(args.stages)
        target_rot = args.target_rot
        seed_offsets = args.seed_wrist_1_offsets

        env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=1)
        if target_rot is None:
            target_rot = getattr(env_cfg, "tool_target_rot", None)
        if seed_offsets is None:
            # A head-on pose is a quarter turn of wrist_1 away from a top-down
            # one, but which quarter turn depends on the joint's sign
            # convention, so try all four rather than assert one.
            seed_offsets = [0.0, 0.5 * math.pi, -0.5 * math.pi, math.pi] if target_rot is not None else [0.0]
        num_envs = len(stages) * len(seed_offsets)

        env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=num_envs)
        env_cfg.configure_robustness(0)
        # A servo run must not be interrupted by the task's own episode logic.
        # The timeout is the important one: at 12 s the episode resets the arm
        # to its start pose, which silently undoes the whole solve.
        # insertion_success stays: the curriculum term reads it by name.
        env_cfg.terminations.insertion_failed = None
        env_cfg.rewards.failure = None
        env_cfg.episode_length_s = 1.0e6
        if args.finger_joint is not None:
            # This script drives the fingers itself; the interval event would
            # overwrite the command every step.
            env_cfg.events.hold_gripper_closed = None
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        robot = task.scene["robot"]
        arm_ids, arm_names = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        finger_ids, finger_names = robot.find_joints(list(ROBOTIQ_2F85_JOINT_NAMES), preserve_order=True)
        if tuple(finger_names) != ROBOTIQ_2F85_JOINT_NAMES:
            raise RuntimeError(f"unexpected Robotiq joint order: {finger_names}")
        finger_targets = (
            torch.tensor(ROBOTIQ_2F85_COUPLING_SIGNS, device=task.device).mul(args.finger_joint).expand(num_envs, -1)
            if args.finger_joint is not None
            else None
        )

        # One environment per (stage, seed) pair, so every combination solves at
        # once. Stage varies fastest so the report groups by seed.
        stage_tensor = torch.tensor(stages, dtype=torch.long, device=task.device).repeat(len(seed_offsets))
        offset_tensor = torch.tensor(seed_offsets, dtype=torch.float32, device=task.device).repeat_interleave(
            len(stages)
        )
        task._insertion_curriculum_stage = stage_tensor.clone()
        env.reset()
        task._insertion_curriculum_stage = stage_tensor.clone()

        # Seed after the reset event has written the task's own start pose.
        seeded = robot.data.joint_pos[:, arm_ids].clone()
        seeded[:, 3] += offset_tensor
        robot.write_joint_state_to_sim(seeded, torch.zeros_like(seeded), joint_ids=arm_ids)
        if finger_targets is not None:
            robot.write_joint_state_to_sim(
                finger_targets, torch.zeros_like(finger_targets), joint_ids=finger_ids
            )
        # Re-place the blade explicitly. Setting the stage tensor before
        # ``env.reset()`` is not enough: the curriculum term rewrites it during
        # the reset, so every environment silently gets stage 0's blade pose and
        # the per-stage answers come out identical.
        blade = task.scene["spare_blade"]
        reset_ids = torch.arange(num_envs, device=task.device)
        reset_insertion_blade(task, reset_ids, **task.cfg.events.reset_blade.params)
        blade_start_pose = blade.data.root_state_w[:, :7].clone()
        blade_zero_velocity = torch.zeros((num_envs, 6), device=task.device)

        action = torch.zeros((num_envs, task.action_manager.total_action_dim), device=task.device)
        # The differential IK adds its delta in the robot *root* frame, and the
        # action term multiplies by a per-axis scale first, so a world-frame
        # error has to be rotated and descaled before it can be commanded.
        scale = torch.tensor(task.cfg.actions.arm.scale[:3], device=task.device).clamp_min(1.0e-6)
        angular_scale = torch.tensor(task.cfg.actions.arm.scale[3:], device=task.device).clamp_min(1.0e-6)
        desired_w = (
            torch.tensor(target_rot, dtype=torch.float32, device=task.device).expand(num_envs, -1)
            if target_rot is not None
            else None
        )

        with torch.inference_mode():
            for step in range(args.steps):
                if finger_targets is not None:
                    robot.set_joint_position_target(finger_targets, joint_ids=finger_ids)
                if args.pin_blade:
                    blade.write_root_pose_to_sim(blade_start_pose)
                    blade.write_root_velocity_to_sim(blade_zero_velocity)
                tool, tool_rot = end_effector_pose_world(task)
                error_w = _handle_centre(task) - tool
                error_b = quat_apply(quat_inv(robot.data.root_quat_w), error_w)
                action[:, 0:3] = (error_b / scale).clamp(-1.0, 1.0)
                if desired_w is not None:
                    # apply_delta_pose pre-multiplies the delta onto the
                    # end-effector quaternion expressed in the root frame, so
                    # the axis-angle command lives in the root frame too.
                    root_inverse = quat_inv(robot.data.root_quat_w)
                    current_b = quat_mul(root_inverse, tool_rot)
                    desired_b = quat_mul(root_inverse, desired_w)
                    delta = axis_angle_from_quat(quat_mul(desired_b, quat_inv(current_b)))
                    action[:, 3:6] = (delta / angular_scale).clamp(-1.0, 1.0)
                env.step(action)
                if step % 100 == 0:
                    distances = torch.linalg.vector_norm(error_w, dim=-1) * 1000.0
                    print(f"[CALIB] step {step:5d}: residual mm {[round(float(v), 1) for v in distances]}")

            tool, tool_rot = end_effector_pose_world(task)
            residual = torch.linalg.vector_norm(_handle_centre(task) - tool, dim=-1)
            angular_residual = (
                torch.linalg.vector_norm(axis_angle_from_quat(quat_mul(desired_w, quat_inv(tool_rot))), dim=-1)
                if desired_w is not None
                else torch.zeros_like(residual)
            )
            joints = robot.data.joint_pos[:, arm_ids].clone()

        rows = []
        for index in range(num_envs):
            converged = bool(
                float(residual[index]) <= args.tolerance_m
                and float(angular_residual[index]) <= (args.tolerance_rad if desired_w is not None else math.inf)
            )
            rows.append(
                {
                    "curriculum_stage": int(stage_tensor[index]),
                    "seed_wrist_1_offset_rad": round(float(offset_tensor[index]), 7),
                    "residual_tool_to_handle_m": float(residual[index]),
                    "residual_orientation_rad": float(angular_residual[index]),
                    "converged": converged,
                    "blade_pose_local": [
                        round(float(value - origin), 6)
                        for value, origin in zip(
                            blade_start_pose[index],
                            list(task.scene.env_origins[index]) + [0.0] * 4,
                            strict=True,
                        )
                    ],
                    "arm_joint_pos_rad": [round(float(value), 7) for value in joints[index]],
                }
            )
            state = "converged" if converged else "DID NOT CONVERGE"
            print(
                f"[CALIB] stage {rows[-1]['curriculum_stage']} seed {rows[-1]['seed_wrist_1_offset_rad']:+.4f}: "
                f"{float(residual[index]) * 1000:.2f} mm, {float(angular_residual[index]):.4f} rad ({state})"
            )
            print(f"[CALIB]   {tuple(rows[-1]['arm_joint_pos_rad'])},")

        converged_stages = {row["curriculum_stage"] for row in rows if row["converged"]}
        return {
            "status": "passed",
            "title": "Grasp-pose kinematic calibration",
            "evidence_type": "simulation_kinematic_calibration",
            "task": args.task,
            "tool_offset_pos": list(task.cfg.tool_offset_pos),
            "target_orientation_w": list(target_rot) if target_rot is not None else None,
            "degrees_of_freedom": 6 if target_rot is not None else 3,
            "arm_joint_names": list(arm_names),
            "tolerance_m": args.tolerance_m,
            "tolerance_rad": args.tolerance_rad if target_rot is not None else None,
            "servo_steps": args.steps,
            "held_finger_joint_rad": args.finger_joint,
            "blade_pinned_at_reset_pose": bool(args.pin_blade),
            "seed_wrist_1_offsets_rad": [round(float(value), 7) for value in seed_offsets],
            "all_stages_converged": converged_stages == set(stages),
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
