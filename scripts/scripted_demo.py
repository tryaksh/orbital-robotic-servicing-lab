"""Run a deterministic Cartesian blade-insertion expert in Isaac Sim."""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-BladeSwap-Play-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument(
        "--grasp_mode",
        choices=("kinematic", "compliant", "none"),
        default="kinematic",
        help="Kinematic is the reliable cinematic baseline; none tests raw contact grasping.",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("artifacts/scripted_demo_report.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


args = _parser().parse_args()
if "Play" in args.task or args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.assets import BLADE_HANDLE_OFFSET
from zero_g_blade_swap.tasks.blade_swap.mdp.commands import (
    gripper_grasp_orientation,
    gripper_handle_orientation_error,
)
from zero_g_blade_swap.tasks.blade_swap.mdp.expert import ScriptedBladeSwapExpert
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world


def _final_telemetry(base_env, actions: torch.Tensor) -> dict[str, object]:
    ee_pos, ee_quat = end_effector_pose_world(base_env)
    spare = base_env.scene["spare_blade"]
    handle_offset = spare.data.root_pos_w.new_tensor(BLADE_HANDLE_OFFSET).expand(base_env.num_envs, -1)
    handle_pos, handle_quat = combine_frame_transforms(spare.data.root_pos_w, spare.data.root_quat_w, handle_offset)
    origins = base_env.scene.env_origins
    goal = base_env.command_manager.get_command("blade_goal")
    gripper = base_env.action_manager.get_term("gripper")
    robot = base_env.scene["robot"]
    gripper_joint_ids, gripper_joint_names = robot.find_joints(list(gripper.cfg.joint_names), preserve_order=True)
    return {
        "phase": int(base_env._swap_phase[0]),
        "end_effector_position_m": (ee_pos[0] - origins[0]).tolist(),
        "end_effector_orientation_wxyz": ee_quat[0].tolist(),
        "active_goal_position_m": goal[0, :3].tolist(),
        "spare_blade_position_m": (spare.data.root_pos_w[0] - origins[0]).tolist(),
        "spare_handle_position_m": (handle_pos[0] - origins[0]).tolist(),
        "spare_handle_orientation_wxyz": handle_quat[0].tolist(),
        "desired_gripper_orientation_wxyz": gripper_grasp_orientation(handle_quat)[0].tolist(),
        "tcp_to_spare_handle_m": float(torch.linalg.vector_norm(ee_pos[0] - handle_pos[0])),
        "tcp_to_spare_handle_orientation_rad": float(
            gripper_handle_orientation_error(ee_quat, handle_quat)[0]
        ),
        "spare_linear_speed_mps": float(torch.linalg.vector_norm(spare.data.root_lin_vel_w[0])),
        "last_action": actions[0].tolist(),
        "gripper_raw_command": float(gripper.raw_actions[0, 0]),
        "gripper_joint_targets": gripper.processed_actions[0].tolist(),
        "gripper_joint_names": gripper_joint_names,
        "gripper_joint_positions": robot.data.joint_pos[0, gripper_joint_ids].tolist(),
        "grasp_mode": args.grasp_mode,
    }


def _nominal_demo_cfg():
    device = args.device or "cuda:0"
    cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps * cfg.decimation * cfg.sim.dt + 5.0)
    cfg.commands.blade_goal.position_jitter = (0.0, 0.0, 0.0)
    # The nominal cinematic baseline uses the explicit compliant attachment.
    # Keep the proxy handles visible, but prevent the unvalidated primitive
    # handle/Robotiq collision pair from injecting artificial squeeze torque.
    cfg.scene.blade.spawn.handle_collision_enabled = args.grasp_mode == "none"
    cfg.scene.spare_blade.spawn.handle_collision_enabled = args.grasp_mode == "none"

    # First prove nominal behavior.  Robust randomized replay is the next gate,
    # after the deterministic expert completes the real contact task.
    for name in (
        "blade_mass",
        "spare_blade_mass",
        "slot_sliding_friction",
        "slot_left_guide_sliding_friction",
        "slot_right_guide_sliding_friction",
        "base_wobble_excitation",
        "rack_stiction",
        "supply_stiction",
        "rack_albedo",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    if hasattr(cfg.events, "orbital_sun"):
        sun = cfg.events.orbital_sun.params
        sun["intensity_range"] = (5_000.0, 5_000.0)
        sun["pitch_range_deg"] = (45.0, 45.0)
        sun["yaw_range_deg"] = (225.0, 225.0)
        sun["color_temperature_range"] = (6_500.0, 6_500.0)
    if not args.headless:
        cfg.sim.render.rendering_mode = args.rendering_mode or "quality"
        cfg.sim.render.antialiasing_mode = "DLSS"
        cfg.sim.render.enable_reflections = True
    return cfg


def main() -> dict[str, object]:
    env = None
    try:
        cfg = _nominal_demo_cfg()
        render_mode = "rgb_array" if args.video else None
        env = gym.make(args.task, cfg=cfg, render_mode=render_mode)
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder="videos/scripted_demo",
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env.reset(seed=args.seed)
        base_env = env.unwrapped
        expert = ScriptedBladeSwapExpert(
            base_env,
            grasp_assist=args.grasp_mode != "none",
            kinematic_assist=args.grasp_mode == "kinematic",
        )
        expert.reset()
        maximum_phase = int(base_env._swap_phase.max())
        phase_first_seen = {str(maximum_phase): 0}
        termination_counts = {
            name: 0 for name in ("full_success", "time_out", "blade_lost", "mount_unstable", "non_finite")
        }
        steps = 0
        successes = 0
        actions = torch.zeros((base_env.num_envs, 7), device=base_env.device)
        while simulation_app.is_running() and steps < args.steps:
            started = time.perf_counter()
            with torch.inference_mode():
                actions = expert.compute_actions()
                _, _, terminated, truncated, _ = env.step(actions)
                phase = int(base_env._swap_phase.max())
                maximum_phase = max(maximum_phase, phase)
                phase_first_seen.setdefault(str(phase), steps + 1)
                for name in termination_counts:
                    count = int(base_env.termination_manager.get_term(name).sum())
                    termination_counts[name] += count
                    if name == "full_success":
                        successes += count
                done_ids = (terminated | truncated).nonzero().flatten()
                if len(done_ids) > 0:
                    expert.reset(done_ids)
            steps += 1
            if successes > 0:
                break
            if args.video and steps >= args.video_length:
                break
            delay = float(base_env.step_dt) - (time.perf_counter() - started)
            if args.real_time and delay > 0:
                time.sleep(delay)
        return {
            "status": "passed" if successes > 0 else "incomplete",
            "task": args.task,
            "device": str(base_env.device),
            "steps": steps,
            "successes": successes,
            "grasp_mode": args.grasp_mode,
            "handle_collision_enabled": args.grasp_mode == "none",
            "maximum_phase_reached": maximum_phase,
            "phase_first_seen_step": phase_first_seen,
            "termination_counts": termination_counts,
            "final_telemetry": _final_telemetry(base_env, actions),
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
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        simulation_app.close()
