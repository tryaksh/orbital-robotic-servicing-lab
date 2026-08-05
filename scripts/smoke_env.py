"""Run deterministic teacher and vision smoke tests against the installed task."""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("teacher", "vision", "all"), default="all")
    parser.add_argument("--teacher_steps", type=int, default=100)
    parser.add_argument("--teacher_envs", type=int, default=1)
    parser.add_argument("--mount_stability_steps", type=int, default=300)
    parser.add_argument("--vision_steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke_report.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.teacher_steps < 100:
    parser.error("--teacher_steps must be at least 100")
if args.vision_steps < 2:
    parser.error("--vision_steps must be at least 2")
args.headless = True
args.enable_cameras = args.profile in ("vision", "all")
# AppLauncher receives the parsed namespace. Clearing argv prevents application
# plugins from seeing project-specific flags such as --profile and --output.
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import axis_angle_from_quat, combine_frame_transforms, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.assets import BLADE_HANDLE_OFFSET
from zero_g_blade_swap.tasks.blade_swap.mdp.observations import end_effector_pose_world


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_leaves(child)


def _assert_finite(value, label: str) -> None:
    for tensor in _tensor_leaves(value):
        _assert(bool(torch.isfinite(tensor).all()), f"{label} contains NaN or Inf")


def _gpu_memory_mib() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(int(line) for line in completed.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _random_actions(env, scale: float = 0.2) -> torch.Tensor:
    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    return torch.empty((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device, dtype=torch.float32).uniform_(
        -scale, scale
    )


def _step_and_validate(env, actions: torch.Tensor):
    observations, reward, terminated, truncated, extras = env.step(actions)
    _assert_finite(observations, "observations")
    _assert_finite(reward, "reward")
    _assert_finite(terminated, "terminated")
    _assert_finite(truncated, "truncated")
    return observations, reward, terminated, truncated, extras


def _observation_shapes(observations: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in observations.items()}


def _geometry_snapshot(env) -> dict[str, object]:
    ee_position, ee_rotation = end_effector_pose_world(env.unwrapped)
    ee_position = ee_position - env.unwrapped.scene.env_origins
    handle_offset = ee_position.new_tensor(BLADE_HANDLE_OFFSET).expand(env.unwrapped.num_envs, -1)

    handles: dict[str, torch.Tensor] = {}
    for name in ("blade", "spare_blade"):
        blade = env.unwrapped.scene[name]
        local_position = blade.data.root_pos_w - env.unwrapped.scene.env_origins
        handle_position, _ = combine_frame_transforms(
            local_position,
            blade.data.root_quat_w,
            handle_offset,
        )
        handles[name] = handle_position

    command = env.unwrapped.command_manager.get_command("blade_goal")
    phase = env.unwrapped._swap_phase
    return {
        "robot_joint_positions_first_env": [float(value) for value in env.unwrapped.scene["robot"].data.joint_pos[0]],
        "end_effector_position_first_env_m": [float(value) for value in ee_position[0]],
        "end_effector_rotation_first_env_wxyz": [float(value) for value in ee_rotation[0]],
        "failed_handle_position_first_env_m": [float(value) for value in handles["blade"][0]],
        "spare_handle_position_first_env_m": [float(value) for value in handles["spare_blade"][0]],
        "active_goal_position_first_env_m": [float(value) for value in command[0, :3]],
        "active_goal_rotation_first_env_wxyz": [float(value) for value in command[0, 3:7]],
        "phase_first_env": int(phase[0]),
        "distance_to_spare_handle_first_env_m": float(torch.linalg.vector_norm(ee_position[0] - handles["spare_blade"][0])),
    }


def _mount_relative_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    anchor = env.unwrapped.scene["mount_anchor"]
    relative_position, relative_quaternion = subtract_frame_transforms(
        anchor.data.root_pos_w,
        anchor.data.root_quat_w,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )
    return relative_position, axis_angle_from_quat(relative_quaternion)


def _natural_mount_stability_check(env) -> dict[str, object]:
    zero_actions = torch.zeros(
        (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device
    )
    termination_counts = {name: 0 for name in ("mount_unstable", "blade_lost", "non_finite")}
    maximum_translation_axis = 0.0
    maximum_rotation_axis = 0.0
    maximum_blade_speed = {name: 0.0 for name in ("blade", "spare_blade")}
    first_loss_pre_step: dict[str, object] | None = None
    first_milestone_rate = None
    for step in range(args.mount_stability_steps):
        pre_step = {
            name: {
                "position": (env.unwrapped.scene[name].data.root_pos_w - env.unwrapped.scene.env_origins).clone(),
                "velocity": env.unwrapped.scene[name].data.root_lin_vel_w.clone(),
            }
            for name in ("blade", "spare_blade")
        }
        _step_and_validate(env, zero_actions)
        for name in termination_counts:
            termination_counts[name] += int(env.unwrapped.termination_manager.get_term(name).sum())
        relative_position, relative_rotation = _mount_relative_pose(env)
        maximum_translation_axis = max(maximum_translation_axis, float(relative_position.abs().max()))
        maximum_rotation_axis = max(maximum_rotation_axis, float(relative_rotation.abs().max()))
        for name in maximum_blade_speed:
            maximum_blade_speed[name] = max(
                maximum_blade_speed[name],
                float(torch.linalg.vector_norm(pre_step[name]["velocity"], dim=-1).max()),
            )
        lost_mask = env.unwrapped.termination_manager.get_term("blade_lost")
        if bool(lost_mask.any()) and first_loss_pre_step is None:
            lost_ids = lost_mask.nonzero().flatten()
            first_loss_pre_step = {
                "step": step,
                "environment_ids": [int(value) for value in lost_ids],
                **{
                    name: {
                        "positions_m": pre_step[name]["position"][lost_ids].tolist(),
                        "velocities_mps": pre_step[name]["velocity"][lost_ids].tolist(),
                    }
                    for name in pre_step
                },
            }
        if step == 0:
            reward_terms = dict(env.unwrapped.reward_manager.get_active_iterable_terms(0))
            first_milestone_rate = float(reward_terms["phase_milestone"][0])

    _assert(termination_counts["mount_unstable"] == 0, f"Natural wobble tripped mount: {termination_counts}")
    _assert(
        termination_counts["blade_lost"] == 0,
        f"Stable scene lost a blade: {termination_counts}; pre-step={first_loss_pre_step}",
    )
    _assert(termination_counts["non_finite"] == 0, f"Stable scene became non-finite: {termination_counts}")
    _assert(abs(float(first_milestone_rate)) < 1.0e-8, f"Reset leaked phase reward: {first_milestone_rate}")
    return {
        "steps": args.mount_stability_steps,
        "mount_terminations": termination_counts["mount_unstable"],
        "maximum_translation_axis_m": maximum_translation_axis,
        "maximum_rotation_axis_rad": maximum_rotation_axis,
        "maximum_blade_speed_mps": maximum_blade_speed,
        "first_step_phase_milestone_rate": float(first_milestone_rate),
    }


def _mount_compliance_check(env) -> dict[str, float]:
    robot = env.unwrapped.scene["robot"]
    env.reset()
    body_ids, _ = robot.find_bodies("base_link")
    _assert(len(body_ids) == 1, f"Expected one base_link, found {body_ids}")
    base_id = int(body_ids[0])
    # Pause the recurring random pulse so this response measurement isolates
    # the D6 spring.  Natural randomized stability is tested separately above.
    interval_names = env.unwrapped.event_manager.active_terms["interval"]
    wobble_index = interval_names.index("base_wobble_excitation")
    env.unwrapped.event_manager._interval_term_time_left[wobble_index].fill_(1.0e6)  # noqa: SLF001
    wobble_cfg = env.unwrapped.event_manager.get_term_cfg("base_wobble_excitation")
    wobble_cfg.func.reset()

    zero_actions = torch.zeros(
        (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device
    )
    robot.set_external_force_and_torque(
        torch.zeros((env.unwrapped.num_envs, 1, 3), device=env.unwrapped.device),
        torch.zeros((env.unwrapped.num_envs, 1, 3), device=env.unwrapped.device),
        body_ids=[base_id],
        is_global=True,
    )
    _step_and_validate(env, zero_actions)
    start = robot.data.root_pos_w.clone()
    initial_position, initial_rotation_vector = _mount_relative_pose(env)
    initial_displacement = torch.linalg.vector_norm(initial_position, dim=-1)
    initial_rotation = torch.linalg.vector_norm(initial_rotation_vector, dim=-1)
    anchor_offset = initial_displacement
    _assert(
        float(initial_displacement.max()) <= 0.018,
        "Robot begins outside its mount translation envelope: "
        f"relative_position={initial_position.tolist()}, displacement={initial_displacement.tolist()}",
    )
    _assert(
        float(initial_rotation.max()) <= 0.0436332313,
        "Robot begins outside its mount rotation envelope: "
        f"relative_axis_angle={initial_rotation_vector.tolist()}, rotation={initial_rotation.tolist()}",
    )
    _assert(
        float(anchor_offset.max()) <= 1.0e-3,
        f"Robot base and mount anchor frames do not coincide: offset={anchor_offset.tolist()}",
    )

    forces = torch.zeros((env.unwrapped.num_envs, 1, 3), device=env.unwrapped.device)
    torques = torch.zeros_like(forces)
    forces[..., 1] = 30.0
    robot.set_external_force_and_torque(forces, torques, body_ids=[base_id], is_global=True)
    for _ in range(5):
        _, _, terminated, truncated, _ = _step_and_validate(env, zero_actions)
        _assert(not bool((terminated | truncated).any()), "Bounded 30 N mount pulse caused a termination")
    displaced = torch.linalg.vector_norm(robot.data.root_pos_w - start, dim=-1).max()

    robot.set_external_force_and_torque(torch.zeros_like(forces), torch.zeros_like(torques), body_ids=[base_id])
    for _ in range(30):
        _step_and_validate(env, zero_actions)
    restored = torch.linalg.vector_norm(robot.data.root_pos_w - start, dim=-1).max()
    displaced_value = float(displaced)
    restored_value = float(restored)
    _assert(displaced_value > 1.0e-6, f"Compliant mount did not respond to 30 N wrench: {displaced_value}")
    _assert(restored_value < displaced_value, f"Mount did not restore: {displaced_value} -> {restored_value}")
    _assert(displaced_value < 0.04, f"Mount exceeded 40 mm safety envelope: {displaced_value}")
    return {
        "initial_displacement_max_m": float(initial_displacement.max()),
        "initial_rotation_max_rad": float(initial_rotation.max()),
        "anchor_offset_max_m": float(anchor_offset.max()),
        "forced_displacement_m": displaced_value,
        "restored_displacement_m": restored_value,
    }


def _teacher_smoke() -> dict[str, object]:
    task = "Isaac-ZeroG-BladeSwap-Teacher-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.teacher_envs)
    _assert(tuple(cfg.sim.gravity) == (0.0, 0.0, 0.0), f"Gravity is not zero: {cfg.sim.gravity}")
    _assert(cfg.scene.camera is None, "Teacher config unexpectedly allocates a camera")
    env = gym.make(task, cfg=cfg)
    try:
        observations, _ = env.reset(seed=args.seed)
        _assert(set(observations) >= {"policy", "critic"}, f"Teacher groups are {sorted(observations)}")
        _assert_finite(observations, "reset observations")
        _assert(
            bool(torch.equal(env.unwrapped._last_rewarded_phase, env.unwrapped._swap_phase)),
            "Curriculum reset did not initialize milestone bookkeeping to the active phase",
        )
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        _assert(action_dim == 7, f"Expected seven actions, received {action_dim}")
        robot = env.unwrapped.scene["robot"]
        required_joints = {
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "finger_joint",
        }
        _assert(
            required_joints.issubset(robot.joint_names), f"Missing joints: {required_joints - set(robot.joint_names)}"
        )
        _assert("wrist_3_link" in robot.body_names and "base_link" in robot.body_names, "Robot body contract failed")
        sensors = getattr(env.unwrapped.scene, "sensors", {})
        _assert(not any("contact" in name.lower() for name in sensors), f"Unexpected contact sensors: {list(sensors)}")

        blade = env.unwrapped.scene["blade"]
        masses = blade.root_physx_view.get_masses().reshape(-1)
        _assert(bool(((masses >= 5.0) & (masses <= 15.0)).all()), f"Blade masses out of range: {masses}")
        slot = env.unwrapped.scene["blade_slot"]
        materials = slot.root_physx_view.get_material_properties().reshape(-1, 3)
        dynamic_friction = materials[:, 1]
        _assert(
            bool(((dynamic_friction >= 0.20) & (dynamic_friction <= 1.50)).all()),
            f"Dynamic friction out of range: {dynamic_friction}",
        )
        geometry = _geometry_snapshot(env)
        stability = _natural_mount_stability_check(env)

        completed = 0
        terminations = 0
        while completed < args.teacher_steps:
            _, _, terminated, truncated, _ = _step_and_validate(env, _random_actions(env))
            terminations += int((terminated | truncated).sum())
            completed += 1
        mount = _mount_compliance_check(env)
        return {
            "ok": True,
            "task": task,
            "num_envs": args.teacher_envs,
            "steps": completed,
            "action_dim": action_dim,
            "observation_shapes": _observation_shapes(observations),
            "blade_mass_kg": [float(value) for value in masses],
            "dynamic_friction": [float(value) for value in dynamic_friction],
            "mount": mount,
            "natural_mount_stability": stability,
            "geometry": geometry,
            "episode_terminations": terminations,
            "gpu_memory_mib": _gpu_memory_mib(),
        }
    finally:
        env.close()


def _vision_smoke() -> dict[str, object]:
    task = "Isaac-ZeroG-BladeSwap-Vision-v0"
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=8)
    _assert(tuple(cfg.sim.gravity) == (0.0, 0.0, 0.0), f"Gravity is not zero: {cfg.sim.gravity}")
    _assert(cfg.scene.camera is not None, "Vision config has no tiled camera")
    _assert(cfg.scene.camera.width == 64 and cfg.scene.camera.height == 64, "Camera is not 64x64")
    _assert(cfg.sim.render_interval == 8, f"Expected render_interval=8, got {cfg.sim.render_interval}")
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    try:
        observations, _ = env.reset(seed=args.seed)
        required = {"proprio", "rgb", "critic", "blade_pose"}
        _assert(required.issubset(observations), f"Missing vision groups: {required - set(observations)}")
        rgb = observations["rgb"]
        _assert(tuple(rgb.shape) == (8, 64, 64, 3), f"Unexpected RGB shape: {tuple(rgb.shape)}")
        _assert_finite(observations, "vision observations")
        _assert(float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0, "RGB is not normalized to [0,1]")

        # The camera is unchanged between these calls, while radiation noise is
        # independently sampled inside the observation term.
        repeated_a = env.unwrapped.observation_manager.compute()["rgb"]
        repeated_b = env.unwrapped.observation_manager.compute()["rgb"]
        noise_delta = float((repeated_a - repeated_b).std())
        _assert(noise_delta > 1.0e-4, f"No measurable Gaussian camera noise: std={noise_delta}")
        dark_fraction = float((rgb < 0.02).all(dim=-1).float().mean())
        _assert(dark_fraction > 0.01, f"Expected black orbital background, dark fraction={dark_fraction}")
        per_env_mean = rgb.mean(dim=(1, 2, 3))
        environment_variation = float(per_env_mean.std())

        terminations = 0
        for _ in range(args.vision_steps):
            observations, _, terminated, truncated, _ = _step_and_validate(env, _random_actions(env))
            terminations += int((terminated | truncated).sum())
        return {
            "ok": True,
            "task": task,
            "num_envs": 8,
            "steps": args.vision_steps,
            "observation_shapes": _observation_shapes(observations),
            "rgb_min": float(rgb.min()),
            "rgb_max": float(rgb.max()),
            "rgb_std": float(rgb.std()),
            "radiation_noise_delta_std": noise_delta,
            "dark_background_fraction": dark_fraction,
            "per_environment_mean_std": environment_variation,
            "episode_terminations": terminations,
            "gpu_memory_mib": _gpu_memory_mib(),
        }
    finally:
        env.close()


def main() -> int:
    report: dict[str, object] = {
        "profile": args.profile,
        "seed": args.seed,
        "ok": False,
        "results": {},
    }
    try:
        if args.profile in ("teacher", "all"):
            report["results"]["teacher"] = _teacher_smoke()
        if args.profile in ("vision", "all"):
            report["results"]["vision"] = _vision_smoke()
        report["ok"] = True
        return_code = 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 2
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    return return_code


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
