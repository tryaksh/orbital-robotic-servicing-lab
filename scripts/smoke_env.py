"""Run deterministic state and vision smoke tests against the installed tasks.

This is the cheap pre-flight. Isaac Lab configurations cannot be constructed
without the Kit runtime, so a config bug only surfaces on a full simulator
launch; catching one here costs a minute instead of an hour of GPU.

Repointed on 2026-08-10 from the deleted eight-phase swap task. Two checks went
with it and are recorded as lost rather than quietly dropped:

- The D6 mount compliance response, which needed the wobble excitation the
  insertion profiles only enable at robustness level 4. Level 4 is blocked
  behind level 3 (see docs/status.md), so nothing can exercise it today.
- The eight-phase reward-leak check, which asserted that a curriculum reset did
  not award free phase milestones. There are no phases any more.

What is still checked is what still exists: zero gravity, the action and
observation contract, contact reporting, blade mass, idle stability, and, for
the vision profile, that the camera and both Replicator randomizers actually
produce varied images.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

STATE_TASK = "Isaac-ZeroG-Blade-Insertion-ForceFeedback-v0"
VISION_TASK = "Isaac-ZeroG-Blade-Insertion-Vision-v0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("state", "vision", "all"), default="all")
    parser.add_argument("--state_steps", type=int, default=100)
    parser.add_argument("--state_envs", type=int, default=1)
    parser.add_argument("--idle_steps", type=int, default=300)
    parser.add_argument("--vision_steps", type=int, default=32)
    parser.add_argument(
        "--robustness_level",
        type=int,
        choices=range(5),
        default=2,
        help="Insertion profile to smoke; 2 is the level the promoted policies were certified at.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke_report.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.state_steps < 100:
    parser.error("--state_steps must be at least 100")
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

from isaaclab.utils.math import axis_angle_from_quat, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.mdp.insertion import insertion_error_metrics


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


def _geometry_snapshot(env) -> dict[str, object]:
    axial, lateral, orientation = insertion_error_metrics(env.unwrapped)
    blade = env.unwrapped.scene["spare_blade"]
    local_position = blade.data.root_pos_w - env.unwrapped.scene.env_origins
    return {
        "robot_joint_positions_first_env": [float(value) for value in env.unwrapped.scene["robot"].data.joint_pos[0]],
        "blade_position_first_env_m": [float(value) for value in local_position[0]],
        "axial_error_first_env_m": float(axial[0]),
        "lateral_error_first_env_m": float(lateral[0]),
        "orientation_error_first_env_rad": float(orientation[0]),
    }


def _idle_stability_check(env) -> dict[str, object]:
    """Hold every action at zero and require the scene to stay put.

    A task that destabilises without any commanded motion has a physics problem,
    not a learning problem, and no amount of PPO will fix it.
    """

    zero_actions = torch.zeros(
        (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device
    )
    watched = [name for name in ("mount_unstable", "non_finite", "insertion_failed") if name in
               env.unwrapped.termination_manager.active_terms]
    termination_counts = {name: 0 for name in watched}
    maximum_translation_axis = 0.0
    maximum_rotation_axis = 0.0
    maximum_blade_speed = 0.0
    for _ in range(args.idle_steps):
        _step_and_validate(env, zero_actions)
        for name in watched:
            termination_counts[name] += int(env.unwrapped.termination_manager.get_term(name).sum())
        relative_position, relative_rotation = _mount_relative_pose(env)
        maximum_translation_axis = max(maximum_translation_axis, float(relative_position.abs().max()))
        maximum_rotation_axis = max(maximum_rotation_axis, float(relative_rotation.abs().max()))
        blade = env.unwrapped.scene["spare_blade"]
        maximum_blade_speed = max(
            maximum_blade_speed, float(torch.linalg.vector_norm(blade.data.root_lin_vel_w, dim=-1).max())
        )

    _assert(termination_counts.get("mount_unstable", 0) == 0, f"Idle scene tripped the mount: {termination_counts}")
    _assert(termination_counts.get("non_finite", 0) == 0, f"Idle scene became non-finite: {termination_counts}")
    _assert(
        termination_counts.get("insertion_failed", 0) == 0,
        f"Idle scene failed its own insertion predicate: {termination_counts}",
    )
    return {
        "steps": args.idle_steps,
        "terminations": termination_counts,
        "maximum_translation_axis_m": maximum_translation_axis,
        "maximum_rotation_axis_rad": maximum_rotation_axis,
        "maximum_blade_speed_mps": maximum_blade_speed,
    }


def _state_smoke() -> dict[str, object]:
    cfg = parse_env_cfg(STATE_TASK, device=args.device or "cuda:0", num_envs=args.state_envs)
    cfg.configure_robustness(args.robustness_level)
    _assert(tuple(cfg.sim.gravity) == (0.0, 0.0, 0.0), f"Gravity is not zero: {cfg.sim.gravity}")
    _assert(getattr(cfg.scene, "camera", None) is None, "State config unexpectedly allocates a camera")
    env = gym.make(STATE_TASK, cfg=cfg)
    try:
        observations, _ = env.reset(seed=args.seed)
        _assert(set(observations) == {"policy"}, f"State groups are {sorted(observations)}")
        _assert_finite(observations, "reset observations")
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        _assert(action_dim == 6, f"Expected six Cartesian actions, received {action_dim}")

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

        # The force lineage exists to observe contact. A sensor that resolves no
        # bodies reports a constant zero, which looks exactly like a gentle
        # insertion, so assert it is there rather than trusting the config.
        sensors = env.unwrapped.scene.sensors
        _assert("blade_contact" in sensors, f"Force task has no blade contact sensor: {sorted(sensors)}")
        _assert(
            sensors["blade_contact"].data.net_forces_w is not None,
            "The blade contact sensor resolved no bodies; check clone_in_fabric and activate_contact_sensors",
        )

        masses = env.unwrapped.scene["spare_blade"].root_physx_view.get_masses().reshape(-1)
        if args.robustness_level >= 2:
            _assert(bool(((masses >= 5.0) & (masses <= 15.0)).all()), f"Blade masses out of range: {masses}")

        geometry = _geometry_snapshot(env)
        stability = _idle_stability_check(env)

        completed = 0
        terminations = 0
        while completed < args.state_steps:
            _, _, terminated, truncated, _ = _step_and_validate(env, _random_actions(env))
            terminations += int((terminated | truncated).sum())
            completed += 1
        return {
            "ok": True,
            "task": STATE_TASK,
            "robustness_level": args.robustness_level,
            "num_envs": args.state_envs,
            "steps": completed,
            "action_dim": action_dim,
            "observation_shapes": _observation_shapes(observations),
            "blade_mass_kg": [float(value) for value in masses],
            "idle_stability": stability,
            "geometry": geometry,
            "episode_terminations": terminations,
            "gpu_memory_mib": _gpu_memory_mib(),
        }
    finally:
        env.close()


def _vision_smoke() -> dict[str, object]:
    cfg = parse_env_cfg(VISION_TASK, device=args.device or "cuda:0", num_envs=8)
    _assert(tuple(cfg.sim.gravity) == (0.0, 0.0, 0.0), f"Gravity is not zero: {cfg.sim.gravity}")
    _assert(cfg.scene.camera is not None, "Vision config has no tiled camera")
    _assert(cfg.scene.camera.width == 256 and cfg.scene.camera.height == 256, "Camera is not 256x256")
    _assert(cfg.sim.render_interval == 8, f"Expected render_interval=8, got {cfg.sim.render_interval}")
    # Both Replicator randomizers address one prim per environment and refuse to
    # construct against replicated physics.
    _assert(not cfg.scene.replicate_physics, "Vision config replicates physics; the randomizers cannot run")
    env = gym.make(VISION_TASK, cfg=cfg, render_mode="rgb_array")
    try:
        observations, _ = env.reset(seed=args.seed)
        required = {"proprio", "rgb", "critic", "blade_pose"}
        _assert(required.issubset(observations), f"Missing vision groups: {required - set(observations)}")
        rgb = observations["rgb"]
        _assert(tuple(rgb.shape) == (8, 256, 256, 3), f"Unexpected RGB shape: {tuple(rgb.shape)}")
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
            "task": VISION_TASK,
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
        if args.profile in ("state", "all"):
            report["results"]["state"] = _state_smoke()
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
