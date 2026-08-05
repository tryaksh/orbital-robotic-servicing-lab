"""Evaluate or record an RL-Games blade-swap policy."""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-BladeSwap-Play-v0")
    parser.add_argument("--policy", choices=("teacher", "vision"), default="teacher")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; zero runs until the app closes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=600)
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/play_report.json"),
        help="Machine-readable checkpoint load/play validation result.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if "Vision" in args.task or "Play" in args.task or args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.agents import register_rl_games_networks


def _checkpoint() -> Path:
    if args.checkpoint is not None:
        path = args.checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    experiment = args.experiment or f"zero_g_blade_swap_{args.policy}"
    candidates = list((Path("logs") / "rl_games" / experiment).glob("**/nn/*.pth"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found; pass --checkpoint explicitly")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def main() -> dict[str, object]:
    env = None
    try:
        checkpoint = _checkpoint()
        rl_device = args.device or "cuda:0"
        if rl_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"GPU playback requested on {rl_device}, but PyTorch cannot access CUDA")
        device_description = (
            torch.cuda.get_device_name(torch.device(rl_device)) if rl_device.startswith("cuda") else "CPU"
        )
        print(f"[INFO] Simulation and policy device: {rl_device} ({device_description})")
        env_cfg = parse_env_cfg(args.task, device=rl_device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        agent_task = f"Isaac-ZeroG-BladeSwap-{'Teacher' if args.policy == 'teacher' else 'Vision'}-v0"
        agent_cfg = load_cfg_from_registry(agent_task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["seed"] = args.seed
        agent_cfg["params"]["config"]["device"] = rl_device
        agent_cfg["params"]["config"]["device_name"] = rl_device
        options = agent_cfg["params"]["env"]
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(checkpoint.parent.parent / "videos" / "play"),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RlGamesVecEnvWrapper(
            env,
            agent_cfg["params"]["config"]["device"],
            options.get("clip_observations", math.inf),
            options.get("clip_actions", math.inf),
            options.get("obs_groups"),
            options.get("concate_obs_groups", True),
        )
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: env})
        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = str(checkpoint)
        register_rl_games_networks()
        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(str(checkpoint))
        player.reset()

        obs = env.reset()
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
        player.get_batch_size(obs, 1)
        step = 0
        dt = env.unwrapped.step_dt
        termination_names = ("full_success", "time_out", "blade_lost", "mount_unstable", "non_finite")
        termination_counts = {name: 0 for name in termination_names}
        episodes_completed = 0
        maximum_phase = int(env.unwrapped._swap_phase.max())
        reward_sum = torch.zeros(args.num_envs, device=rl_device)
        while simulation_app.is_running() and (args.steps <= 0 or step < args.steps):
            started = time.perf_counter()
            with torch.inference_mode():
                network_obs = player.obs_to_torch(obs)
                actions = player.get_action(network_obs, is_deterministic=True)
                if not bool(torch.isfinite(actions).all()):
                    raise RuntimeError(f"Policy produced a non-finite action at step {step}")
                obs, rewards, dones, _ = env.step(actions)
                observation_tensor = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
                if not bool(torch.isfinite(observation_tensor).all()):
                    raise RuntimeError(f"Environment produced a non-finite observation at step {step}")
                reward_sum += rewards
                episodes_completed += int(dones.sum())
                maximum_phase = max(maximum_phase, int(env.unwrapped._swap_phase.max()))
                for name in termination_names:
                    termination_counts[name] += int(env.unwrapped.termination_manager.get_term(name).sum())
                if player.is_rnn and player.states is not None:
                    for state in player.states:
                        state[:, dones, :] = 0.0
            step += 1
            if args.video and step >= args.video_length:
                break
            delay = dt - (time.perf_counter() - started)
            if args.real_time and delay > 0:
                time.sleep(delay)
        if args.steps > 0 and step != args.steps:
            raise RuntimeError(f"Simulation stopped after {step} of {args.steps} requested steps")
        final_phases = torch.bincount(env.unwrapped._swap_phase, minlength=8)
        return {
            "status": "passed",
            "task": args.task,
            "policy": args.policy,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper(),
            "device": rl_device,
            "gpu": device_description,
            "num_envs": args.num_envs,
            "steps": step,
            "episodes_completed": episodes_completed,
            "termination_counts": termination_counts,
            "maximum_phase_reached": maximum_phase,
            "final_phase_histogram": {str(index): int(count) for index, count in enumerate(final_phases)},
            "mean_cumulative_reward_per_environment": float(reward_sum.mean()),
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
        report = {
            "status": "failed",
            "task": args.task,
            "policy": args.policy,
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "num_envs": args.num_envs,
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        simulation_app.close()
