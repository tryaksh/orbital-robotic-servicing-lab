"""Train the zero-g blade-swap task with RL-Games PPO."""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-BladeSwap-Teacher-v0")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume an RL-Games .pth checkpoint.")
    parser.add_argument("--bc_checkpoint", type=Path, default=None, help="Initialize the vision actor from BC.")
    parser.add_argument("--smoke", action="store_true", help="Run two PPO epochs with small batches.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=300)
    parser.add_argument("--video_interval", type=int, default=10_000)
    parser.add_argument("--run_name", default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_parser()
args = parser.parse_args()
if "Vision" in args.task or args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/Omniverse-dependent imports must happen after AppLauncher constructed the app.
import math
from datetime import datetime

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.agents import register_rl_games_networks


def _fit_minibatch(agent_cfg: dict, num_envs: int) -> None:
    config = agent_cfg["params"]["config"]
    batch = num_envs * int(config["horizon_length"])
    requested = min(int(config["minibatch_size"]), batch)
    divisors = [value for value in range(requested, 0, -1) if batch % value == 0]
    config["minibatch_size"] = divisors[0]
    central = config.get("central_value_config")
    if central is not None:
        central["minibatch_size"] = config["minibatch_size"]


def main() -> None:
    env = None
    try:
        rl_device = args.device or "cuda:0"
        if rl_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"GPU training requested on {rl_device}, but PyTorch cannot access CUDA")
        device_description = (
            torch.cuda.get_device_name(torch.device(rl_device)) if rl_device.startswith("cuda") else "CPU"
        )
        print(f"[INFO] Simulation and PPO device: {rl_device} ({device_description})")
        env_cfg = parse_env_cfg(args.task, device=rl_device, num_envs=args.num_envs)
        agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["seed"] = args.seed
        agent_cfg["params"]["config"]["device"] = rl_device
        agent_cfg["params"]["config"]["device_name"] = rl_device
        if args.max_iterations is not None:
            agent_cfg["params"]["config"]["max_epochs"] = args.max_iterations
        if args.smoke:
            agent_cfg["params"]["config"]["max_epochs"] = 2
            agent_cfg["params"]["config"]["save_best_after"] = 0
            agent_cfg["params"]["config"]["save_frequency"] = 1
        if args.bc_checkpoint is not None:
            if "Vision" not in args.task:
                raise ValueError("--bc_checkpoint is only valid for the Vision task")
            if not args.bc_checkpoint.is_file():
                raise FileNotFoundError(args.bc_checkpoint)
            agent_cfg["params"]["network"]["bc_checkpoint"] = str(args.bc_checkpoint.resolve())

        env_cfg.seed = args.seed
        num_envs = int(env_cfg.scene.num_envs)
        _fit_minibatch(agent_cfg, num_envs)
        name = agent_cfg["params"]["config"]["name"]
        run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root = Path("logs") / "rl_games" / name
        run_dir = (log_root / run_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        agent_cfg["params"]["config"]["train_dir"] = str(log_root.resolve())
        agent_cfg["params"]["config"]["full_experiment_name"] = run_name
        env_cfg.log_dir = str(run_dir)
        dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)

        render_mode = "rgb_array" if args.video else None
        env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(run_dir / "videos" / "train"),
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            )

        rl_device = agent_cfg["params"]["config"]["device"]
        env_options = agent_cfg["params"]["env"]
        env = RlGamesVecEnvWrapper(
            env,
            rl_device,
            env_options.get("clip_observations", math.inf),
            env_options.get("clip_actions", math.inf),
            env_options.get("obs_groups"),
            env_options.get("concate_obs_groups", True),
        )
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: env})
        agent_cfg["params"]["config"]["num_actors"] = num_envs

        register_rl_games_networks()
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)
        runner.reset()
        run_args = {"train": True, "play": False, "sigma": None}
        if args.checkpoint is not None:
            if not args.checkpoint.is_file():
                raise FileNotFoundError(args.checkpoint)
            run_args["checkpoint"] = str(args.checkpoint.resolve())
        print(f"[INFO] Training {args.task} with {num_envs} environments; logs: {run_dir}")
        runner.run(run_args)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
