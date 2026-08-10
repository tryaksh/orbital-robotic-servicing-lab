"""Collect synchronized RGB/proprio/pose demonstrations from a state teacher.

Repointed on 2026-08-10 from the deleted eight-phase swap task onto the vision
insertion task. The per-sample label changed with it: the old dataset stored a
one-hot task phase, and this one stores the *true blade-to-goal error*, which is
what a P3 pose-regression head has to learn to predict from the image.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-Insertion-Vision-Play-v0")
    parser.add_argument(
        "--teacher_task",
        default="Isaac-ZeroG-Blade-Insertion-ForceFeedback-v0",
        help="Task whose PPO configuration the state teacher checkpoint was trained under.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/teacher_250k.h5"))
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression", type=int, choices=range(0, 10), default=4)
    parser.add_argument("--teacher_group", default="policy")
    parser.add_argument("--proprio_group", default="proprio")
    parser.add_argument("--rgb_group", default="rgb")
    parser.add_argument("--pose_group", default="blade_pose")
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
from datetime import UTC, datetime

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required: C:\\isaac-sim\\python.bat -m pip install h5py") from exc

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.agents import register_rl_games_networks


def _require_groups(observations: dict[str, torch.Tensor]) -> None:
    required = {args.teacher_group, args.proprio_group, args.rgb_group, args.pose_group}
    missing = sorted(required.difference(observations))
    if missing:
        raise KeyError(
            f"The {args.task} observation config is missing {missing}. Demonstration collection requires the "
            f"environment to expose the teacher's state, proprioception, RGB, and the ground-truth blade pose "
            f"concurrently; available groups: {sorted(observations)}"
        )


class Hdf5Appender:
    def __init__(self, path: Path, compression: int, metadata: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        for key, value in metadata.items():
            self.file.attrs[key] = value
        self.compression = compression
        self.datasets: dict[str, h5py.Dataset] = {}
        self.count = 0

    def append(self, batch: dict[str, np.ndarray]) -> None:
        amount = len(next(iter(batch.values())))
        for name, values in batch.items():
            values = np.ascontiguousarray(values)
            if name not in self.datasets:
                # One simulator batch per chunk keeps image compression efficient while
                # bounding temporary writes to roughly 1.5 MiB for 128x64x64 RGB.
                chunk_rows = min(256, amount)
                self.datasets[name] = self.file.create_dataset(
                    name,
                    shape=(0, *values.shape[1:]),
                    maxshape=(None, *values.shape[1:]),
                    chunks=(chunk_rows, *values.shape[1:]),
                    dtype=values.dtype,
                    compression="gzip" if self.compression else None,
                    compression_opts=self.compression if self.compression else None,
                    shuffle=bool(self.compression),
                )
            dataset = self.datasets[name]
            dataset.resize(self.count + amount, axis=0)
            dataset[self.count : self.count + amount] = values
        self.count += amount

    def close(self) -> None:
        self.file.attrs["samples"] = self.count
        self.file.flush()
        self.file.close()


def _raw_observations(env) -> dict[str, torch.Tensor]:
    observations = env.unwrapped.observation_manager.compute()
    if not isinstance(observations, dict):
        raise TypeError("ManagerBasedRLEnv observation manager did not return a group dictionary")
    return observations


def main() -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    raw_env = None
    writer = None
    try:
        rl_device = args.device or "cuda:0"
        env_cfg = parse_env_cfg(args.task, device=rl_device, num_envs=args.num_envs)
        env_cfg.seed = args.seed
        teacher_cfg = load_cfg_from_registry(args.teacher_task, "rl_games_cfg_entry_point")
        teacher_cfg["params"]["seed"] = args.seed
        teacher_cfg["params"]["config"]["device"] = rl_device
        teacher_cfg["params"]["config"]["device_name"] = rl_device
        teacher_cfg["params"]["config"]["num_actors"] = args.num_envs
        teacher_cfg["params"]["load_checkpoint"] = True
        teacher_cfg["params"]["load_path"] = str(checkpoint)

        raw_env = gym.make(args.task, cfg=env_cfg)
        first_raw, _ = raw_env.reset()
        _require_groups(first_raw)
        env_options = teacher_cfg["params"]["env"]
        collection_groups = {"obs": [args.teacher_group], "states": []}
        env = RlGamesVecEnvWrapper(
            raw_env,
            teacher_cfg["params"]["config"]["device"],
            env_options.get("clip_observations", math.inf),
            env_options.get("clip_actions", math.inf),
            collection_groups,
            True,
        )
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **_: env})
        register_rl_games_networks()
        runner = Runner()
        runner.load(teacher_cfg)
        player = runner.create_player()
        player.restore(str(checkpoint))
        player.reset()

        processed = env.reset()
        teacher_obs = processed["obs"]
        player.get_batch_size(teacher_obs, 1)
        writer = Hdf5Appender(
            args.output.resolve(),
            args.compression,
            {
                "format": "zero-g-blade-insertion-demonstrations-v2",
                "created_utc": datetime.now(UTC).isoformat(),
                "task": args.task,
                "teacher_task": args.teacher_task,
                "teacher_checkpoint": str(checkpoint),
                "seed": args.seed,
            },
        )

        while writer.count < args.samples:
            raw_obs = _raw_observations(raw_env)
            with torch.inference_mode():
                actions = player.get_action(player.obs_to_torch(teacher_obs), is_deterministic=True)
            remaining = args.samples - writer.count
            take = min(args.num_envs, remaining)
            rgb = raw_obs[args.rgb_group][:take].detach().cpu().numpy()
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
            elif rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            writer.append(
                {
                    "rgb": rgb,
                    "proprio": raw_obs[args.proprio_group][:take].detach().cpu().numpy().astype(np.float32),
                    "blade_pose": raw_obs[args.pose_group][:take].detach().cpu().numpy().astype(np.float32),
                    "teacher_action": actions[:take].detach().cpu().numpy().astype(np.float32),
                }
            )
            processed, _, dones, _ = env.step(actions)
            teacher_obs = processed["obs"]
            if player.is_rnn and player.states is not None:
                for state in player.states:
                    state[:, dones, :] = 0.0
            if writer.count % max(args.num_envs, 10_000) < args.num_envs:
                print(f"[INFO] Collected {writer.count:,}/{args.samples:,} samples")
        print(f"[INFO] Dataset written to {args.output.resolve()}")
    finally:
        if writer is not None:
            writer.close()
        if raw_env is not None:
            raw_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
