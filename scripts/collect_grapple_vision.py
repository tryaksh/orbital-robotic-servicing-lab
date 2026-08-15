"""Record what the servicing camera sees, and where the module actually was.

Supervision for ``mdp.ModulePoseHead``. Each row is a 64x64 RGB frame and the
module's pose in its own environment's frame at the moment that frame was
rendered, under randomized orbital lighting, randomized rack albedo, camera
radiation noise, and an unknown per-episode module displacement.

**Why the arm holds still.** The quantity being learned is where a rigid body is
in an image. The module is held by its rails until something grasps it, so its
pose is constant within an episode and every frame of a rollout would be a near
duplicate. Resetting often instead buys independent samples at the same cost,
which is what a pose regressor actually needs. The policies come back in at
evaluation, where they belong: that is the part that has to be on-distribution,
and it is measured rather than assumed.

Written as one contiguous array of uint8 frames rather than HDF5, because 64x64
RGB at these counts fits comfortably in memory and a single ``.npz`` needs no
optional dependency.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-GrappleVision-Collect-v0")
    parser.add_argument("--output", type=Path, default=Path("datasets/grapple_vision.npz"))
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=2,
        help="Steps after each reset before the frame is kept, so the render matches the new pose.",
    )
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--curriculum_stages", type=int, nargs="+", default=[0, 1, 2])
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
        env_cfg.seed = args.seed
        # The oracle path, so the term does not demand a head that does not exist
        # yet. Nothing here reads the grip error anyway; the arm holds still.
        env_cfg.pose_head_oracle_blend = 1.0
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        stages = torch.tensor(args.curriculum_stages, device=task.device)
        # The workflow profile carries no curriculum, so nothing has created the
        # stage buffer the reset events read. Create it rather than depending on
        # a term this task deliberately does not have.
        if getattr(task, "_insertion_curriculum_stage", None) is None:
            task._insertion_curriculum_stage = torch.zeros(
                task.num_envs, dtype=torch.long, device=task.device
            )

        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        zero = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        collected = 0
        rounds = 0
        while collected < args.samples:
            # Spread the three reset distances across environments so one round
            # samples all of them rather than alternating whole rounds.
            task._insertion_curriculum_stage[:] = stages[
                torch.randint(0, len(stages), (task.num_envs,), device=task.device)
            ]
            task.reset()
            for _ in range(max(1, args.settle_steps)):
                observations, _, _, _, _ = task.step(zero)
            frame = observations["rgb"]
            label = observations["pose_label"]
            images.append((frame.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy())
            labels.append(label.to(torch.float32).cpu().numpy())
            collected += int(frame.shape[0])
            rounds += 1
            if rounds % 25 == 0:
                print(f"[INFO] {collected} / {args.samples} frames", flush=True)

        image_array = np.concatenate(images, axis=0)[: args.samples]
        label_array = np.concatenate(labels, axis=0)[: args.samples]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, images=image_array, labels=label_array)
        spread = label_array.max(axis=0) - label_array.min(axis=0)
        print(f"[INFO] wrote {args.output}: images {image_array.shape}, labels {label_array.shape}")
        print(f"[INFO] label spread per channel (m, m, m, rad, rad, rad): {np.round(spread, 5).tolist()}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
