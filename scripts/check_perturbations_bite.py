"""Prove the perturbations reach the image before trusting any robustness curve.

This project has published one inert probe already: a yaw gate that reported an
identical number with and without the feature it was built to test, because the
rails were holding the module still. The rule written after it is that a probe
must be shown to move the thing it measures.

``scripts/sweep_perception_robustness.py`` reported a pose error flat to within
a millimetre across a 20 mm camera displacement, a 40 mrad rotation, and a
ten-fold change in sun intensity. Either the head is extraordinarily robust or
none of those perturbations reached the renderer. This script decides which, by
rendering the same scene twice and differencing the pixels.

A perturbation that does not change the image cannot be evidence about anything.
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

#: Mean absolute difference, in 8-bit levels, below which two renders are the
#: same picture. Camera noise alone moves a frame by a few levels.
INERT_THRESHOLD_LEVELS = 2.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-GrappleVision-Collect-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7091)
    parser.add_argument("--report", type=Path, default=Path("evidence/perturbations_bite.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.tasks.blade_swap.mdp.perception import jitter_camera_pose


def main() -> dict[str, object]:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.pose_head_oracle_blend = 1.0
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        if getattr(task, "_insertion_curriculum_stage", None) is None:
            task._insertion_curriculum_stage = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
        zero = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        camera = task.scene.sensors["camera"]

        def render() -> torch.Tensor:
            for _ in range(2):
                observations, _, _, _, _ = task.step(zero)
            return observations["rgb"].clone() * 255.0

        def difference(a: torch.Tensor, b: torch.Tensor) -> float:
            return float((a - b).abs().mean())

        checks: list[dict[str, object]] = []

        # 1. The floor: two renders of the same scene, nothing changed. Whatever
        #    this is, it is the noise the other numbers have to beat.
        torch.manual_seed(0)
        task.reset()
        baseline = render()
        pose_before = camera.data.pos_w.clone()
        repeat = render()
        checks.append(
            {
                "perturbation": "none (camera noise floor)",
                "mean_abs_difference_levels": difference(baseline, repeat),
                "expected": "small",
            }
        )

        # 2. Does set_world_poses actually move the camera, and does the picture
        #    follow it? Both halves are checked, because a pose that changes in
        #    the buffer and not in the render is the failure being hunted.
        jitter_camera_pose(task, None, position_noise_m=0.050, rotation_noise_rad=0.0)
        moved = float(torch.linalg.vector_norm(camera.data.pos_w - pose_before, dim=-1).mean())
        shifted = render()
        checks.append(
            {
                "perturbation": "camera displaced 50 mm",
                "camera_actually_moved_mm": 1_000.0 * moved,
                "mean_abs_difference_levels": difference(baseline, shifted),
                "expected": "large",
            }
        )

        # 3. Sun intensity, driven the way the sweep drives it.
        task.reset()
        sun = task.event_manager.get_term_cfg("orbital_sun")
        trained = tuple(sun.params["intensity_range"])
        sun.params["intensity_range"] = (1_000.0, 1_000.0)
        task.reset()
        dim = render()
        sun.params["intensity_range"] = (16_000.0, 16_000.0)
        task.reset()
        bright = render()
        sun.params["intensity_range"] = trained
        checks.append(
            {
                "perturbation": "sun 1,000 against 16,000",
                "mean_abs_difference_levels": difference(dim, bright),
                "dim_mean_level": float(dim.mean()),
                "bright_mean_level": float(bright.mean()),
                "expected": "large",
            }
        )

        for check in checks:
            check["bites"] = bool(check["mean_abs_difference_levels"] > INERT_THRESHOLD_LEVELS)

        report = {
            "status": "measured",
            "title": "Do the perception perturbations reach the image at all",
            "evidence_type": "simulation_probe_validation",
            "threshold_levels": INERT_THRESHOLD_LEVELS,
            "checks": checks,
            "verdict": {
                "camera_displacement_is_live": checks[1]["bites"],
                "sun_intensity_is_live": checks[2]["bites"],
                "note": (
                    "A robustness curve measured with an inert perturbation is not a weak result, it is a "
                    "false one. This project published an inert yaw probe once and the rule written "
                    "afterwards is that a probe must be shown to move what it measures."
                ),
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for check in checks:
            print(
                f"[{'LIVE' if check['bites'] else 'INERT'}] {check['perturbation']}: "
                f"{check['mean_abs_difference_levels']:.2f} levels",
                flush=True,
            )
        return report
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
