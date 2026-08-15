"""How well does the camera have to be mounted for the pose head to work?

The first question an integrator asks about a vision-driven system is what
calibration tolerance it demands, because that number sets the mount, the
fixture, and the recalibration interval. A pose head measured only through a
perfect camera has not answered it.

The offset is applied to the sensor's **configured mount**, before the
environment is built, and is therefore constant for a run. That is deliberate on
two counts: it is where a perturbation demonstrably reaches the render — an
earlier attempt to move the camera per episode through ``set_world_poses`` moved
it by exactly 0.0 mm and was deleted — and a mis-mounted camera is mis-mounted
all day, so a per-run offset models calibration error more faithfully than a
per-episode draw would.

One magnitude per invocation, along a fixed direction, so each point is a clean
independent run. ``scripts/run_camera_calibration_sweep.sh`` walks the curve and
pools the points.
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

#: From docs/service_interface_spec.md: the grip error a capture tolerates.
CAPTURE_TOLERANCE_MM = 20.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-GrappleVision-Collect-v0")
    parser.add_argument("--head", type=Path, default=Path("checkpoints/module_pose_head.pth"))
    parser.add_argument("--offset_mm", type=float, default=0.0, help="Mount position error along a fixed diagonal.")
    parser.add_argument("--tilt_mrad", type=float, default=0.0, help="Mount rotation error about a fixed axis.")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7090)
    parser.add_argument("--report", type=Path, required=True)
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
from zero_g_blade_swap.pose_head import load_pose_head
from zero_g_blade_swap.tasks.blade_swap.mdp.perception import module_pose_label


def main() -> dict[str, object]:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.pose_head_oracle_blend = 1.0

        # A fixed unit diagonal, so a magnitude means the same displacement every
        # time and the curve is a function of one number.
        direction = (1.0 / math.sqrt(3.0),) * 3
        offset = args.offset_mm / 1_000.0
        nominal = tuple(env_cfg.scene.camera.offset.pos)
        env_cfg.scene.camera.offset.pos = tuple(p + offset * d for p, d in zip(nominal, direction, strict=True))
        if args.tilt_mrad != 0.0:
            # Small rotation about the camera's own vertical, applied to the
            # configured quaternion. Small-angle, so a half-angle approximation
            # is exact enough at the milliradian scale being swept.
            half = 0.5 * args.tilt_mrad / 1_000.0
            w, x, y, z = env_cfg.scene.camera.offset.rot
            cw, cz = math.cos(half), math.sin(half)
            env_cfg.scene.camera.offset.rot = (
                cw * w - cz * z,
                cw * x - cz * y,
                cw * y + cz * x,
                cw * z + cz * w,
            )

        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        if getattr(task, "_insertion_curriculum_stage", None) is None:
            task._insertion_curriculum_stage = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
        stages = torch.tensor([0, 1, 2], device=task.device)
        zero = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        head = load_pose_head(args.head, task.device)

        errors = []
        frames = []
        for _ in range(args.rounds):
            task._insertion_curriculum_stage[:] = stages[
                torch.randint(0, len(stages), (task.num_envs,), device=task.device)
            ]
            task.reset()
            for _ in range(2):
                observations, _, _, _, _ = task.step(zero)
            with torch.inference_mode():
                predicted = head(observations["rgb"])
            truth = module_pose_label(task)
            errors.append(1_000.0 * torch.linalg.vector_norm(predicted[:, :3] - truth[:, :3], dim=-1))
            frames.append(observations["rgb"].mean().item() * 255.0)
        error = torch.cat(errors)

        report = {
            "status": "measured",
            "title": "Pose head against camera mount calibration error",
            "evidence_type": "simulation_capability_envelope",
            "protocol": {
                "task": args.task,
                "head": str(args.head),
                "mount_position_error_mm": args.offset_mm,
                "mount_tilt_error_mrad": args.tilt_mrad,
                "applied": "to the sensor's configured mount, before the environment is built",
                "nominal_camera_position_m": list(nominal),
                "perturbed_camera_position_m": list(env_cfg.scene.camera.offset.pos),
                "environments": args.num_envs,
                "resets": args.rounds,
                "samples": int(error.numel()),
                "seed": args.seed,
                "retraining": "none",
            },
            "position_error_mm": {
                "mean": float(error.mean()),
                "p50": float(error.median()),
                "p95": float(error.quantile(0.95)),
                "max": float(error.max()),
            },
            "inside_capture_tolerance_fraction": float((error < CAPTURE_TOLERANCE_MM).float().mean()),
            "capture_tolerance_mm": CAPTURE_TOLERANCE_MM,
            "mean_frame_level": sum(frames) / len(frames),
            "gate": {"applies": False, "note": "capability envelope, not a promotion gate"},
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"[INFO] mount {args.offset_mm:5.1f} mm / {args.tilt_mrad:5.1f} mrad -> "
            f"{report['position_error_mm']['mean']:6.2f} mm mean, "
            f"{report['position_error_mm']['p95']:6.2f} mm p95, "
            f"{100.0 * report['inside_capture_tolerance_fraction']:5.1f}% inside tolerance"
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
