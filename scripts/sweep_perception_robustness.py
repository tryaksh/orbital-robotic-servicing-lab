"""What the pose head costs when the world stops being ideal.

A perception result measured through a perfectly calibrated camera, on the same
lighting distribution it trained on, is the easiest version of its own problem.
Nobody can build against it, because the first question an integrator asks is
"to what tolerance do I have to mount and calibrate this?" and the second is
"what happens on a Tuesday when the sun is somewhere it never was in training?"

This sweeps both, on the *trained* head, with no retraining anywhere:

* **camera calibration** — the mount is displaced by up to a stated position and
  rotation error, drawn per episode and never revealed. This is the number an
  integrator needs;
* **lighting outside the trained envelope** — the orbital sun is pushed brighter
  and dimmer than any frame the head ever saw, which is the honest version of
  "does it generalise" rather than a held-out split of the same distribution.

Reports millimetres of position error at each point, against the 20 mm capture
tolerance the interface specification sets. The workflow success rate under the
chosen operating point is certified separately; this is the cheap curve that
says where that point should be.
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

#: From docs/service_interface_spec.md: the grip error a capture tolerates.
CAPTURE_TOLERANCE_MM = 20.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-GrappleVision-Collect-v0")
    parser.add_argument("--head", type=Path, default=Path("checkpoints/module_pose_head.pth"))
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=12, help="Resets per sweep point.")
    parser.add_argument("--seed", type=int, default=7090)
    parser.add_argument(
        "--calibration_mm",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
        help="Camera mount position error, in millimetres, swept one point at a time.",
    )
    parser.add_argument(
        "--calibration_mrad",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 5.0, 10.0, 20.0, 40.0],
        help="Camera mount rotation error, in milliradians.",
    )
    parser.add_argument(
        "--sun_intensity",
        type=float,
        nargs="+",
        default=[1_200.0, 2_500.0, 5_250.0, 8_000.0, 14_000.0],
        help="Orbital sun intensity. Training sampled 2500 to 8000, so the ends are outside it.",
    )
    parser.add_argument("--report", type=Path, default=Path("evidence/perception_robustness.json"))
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
from zero_g_blade_swap.tasks.blade_swap.mdp.perception import jitter_camera_pose, module_pose_label


def _measure(task, head, zero, rounds: int, stages: torch.Tensor) -> dict[str, float]:
    """Position error over ``rounds`` fresh resets, in millimetres."""

    errors = []
    for _ in range(rounds):
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
    error = torch.cat(errors)
    return {
        "samples": int(error.numel()),
        "position_error_mm_mean": float(error.mean()),
        "position_error_mm_p95": float(error.quantile(0.95)),
        "position_error_mm_max": float(error.max()),
        "inside_capture_tolerance_fraction": float((error < CAPTURE_TOLERANCE_MM).float().mean()),
    }


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
        stages = torch.tensor([0, 1, 2], device=task.device)
        zero = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)
        head = load_pose_head(args.head, task.device)

        # The jitter term is installed by hand rather than declared on the task,
        # so the certified evaluations keep the exact event set they were run
        # under and only this sweep perturbs the mount.
        jitter = {"position_noise_m": 0.0, "rotation_noise_rad": 0.0}
        original_reset = task._reset_idx

        def reset_with_jitter(env_ids):
            original_reset(env_ids)
            jitter_camera_pose(task, env_ids, **jitter)

        task._reset_idx = reset_with_jitter

        calibration = []
        for millimetres in args.calibration_mm:
            jitter.update(position_noise_m=millimetres / 1_000.0, rotation_noise_rad=0.0)
            entry = {"mount_position_error_mm": millimetres, **_measure(task, head, zero, args.rounds, stages)}
            calibration.append(entry)
            print(f"[INFO] mount {millimetres:5.1f} mm -> {entry['position_error_mm_mean']:6.2f} mm", flush=True)

        rotation = []
        for milliradians in args.calibration_mrad:
            jitter.update(position_noise_m=0.0, rotation_noise_rad=milliradians / 1_000.0)
            entry = {"mount_rotation_error_mrad": milliradians, **_measure(task, head, zero, args.rounds, stages)}
            rotation.append(entry)
            print(f"[INFO] mount {milliradians:5.1f} mrad -> {entry['position_error_mm_mean']:6.2f} mm", flush=True)

        jitter.update(position_noise_m=0.0, rotation_noise_rad=0.0)
        lighting = []
        sun = task.event_manager.get_term_cfg("orbital_sun")
        trained = tuple(sun.params["intensity_range"])
        for intensity in args.sun_intensity:
            sun.params["intensity_range"] = (intensity, intensity)
            entry = {
                "sun_intensity": intensity,
                "outside_trained_range": not (trained[0] <= intensity <= trained[1]),
                **_measure(task, head, zero, args.rounds, stages),
            }
            lighting.append(entry)
            print(
                f"[INFO] sun {intensity:8.0f} -> {entry['position_error_mm_mean']:6.2f} mm"
                f"{'  (outside trained range)' if entry['outside_trained_range'] else ''}",
                flush=True,
            )
        sun.params["intensity_range"] = trained

        nominal = calibration[0]["position_error_mm_mean"]
        usable = [row for row in calibration if row["position_error_mm_p95"] < CAPTURE_TOLERANCE_MM]
        report = {
            "status": "measured",
            "title": "Pose head under camera miscalibration and untrained lighting",
            "evidence_type": "simulation_capability_envelope",
            "protocol": {
                "task": args.task,
                "head": str(args.head),
                "environments": args.num_envs,
                "resets_per_point": args.rounds,
                "samples_per_point": args.num_envs * args.rounds,
                "seed": args.seed,
                "retraining": "none; the head is the one certified in evidence/module_pose_head.json",
                "trained_sun_intensity_range": list(trained),
            },
            "camera_mount_position_error": calibration,
            "camera_mount_rotation_error": rotation,
            "sun_intensity": lighting,
            "summary": {
                "nominal_position_error_mm": nominal,
                "largest_mount_error_mm_inside_capture_tolerance": (
                    max(row["mount_position_error_mm"] for row in usable) if usable else None
                ),
                "capture_tolerance_mm": CAPTURE_TOLERANCE_MM,
            },
            "gate": {
                "applies": False,
                "note": "A capability envelope, not a promotion gate: it maps where the estimator stops working.",
            },
            "scope_and_limitations": [
                "Simulation only. A rendered camera under a rendered sun, not a calibrated instrument.",
                "The mount error is drawn per episode and constant within it, which models a calibration "
                "offset rather than vibration.",
                "Lighting is swept on intensity alone. Angle, colour temperature and albedo stay randomized "
                "over their trained ranges.",
            ],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] wrote {args.report}")
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
