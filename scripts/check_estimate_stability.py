"""Is the pose estimate steady, or does it jitter the policy around?

A perception number quoted as a mean error describes an *open-loop* quantity. In
a closed loop the policy servos to the estimate every control step, so an
estimate that is wrong by a constant 3 mm is nearly harmless — the arm goes to a
slightly wrong place and the capture tolerance absorbs it — while an estimate
that is right on average but rattles by 3 mm between consecutive frames is a
disturbance injected at 30 Hz.

The calibration-robust head is more accurate under a mis-mounted camera and
scored 38.19% in the workflow against the brittle head's 80.38%. Mean accuracy
cannot explain that. This measures the other axis.

Protocol: reset, then hold the arm still and render repeatedly. The module is
held by its rails and does not move, so **everything** that varies between
consecutive estimates is the estimator.
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-GrappleVision-Collect-v0")
    parser.add_argument(
        "--heads",
        type=Path,
        nargs="+",
        default=[Path("checkpoints/module_pose_head.pth"), Path("checkpoints/module_pose_head_calib.pth")],
    )
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=24, help="Consecutive frames per episode.")
    parser.add_argument("--seed", type=int, default=7092)
    parser.add_argument("--report", type=Path, default=Path("evidence/estimate_stability.json"))
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
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        if getattr(task, "_insertion_curriculum_stage", None) is None:
            task._insertion_curriculum_stage = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
        zero = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)

        results = []
        for head_path in args.heads:
            head = load_pose_head(head_path, task.device)
            steps, biases = [], []
            for episode in range(args.episodes):
                task._insertion_curriculum_stage[:] = episode % 3
                task.reset()
                estimates = []
                for _ in range(args.steps):
                    observations, _, _, _, _ = task.step(zero)
                    with torch.inference_mode():
                        estimates.append(head(observations["rgb"])[:, :3])
                stack = torch.stack(estimates)  # (steps, envs, 3)
                truth = module_pose_label(task)[:, :3]
                # Step-to-step movement of the estimate, with the module still.
                steps.append(1_000.0 * torch.linalg.vector_norm(stack[1:] - stack[:-1], dim=-1).flatten())
                # How wrong the time-average is: the part filtering cannot fix.
                biases.append(1_000.0 * torch.linalg.vector_norm(stack.mean(dim=0) - truth, dim=-1))
            jitter = torch.cat(steps)
            bias = torch.cat(biases)
            entry = {
                "head": str(head_path),
                "frame_to_frame_jitter_mm_mean": float(jitter.mean()),
                "frame_to_frame_jitter_mm_p95": float(jitter.quantile(0.95)),
                "time_averaged_bias_mm_mean": float(bias.mean()),
                "time_averaged_bias_mm_p95": float(bias.quantile(0.95)),
                "samples": int(jitter.numel()),
            }
            results.append(entry)
            print(
                f"[INFO] {head_path.name}: jitter {entry['frame_to_frame_jitter_mm_mean']:.2f} mm/frame, "
                f"bias {entry['time_averaged_bias_mm_mean']:.2f} mm",
                flush=True,
            )

        report = {
            "status": "measured",
            "title": "Pose estimate stability, with the module held still",
            "evidence_type": "simulation_sensor_characterization",
            "protocol": {
                "task": args.task,
                "environments": args.num_envs,
                "episodes": args.episodes,
                "consecutive_frames_per_episode": args.steps,
                "arm": "held still, so all variation between estimates is the estimator",
                "seed": args.seed,
            },
            "heads": results,
            "interpretation": (
                "Jitter is what a closed loop feels: the policy servos to the estimate every control step, "
                "so frame-to-frame movement is a disturbance at the control rate. Bias is what filtering "
                "cannot remove. A head with low bias and high jitter is a filtering problem; one with low "
                "jitter and high bias is a training problem."
            ),
            "gate": {"applies": False},
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
