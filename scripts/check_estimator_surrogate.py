"""Does the training-time estimator surrogate actually deliver the deployed error?

A skill trained against a surrogate is only worth the GPU if the surrogate is the
estimator. Nothing downstream would catch a mis-scaled sigma: the run would
train, certify and publish a number, and the error would surface months later as
a policy that does not transfer, with no way to attribute it. So this measures
the surrogate against the certificate it claims to reproduce, and refuses to
report if they disagree.

The protocol is the one `check_estimate_stability.py` uses on the real
estimator, for the same reason: hold the arm still and let the module sit in its
rails, so the true state barely moves and **everything** the observation channels
show is the estimator.

Three things come out of it.

1. **Does the residual match the certificate.** The realized p95 of the position
   and orientation error norms must land on the certified p95 the sigma was
   inverted from. This is the self-validation; a mismatch fails the run.
2. **What the velocity channel's noise floor actually is.** This is the
   quantitative claim the whole retrain rests on. The camera period is twice the
   control period, so a differenced pose estimate is zero on one control step
   and a full jump on the next, and the jump is estimator residual rather than
   motion. Against a seated module's real speed -- of order 0.7 mm/s in the
   strict certification -- a noise floor tens of times larger is a channel
   carrying no signal at all.
3. **What a longer filter would buy.** The raw estimate sequence is recorded, so
   the first-order filter is re-applied over a sweep of time constants on the
   CPU afterwards. That answers "is the cheap fix a filter" without a second
   simulator run, and it answers it with the lag the filter costs stated beside
   the noise it removes.

The module pose is read here for error metrics only, which is the same privilege
boundary every perception report in this repository declares.
"""

# ruff: noqa: E402, I001 -- Isaac modules must be imported after AppLauncher.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jinja2  # Preload before Kit extensions to avoid a partially initialized module.
from isaaclab.app import AppLauncher

assert hasattr(jinja2, "Environment"), "The Jinja2 installation is incomplete."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="Isaac-ZeroG-Blade-GrapplePin-ExtractNoised-v0",
        help="A task whose observation group reads the estimator surrogate.",
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1070)
    parser.add_argument(
        "--seated_speed_mm_s",
        type=float,
        default=0.69,
        help=(
            "The module speed the strict certification's seated episodes record. The velocity "
            "channel's noise floor is reported as a multiple of this, because that ratio is the "
            "claim -- not the noise in isolation."
        ),
    )
    parser.add_argument(
        "--filter_sweep_s",
        type=float,
        nargs="+",
        default=(0.0, 0.10, 0.20, 0.30, 0.50, 1.00),
        help="Velocity filter time constants to re-apply offline. 0.10 is the deployed one.",
    )
    parser.add_argument(
        "--quantile_tolerance",
        type=float,
        default=0.25,
        help=(
            "Relative agreement required between the realized and certified p95 before the report "
            "is written. Finite samples of a p95 are noisy; a surrogate that is right will sit well "
            "inside this, and one that is mis-scaled by a factor will not."
        ),
    )
    parser.add_argument("--report", type=Path, default=Path("artifacts/estimator_surrogate_check.json"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
args.headless = True
# AppLauncher receives the parsed namespace; clearing argv keeps Kit plugins from
# seeing this script's own flags.
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.provenance import git_source_revision
from zero_g_blade_swap.tasks.blade_swap.mdp.estimator_surrogate import shared_surrogate_estimator
from zero_g_blade_swap.tasks.blade_swap.mdp.perception import module_pose_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _percentile(values: np.ndarray, probability: float) -> float:
    return float(np.percentile(values, 100.0 * probability))


def _filtered_speed_statistics(
    raw_velocity: np.ndarray,
    step_dt: float,
    time_constant_s: float,
) -> dict[str, float]:
    """Re-apply the deployed first-order filter offline at one time constant.

    ``raw_velocity`` is (steps, envs, 3): the unfiltered linear finite difference
    of consecutive estimates, exactly what the deployed estimator computes before
    it filters.
    """

    alpha = 1.0 if time_constant_s <= 0.0 else step_dt / (time_constant_s + step_dt)
    filtered = np.zeros_like(raw_velocity)
    state = np.zeros_like(raw_velocity[0])
    for index in range(raw_velocity.shape[0]):
        state = state * (1.0 - alpha) + raw_velocity[index] * alpha
        filtered[index] = state
    speed_mm_s = 1000.0 * np.linalg.norm(filtered, axis=-1)
    return {
        "time_constant_s": float(time_constant_s),
        "filter_alpha": float(alpha),
        # A first-order filter reaches 95% of a step in three time constants.
        # This is what the filter costs, and it belongs beside what it buys.
        "settling_95_percent_s": 0.0 if alpha >= 1.0 else 3.0 * float(time_constant_s),
        "speed_mm_s_mean": float(speed_mm_s.mean()),
        "speed_mm_s_p95": _percentile(speed_mm_s, 0.95),
        "speed_mm_s_max": float(speed_mm_s.max()),
    }


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    try:
        env.reset(seed=args.seed)
        inner = env.unwrapped
        estimator = shared_surrogate_estimator(inner)
        step_dt = float(inner.step_dt)
        action_dim = int(inner.action_manager.total_action_dim)
        # Zero actions: the arm holds, the rails hold the module, and what moves
        # in the observation is the estimator.
        actions = torch.zeros((inner.num_envs, action_dim), device=inner.device)

        estimated: list[np.ndarray] = []
        truth: list[np.ndarray] = []
        for _ in range(args.steps):
            env.step(actions)
            pose, _ = estimator.estimate()
            estimated.append(pose.detach().cpu().numpy().copy())
            truth.append(module_pose_label(inner).detach().cpu().numpy().copy())

        estimates = np.stack(estimated)  # (steps, envs, 6)
        truths = np.stack(truth)
        surrogate_description = estimator.describe()
        camera_period_steps = int(estimator.camera_period_steps)
    finally:
        env.close()

    position_error_mm = 1000.0 * np.linalg.norm(estimates[..., :3] - truths[..., :3], axis=-1)
    orientation_error_rad = np.linalg.norm(estimates[..., 3:] - truths[..., 3:], axis=-1)

    certified_position_p95_mm = float(surrogate_description["certified_position_p95_mm"])
    certified_orientation_p95_rad = float(surrogate_description["certified_orientation_p95_rad"])
    realized_position_p95_mm = _percentile(position_error_mm, 0.95)
    realized_orientation_p95_rad = _percentile(orientation_error_rad, 0.95)

    position_ratio = realized_position_p95_mm / certified_position_p95_mm
    orientation_ratio = realized_orientation_p95_rad / certified_orientation_p95_rad
    agrees = (
        abs(position_ratio - 1.0) <= args.quantile_tolerance
        and abs(orientation_ratio - 1.0) <= args.quantile_tolerance
    )

    # The staircase: how often the estimate actually changes. A surrogate at the
    # control rate would show this at 1.0 and would not be a model of a 15 Hz
    # camera at all.
    changed = np.abs(np.diff(estimates, axis=0)).sum(axis=-1) > 0.0
    changed_fraction = float(changed.mean())

    raw_velocity = np.diff(estimates[..., :3], axis=0) / step_dt
    sweep = [_filtered_speed_statistics(raw_velocity, step_dt, tau) for tau in args.filter_sweep_s]
    deployed = min(sweep, key=lambda row: abs(row["time_constant_s"] - 0.10))

    # The prediction this whole line of work started from: a residual of the
    # certified size, appearing over one camera period, is this much velocity.
    predicted_noise_floor_mm_s = certified_position_p95_mm / (camera_period_steps * step_dt)

    report = {
        "title": "Training-time estimator surrogate, measured against the certificate it reproduces",
        "what_this_is": (
            "The arm holds still and the module sits in its rails, so the true state barely moves and "
            "what the observation channels show is the estimator. The realized error quantiles are "
            "compared with the certified ones the sigma was inverted from, and the velocity channel's "
            "noise floor is reported against the speed a seated module actually has."
        ),
        "scope": [
            "Simulation only. No result here was produced on real hardware.",
            "The module pose is read for error metrics only; no policy observation carries it.",
            (
                "The surrogate reproduces the sensor's residual, its sample-and-hold and its miss rate. "
                "It does not reproduce occlusion geometry or the heavy tail a losing episode shows once "
                "the module leaves the cameras' useful envelope, so it is a lower bound on deployment error."
            ),
        ],
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "control_step_s": step_dt,
        "surrogate": surrogate_description,
        "self_validation": {
            "certified_position_p95_mm": certified_position_p95_mm,
            "realized_position_p95_mm": realized_position_p95_mm,
            "position_ratio": position_ratio,
            "certified_orientation_p95_rad": certified_orientation_p95_rad,
            "realized_orientation_p95_rad": realized_orientation_p95_rad,
            "orientation_ratio": orientation_ratio,
            "relative_tolerance": args.quantile_tolerance,
            "agrees": bool(agrees),
        },
        "sample_and_hold": {
            "camera_period_control_steps": camera_period_steps,
            "fraction_of_control_steps_where_the_estimate_moves": changed_fraction,
            "expected_fraction": 1.0 / camera_period_steps,
        },
        "velocity_channel": {
            "seated_module_speed_mm_s": args.seated_speed_mm_s,
            "predicted_noise_floor_mm_s": predicted_noise_floor_mm_s,
            "prediction": (
                "certified position p95 divided by one camera period: a residual of that size, appearing "
                "between two camera frames, is indistinguishable from motion of that speed"
            ),
            "deployed_filter": deployed,
            "deployed_noise_floor_over_seated_speed": deployed["speed_mm_s_mean"] / args.seated_speed_mm_s,
            "filter_sweep": sweep,
        },
        "source_revision": git_source_revision(PROJECT_ROOT),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"position p95   certified {certified_position_p95_mm:.3f} mm   realized {realized_position_p95_mm:.3f} mm")
    print(
        f"orientation p95 certified {certified_orientation_p95_rad * 1000:.2f} mrad   "
        f"realized {realized_orientation_p95_rad * 1000:.2f} mrad"
    )
    print(f"estimate moves on {changed_fraction:.3f} of control steps (expected {1.0 / camera_period_steps:.3f})")
    print(
        f"velocity channel at the deployed filter: {deployed['speed_mm_s_mean']:.2f} mm/s mean, "
        f"{deployed['speed_mm_s_mean'] / args.seated_speed_mm_s:.1f}x a seated module's own speed"
    )
    for row in sweep:
        print(f"  tau={row['time_constant_s']:.2f}s  mean {row['speed_mm_s_mean']:7.2f} mm/s  p95 {row['speed_mm_s_p95']:7.2f}")
    print(f"wrote {args.report}")
    if not agrees:
        print("SURROGATE DOES NOT REPRODUCE ITS CERTIFICATE; do not train against it")
        return 1
    return 0


try:
    exit_code = main()
finally:
    simulation_app.close()
sys.exit(exit_code)
