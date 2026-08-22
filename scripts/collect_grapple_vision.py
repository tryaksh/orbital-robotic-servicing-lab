"""Record what the servicing camera sees, and where the module actually was.

Supervision for ``mdp.ModulePoseHead``. Each row is a 256x256 RGB frame and the
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

Written as one contiguous array of uint8 frames rather than HDF5 so a single
``.npz`` needs no optional dependency. Collection counts should account for the
larger framed camera rather than assuming the former 64x64 memory footprint.
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
    parser.add_argument(
        "--rgb_source",
        choices=("noisy", "raw"),
        default="noisy",
        help=(
            "Save the existing radiation-noise observation for learned heads, or the calibrated "
            "camera RGB stream used by deterministic fiducial perception."
        ),
    )
    parser.add_argument("--curriculum_stages", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--pose_distribution",
        choices=("reset_jitter", "workflow_envelope"),
        default="reset_jitter",
        help=(
            "reset_jitter reproduces the legacy near-rack dataset; workflow_envelope balances both bays "
            "with free-transfer poses, including the neither-bay occupancy class required by relocation."
        ),
    )
    parser.add_argument(
        "--camera_offset_mm",
        type=float,
        default=0.0,
        help=(
            "Displace the camera's configured mount by this much along a random direction, constant for "
            "the run. Collecting several runs at different offsets is how the head is made robust to "
            "calibration error, which it is sharply brittle to when trained through a perfect camera."
        ),
    )
    parser.add_argument("--camera_tilt_mrad", type=float, default=0.0, help="Mount rotation error for the run.")
    parser.add_argument(
        "--diagnose_render_steps",
        action="store_true",
        help=(
            "Print raw camera, noisy observation, and sensor-frame statistics for the first round. "
            "This is a collection-pipeline diagnostic; it does not change the saved arrays."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _parser()
args = parser.parse_args()
if args.settle_steps < 2:
    parser.error("--settle_steps must be at least 2: the 15 Hz camera refreshes every two 30 Hz control steps")
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import axis_angle_from_quat, quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg

import zero_g_blade_swap.tasks.blade_swap  # noqa: F401
from zero_g_blade_swap.grapple_geometry import TRANSIT_CLEAR_BLADE_CENTRE_X
from zero_g_blade_swap.tasks.blade_swap.assets import BLADE_INSERTED_POS, SECOND_SLOT_CENTER_Y

# A raw camera frame containing the rendered workcell is comfortably above
# these conservative floors (the measured 64-env smokes have minimum per-frame
# standard deviation >57 and maximum 255). Radiation noise over an all-black
# render produced std 4.31 and maximum 60 in the rejected corpus. Keep the gate
# between those regimes, not tuned against model accuracy.
MIN_FRAME_SIGNAL_STD = 15.0
MIN_FRAME_SIGNAL_MAX = 128.0
CAMERA_FRAME_WAIT_MARGIN = 4
MAX_CAPTURE_POSITION_DRIFT_M = 1.0e-4
MAX_CAPTURE_ORIENTATION_DRIFT_RAD = 1.0e-4


def _tensor_summary(value: torch.Tensor) -> str:
    value_float = value.to(torch.float32)
    return (
        f"shape={tuple(value.shape)} dtype={value.dtype} "
        f"range=[{float(value_float.min()):.4f}, {float(value_float.max()):.4f}] "
        f"mean={float(value_float.mean()):.4f} std={float(value_float.std()):.4f}"
    )


def _print_render_probe(task, observations: dict[str, torch.Tensor], point: str) -> None:
    camera = task.scene["camera"]
    raw = camera.data.output["rgb"]
    observed = observations["rgb"]
    frame = camera.frame
    print(
        f"[RENDER-DIAG] {point}: sensor_frame=[{int(frame.min())}, {int(frame.max())}] "
        f"sim_step={int(task.common_step_counter)}",
        flush=True,
    )
    print(f"[RENDER-DIAG] raw rgb: {_tensor_summary(raw)}", flush=True)
    print(f"[RENDER-DIAG] observed rgb: {_tensor_summary(observed)}", flush=True)


def _raw_camera_signal(task) -> dict[str, float | bool]:
    """Synchronize with the renderer and characterize every tiled-camera view."""

    raw = task.scene["camera"].data.output["rgb"].to(torch.float32).flatten(start_dim=1)
    per_frame_std = raw.std(dim=1)
    per_frame_max = raw.amax(dim=1)
    passed = bool(torch.all(per_frame_std >= MIN_FRAME_SIGNAL_STD) and torch.all(per_frame_max >= MIN_FRAME_SIGNAL_MAX))
    # Converting reductions to Python values is intentional: besides producing
    # an auditable gate, this synchronizes the external render stream before a
    # torch observation is copied to CPU.
    return {
        "passed": passed,
        "per_frame_std_min": float(per_frame_std.min()),
        "per_frame_std_median": float(per_frame_std.median()),
        "per_frame_max_min": float(per_frame_max.min()),
        "global_max": float(per_frame_max.max()),
    }


def _wait_for_fresh_camera_observation(
    task,
    zero: torch.Tensor,
    minimum_steps: int,
    preserved_root_pose: torch.Tensor | None = None,
):
    """Step until all camera tiles advance and contain rendered scene signal."""

    camera = task.scene["camera"]
    blade = task.scene["spare_blade"]
    frame_before = camera.frame.clone()
    last_signal: dict[str, float | bool] | None = None
    observations = None
    limit = minimum_steps + CAMERA_FRAME_WAIT_MARGIN
    for step_index in range(limit):
        if preserved_root_pose is not None:
            # The collection-only blade is kinematic, and its authored target is
            # reasserted before every possible render step. Thus the RGB frame,
            # pose label, and intended label describe one simultaneous state
            # rather than a body depenetrating while the camera catches up.
            blade.write_root_pose_to_sim(preserved_root_pose)
            blade.write_root_velocity_to_sim(torch.zeros((task.num_envs, 6), device=task.device))
        observations, _, _, _, _ = task.step(zero)
        advanced = bool(torch.all(camera.frame > frame_before))
        if step_index + 1 < minimum_steps or not advanced:
            continue
        last_signal = _raw_camera_signal(task)
        if last_signal["passed"]:
            return observations, last_signal, step_index + 1

    raise RuntimeError(
        "servicing camera did not produce a fresh rendered scene within "
        f"{limit} control steps; frame_before=[{int(frame_before.min())}, {int(frame_before.max())}], "
        f"frame_after=[{int(camera.frame.min())}, {int(camera.frame.max())}], signal={last_signal}. "
        "Collection was aborted before replacing the target dataset."
    )


def _dataset_signal_statistics(images: np.ndarray) -> dict[str, float | int | bool]:
    """Measure a bounded, evenly spaced sample before committing the archive."""

    sample_count = min(len(images), 256)
    indices = np.linspace(0, len(images) - 1, sample_count, dtype=np.int64)
    sample = images[indices].astype(np.float32, copy=False).reshape(sample_count, -1)
    per_frame_std = sample.std(axis=1)
    per_frame_max = sample.max(axis=1)
    return {
        "sampled_frames": sample_count,
        "per_frame_std_min": float(per_frame_std.min()),
        "per_frame_std_median": float(np.median(per_frame_std)),
        "per_frame_max_min": float(per_frame_max.min()),
        "global_mean": float(sample.mean()),
        "global_std": float(sample.std()),
        "global_max": int(per_frame_max.max()),
        "passed": bool(np.all(per_frame_std >= MIN_FRAME_SIGNAL_STD) and np.all(per_frame_max >= MIN_FRAME_SIGNAL_MAX)),
    }


def _sample_workflow_envelope(task, round_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Place modules across both rack approaches and the free-transfer volume.

    The legacy collector only jittered the three insertion reset depths. It
    therefore had no examples of a module between bays and produced zero
    ``[0, 0]`` occupancy labels. This balanced sampler covers the states the
    deployed full-chain estimator is asked to see, while labels are still read
    from the post-physics state paired with the rendered frame.
    """

    count = task.num_envs
    device = task.device
    blade = task.scene["spare_blade"]
    categories = (torch.arange(count, device=device) + round_index) % 3
    local = torch.zeros((count, 3), device=device)
    roll = torch.zeros(count, device=device)
    pitch = torch.zeros(count, device=device)
    yaw = torch.zeros(count, device=device)

    for category, bay_y in ((0, 0.0), (1, SECOND_SLOT_CENTER_Y)):
        mask = categories == category
        samples = int(mask.sum())
        if samples == 0:
            continue
        # Approach through seated: aligned enough to be collision-plausible,
        # but wider than the policy's success tolerances so perception cannot
        # memorize a nominal pose.
        local[mask, 0] = 0.55 + 0.18 * torch.rand(samples, device=device)
        local[mask, 1] = bay_y + 0.012 * (2.0 * torch.rand(samples, device=device) - 1.0)
        local[mask, 2] = BLADE_INSERTED_POS[2] + 0.010 * (2.0 * torch.rand(samples, device=device) - 1.0)
        roll[mask] = 0.05 * (2.0 * torch.rand(samples, device=device) - 1.0)
        pitch[mask] = 0.05 * (2.0 * torch.rand(samples, device=device) - 1.0)
        yaw[mask] = 0.08 * (2.0 * torch.rand(samples, device=device) - 1.0)

    transfer = categories == 2
    samples = int(transfer.sum())
    if samples:
        # Behind the flare plane, across the entire bay pitch. These are true
        # neither-bay examples and include the attitude excursions measured in
        # the failed relocation transit.
        local[transfer, 0] = (TRANSIT_CLEAR_BLADE_CENTRE_X - 0.08) + 0.14 * torch.rand(samples, device=device)
        local[transfer, 1] = (SECOND_SLOT_CENTER_Y - 0.025) + (abs(SECOND_SLOT_CENTER_Y) + 0.050) * torch.rand(
            samples, device=device
        )
        local[transfer, 2] = BLADE_INSERTED_POS[2] + 0.050 * (2.0 * torch.rand(samples, device=device) - 1.0)
        roll[transfer] = 0.25 * (2.0 * torch.rand(samples, device=device) - 1.0)
        pitch[transfer] = 0.25 * (2.0 * torch.rand(samples, device=device) - 1.0)
        yaw[transfer] = 0.40 * (2.0 * torch.rand(samples, device=device) - 1.0)

    orientation = quat_from_euler_xyz(roll, pitch, yaw)
    pose = blade.data.root_state_w[:, :7].clone()
    pose[:, :3] = local + task.scene.env_origins
    pose[:, 3:7] = orientation
    blade.write_root_pose_to_sim(pose)
    blade.write_root_velocity_to_sim(torch.zeros((count, 6), device=device))
    return torch.cat((local, axis_angle_from_quat(orientation)), dim=-1), pose


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
        env_cfg.seed = args.seed
        # The oracle path, so the term does not demand a head that does not exist
        # yet. Nothing here reads the grip error anyway; the arm holds still.
        env_cfg.pose_head_oracle_blend = 1.0
        # Collection is a rendering experiment, not a dynamics rollout.  A
        # free module can move substantially while the asynchronous tiled
        # camera advances; the rejected overview corpus measured 107 mm p95
        # drift in the transfer class.  Make the collection-only body
        # kinematic and reassert the authored pose on every possible render
        # step below.  Deployment keeps the ordinary dynamic body.
        env_cfg.scene.spare_blade.spawn.rigid_props.kinematic_enabled = True
        if args.camera_offset_mm != 0.0 or args.camera_tilt_mrad != 0.0:
            rng = np.random.default_rng(args.seed)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            offset = args.camera_offset_mm / 1_000.0
            nominal = tuple(env_cfg.scene.camera.offset.pos)
            env_cfg.scene.camera.offset.pos = tuple(
                float(p + offset * d) for p, d in zip(nominal, direction, strict=True)
            )
            if args.camera_tilt_mrad != 0.0:
                half = 0.5 * args.camera_tilt_mrad / 1_000.0
                w, x, y, z = env_cfg.scene.camera.offset.rot
                cw, cz = float(np.cos(half)), float(np.sin(half))
                env_cfg.scene.camera.offset.rot = (
                    cw * w - cz * z,
                    cw * x - cz * y,
                    cw * y + cz * x,
                    cw * z + cz * w,
                )
            print(
                f"[INFO] camera mount displaced {args.camera_offset_mm} mm, "
                f"tilted {args.camera_tilt_mrad} mrad for this run"
            )
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        stages = torch.tensor(args.curriculum_stages, device=task.device)
        # The workflow profile carries no curriculum, so nothing has created the
        # stage buffer the reset events read. Create it rather than depending on
        # a term this task deliberately does not have.
        if getattr(task, "_insertion_curriculum_stage", None) is None:
            task._insertion_curriculum_stage = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)

        images: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        intended_labels: list[np.ndarray] = []
        round_ids: list[np.ndarray] = []
        environment_ids: list[np.ndarray] = []
        capture_sensor_frames: list[np.ndarray] = []
        capture_wait_steps: list[np.ndarray] = []
        # Recorded only where the task offers it, so the single-bay collector is
        # unchanged and the dataset it writes keeps loading. On a two-bay rack
        # this is the supervision for the occupancy branch: which bay, if either,
        # currently holds the module.
        occupancies: list[np.ndarray] = []
        zero = torch.zeros((task.num_envs, task.action_manager.total_action_dim), device=task.device)

        # Prime the renderer independently of the diagnostic flag. Tiled-camera
        # results are produced on an external stream; a reduced raw-frame check
        # both synchronizes that stream and refuses the all-zero render that
        # previously became a noise-only 6000-frame corpus.
        task.reset()
        _, warmup_signal, warmup_steps = _wait_for_fresh_camera_observation(task, zero, minimum_steps=args.settle_steps)
        print(
            f"[INFO] camera preflight passed after {warmup_steps} steps: {warmup_signal}",
            flush=True,
        )

        collected = 0
        rounds = 0
        while collected < args.samples:
            # Spread the three reset distances across environments so one round
            # samples all of them rather than alternating whole rounds.
            task._insertion_curriculum_stage[:] = stages[
                torch.randint(0, len(stages), (task.num_envs,), device=task.device)
            ]
            reset_observations, _ = task.reset()
            intended_label = reset_observations["pose_label"].clone()
            authored_root_pose = task.scene["spare_blade"].data.root_state_w[:, :7].clone()
            if args.pose_distribution == "workflow_envelope":
                intended_label, authored_root_pose = _sample_workflow_envelope(task, rounds)
            if args.diagnose_render_steps and rounds == 0:
                _print_render_probe(task, reset_observations, "after reset and manual pose write")
            observations, _, waited_steps = _wait_for_fresh_camera_observation(
                task,
                zero,
                minimum_steps=args.settle_steps,
                preserved_root_pose=authored_root_pose,
            )
            if args.diagnose_render_steps and rounds == 0:
                _print_render_probe(task, observations, f"fresh capture after {waited_steps} settle steps")
            if args.rgb_source == "raw":
                frame = task.scene["camera"].data.output["rgb"][..., :3]
                if frame.dtype == torch.uint8:
                    frame = frame.to(torch.float32).mul_(1.0 / 255.0)
                else:
                    frame = frame.to(torch.float32).clamp_(0.0, 1.0)
            else:
                frame = observations["rgb"]
            label = observations["pose_label"]
            images.append((frame.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy())
            if args.rgb_source == "raw":
                depth = task.scene["camera"].data.output["distance_to_image_plane"]
                depths.append(depth.to(torch.float32).cpu().numpy())
            labels.append(label.to(torch.float32).cpu().numpy())
            intended_labels.append(intended_label.to(torch.float32).cpu().numpy())
            round_ids.append(np.full(task.num_envs, rounds, dtype=np.int64))
            environment_ids.append(np.arange(task.num_envs, dtype=np.int64))
            capture_sensor_frames.append(task.scene["camera"].frame.to(torch.int64).cpu().numpy())
            capture_wait_steps.append(np.full(task.num_envs, waited_steps, dtype=np.int64))
            occupancy = observations.get("occupancy_label")
            if occupancy is not None:
                occupancies.append(occupancy.to(torch.float32).cpu().numpy())
            collected += int(frame.shape[0])
            rounds += 1
            if rounds % 25 == 0:
                print(f"[INFO] {collected} / {args.samples} frames", flush=True)

        image_array = np.concatenate(images, axis=0)[: args.samples]
        label_array = np.concatenate(labels, axis=0)[: args.samples]
        intended_label_array = np.concatenate(intended_labels, axis=0)[: args.samples]
        signal = _dataset_signal_statistics(image_array)
        print(f"[INFO] dataset image-signal gate: {signal}", flush=True)
        if not signal["passed"]:
            raise RuntimeError(
                f"collected images failed the scene-signal gate; target dataset was not replaced: {signal}"
            )
        position_drift_m = np.linalg.norm(label_array[:, :3] - intended_label_array[:, :3], axis=1)
        orientation_drift_rad = np.linalg.norm(label_array[:, 3:] - intended_label_array[:, 3:], axis=1)
        max_position_drift_m = float(position_drift_m.max())
        max_orientation_drift_rad = float(orientation_drift_rad.max())
        drift_gate_passed = bool(
            max_position_drift_m <= MAX_CAPTURE_POSITION_DRIFT_M
            and max_orientation_drift_rad <= MAX_CAPTURE_ORIENTATION_DRIFT_RAD
        )
        print(
            "[INFO] frame/label synchronization gate: "
            f"position max={1_000.0 * max_position_drift_m:.4f} mm "
            f"(limit={1_000.0 * MAX_CAPTURE_POSITION_DRIFT_M:.4f}), "
            f"orientation max={max_orientation_drift_rad:.6f} rad "
            f"(limit={MAX_CAPTURE_ORIENTATION_DRIFT_RAD:.6f}), "
            f"passed={drift_gate_passed}",
            flush=True,
        )
        if not drift_gate_passed:
            raise RuntimeError(
                "collected image labels moved away from their authored render poses; target dataset was not replaced"
            )

        arrays = {
            "images": image_array,
            "labels": label_array,
            "intended_labels": intended_label_array,
            "collection_round": np.concatenate(round_ids)[: args.samples],
            "collection_environment": np.concatenate(environment_ids)[: args.samples],
            "capture_sensor_frame": np.concatenate(capture_sensor_frames)[: args.samples],
            "capture_wait_steps": np.concatenate(capture_wait_steps)[: args.samples],
            "pose_distribution": np.asarray(args.pose_distribution),
            "collection_task": np.asarray(args.task),
            "collection_seed": np.asarray(args.seed, dtype=np.int64),
            "rgb_source": np.asarray(args.rgb_source),
            "collection_num_envs": np.asarray(args.num_envs, dtype=np.int64),
            "collection_settle_steps_minimum": np.asarray(args.settle_steps, dtype=np.int64),
            "camera_frame_wait_margin": np.asarray(CAMERA_FRAME_WAIT_MARGIN, dtype=np.int64),
            "camera_offset_mm": np.asarray(args.camera_offset_mm, dtype=np.float32),
            "camera_tilt_mrad": np.asarray(args.camera_tilt_mrad, dtype=np.float32),
            "collection_pose_hold": np.asarray("kinematic_reasserted_each_control_step"),
            "frame_label_sync_gate_passed": np.asarray(True),
            "capture_position_drift_max_m": np.asarray(max_position_drift_m, dtype=np.float64),
            "capture_orientation_drift_max_rad": np.asarray(max_orientation_drift_rad, dtype=np.float64),
            "capture_position_drift_limit_m": np.asarray(MAX_CAPTURE_POSITION_DRIFT_M, dtype=np.float64),
            "capture_orientation_drift_limit_rad": np.asarray(MAX_CAPTURE_ORIENTATION_DRIFT_RAD, dtype=np.float64),
            "lighting_randomization": np.asarray(
                "VisualRandomizationCfg.orbital_sun_reset; reproducible by collection_seed and round"
            ),
            "image_signal_std": np.asarray(signal["global_std"], dtype=np.float32),
            "image_signal_max": np.asarray(signal["global_max"], dtype=np.int64),
        }
        if occupancies:
            arrays["occupancy"] = np.concatenate(occupancies, axis=0)[: args.samples]
        if depths:
            arrays["depth_image_plane_m"] = np.concatenate(depths, axis=0)[: args.samples]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        spread = label_array.max(axis=0) - label_array.min(axis=0)
        print(f"[INFO] wrote {args.output}: images {image_array.shape}, labels {label_array.shape}")
        print(f"[INFO] label spread per channel (m, m, m, rad, rad, rad): {np.round(spread, 5).tolist()}")
        print(
            f"[INFO] pose drift during render wait: {float(1_000.0 * position_drift_m.mean()):.4f} "
            f"mm mean, {float(1_000.0 * np.quantile(position_drift_m, 0.95)):.4f} mm p95"
        )
        if occupancies:
            occupied = arrays["occupancy"]
            # Printed because a classification dataset that turns out to be 99%
            # one class trains a head that looks accurate and has learned the
            # prior. Better to see it here than in a suspiciously good number.
            print(
                f"[INFO] occupancy label: per-bay positive fraction "
                f"{np.round(occupied.mean(axis=0), 4).tolist()}, "
                f"neither bay in {float((occupied.sum(axis=1) == 0).mean()):.4f} of frames"
            )
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
