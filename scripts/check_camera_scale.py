"""Render one frame and measure what the servicing camera can actually resolve.

The original 64x64 camera at 180 mm resolved a 4 mm displacement but failed the
rendered framing gate: the slot mouth projected outside the image. This script
checks both requirements together before perception training starts, so optics
that pass scale arithmetic while looking at the wrong patch cannot pass.

Three things are reported and all three are gates:

1. the ground resolution in millimetres per pixel, from the configured optics;
2. whether both slot mouths, the free-transfer midpoint, and the module centre
   are inside the frame, computed by projecting their world positions through
   the camera;
3. whether the rendered image carries signal at all rather than a flat field,
   because a correctly aimed camera pointed at featureless geometry is just as
   useless as a mis-aimed one.

CPU-cheap and GPU-light: one environment, one frame.
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

#: The displacement the perception stage has to resolve, from
#: docs/service_interface_spec.md section 7.
REQUIRED_LATERAL_RESOLUTION_M = 0.004
#: Below this a displacement is not a regression problem, it is absent signal.
MINIMUM_PIXELS_PER_DISPLACEMENT = 1.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-Insertion-Vision-v0")
    parser.add_argument("--report", type=Path, default=Path("evidence/camera_scale.json"))
    parser.add_argument("--frame", type=Path, default=Path("artifacts/perception/frame.png"))
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
from zero_g_blade_swap.tasks.blade_swap.assets import BLADE_INSERTED_POS, SECOND_SLOT_CENTER_Y
from zero_g_blade_swap.grapple_geometry import SLOT_MOUTH_X, TRANSIT_CLEAR_BLADE_CENTRE_X


def _project(point_w: torch.Tensor, camera) -> tuple[float, float]:
    """Project a world point into pixel coordinates for environment 0."""

    position = camera.data.pos_w[0]
    orientation = camera.data.quat_w_ros[0]
    from isaaclab.utils.math import quat_inv, quat_apply

    local = quat_apply(quat_inv(orientation).unsqueeze(0), (point_w - position).unsqueeze(0))[0]
    intrinsics = camera.data.intrinsic_matrices[0]
    depth = float(local[2])
    if depth <= 0.0:
        return float("nan"), float("nan")
    u = float(intrinsics[0, 0]) * float(local[0]) / depth + float(intrinsics[0, 2])
    v = float(intrinsics[1, 1]) * float(local[1]) / depth + float(intrinsics[1, 2])
    return u, v


def main() -> dict[str, object]:
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
        env = gym.make(args.task, cfg=env_cfg)
        task = env.unwrapped
        task.reset()
        for _ in range(3):
            task.step(torch.zeros((1, task.action_manager.total_action_dim), device=task.device))

        camera = task.scene["camera"]
        spawn = env_cfg.scene.camera.spawn
        width = int(env_cfg.scene.camera.width)
        height = int(env_cfg.scene.camera.height)
        focal_length = float(spawn.focal_length)
        aperture = float(spawn.horizontal_aperture)
        fov_rad = 2.0 * math.atan(0.5 * aperture / focal_length)

        origin = task.scene.env_origins[0]
        mouth_w = origin + origin.new_tensor((SLOT_MOUTH_X, 0.0, BLADE_INSERTED_POS[2]))
        second_mouth_w = origin + origin.new_tensor(
            (SLOT_MOUTH_X, SECOND_SLOT_CENTER_Y, BLADE_INSERTED_POS[2])
        )
        transfer_w = origin + origin.new_tensor(
            (TRANSIT_CLEAR_BLADE_CENTRE_X, 0.5 * SECOND_SLOT_CENTER_Y, BLADE_INSERTED_POS[2])
        )
        blade_w = task.scene["spare_blade"].data.root_pos_w[0]
        distance = float(torch.linalg.vector_norm(camera.data.pos_w[0] - mouth_w))
        metres_per_pixel = 2.0 * distance * math.tan(0.5 * fov_rad) / width

        mouth_px = _project(mouth_w, camera)
        second_mouth_px = _project(second_mouth_w, camera)
        transfer_px = _project(transfer_w, camera)
        blade_px = _project(blade_w, camera)
        projected = {
            "first_slot_mouth": mouth_px,
            "second_slot_mouth": second_mouth_px,
            "transfer_clear_midpoint": transfer_px,
            "module_centre": blade_px,
        }
        in_frame = {
            name: bool(0 <= pixel[0] < width and 0 <= pixel[1] < height)
            for name, pixel in projected.items()
        }

        rgb = camera.data.output["rgb"][0].to(torch.float32)
        signal = {
            "mean": float(rgb.mean()),
            "std": float(rgb.std()),
            "unique_intensity_levels": int(torch.unique(rgb.mean(dim=-1)).numel()),
        }
        args.frame.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            Image.fromarray(camera.data.output["rgb"][0].clamp(0, 255).to(torch.uint8).cpu().numpy()).save(args.frame)
            wrote_frame = str(args.frame)
        except Exception:  # noqa: BLE001 -- a missing encoder must not fail the gate
            wrote_frame = None

        pixels_per_displacement = REQUIRED_LATERAL_RESOLUTION_M / metres_per_pixel
        passed = (
            pixels_per_displacement >= MINIMUM_PIXELS_PER_DISPLACEMENT
            and all(in_frame.values())
            and signal["std"] > 1.0
        )
        report = {
            "status": "passed" if passed else "failed",
            "title": "Servicing camera scale and framing, measured by rendering",
            "evidence_type": "simulation_sensor_characterization",
            "protocol": {
                "task": args.task,
                "resolution_px": width,
                "focal_length_mm": focal_length,
                "horizontal_aperture_mm": aperture,
                "horizontal_field_of_view_deg": math.degrees(fov_rad),
                "camera_to_slot_mouth_m": distance,
            },
            "resolution": {
                "metres_per_pixel": metres_per_pixel,
                "millimetres_per_pixel": 1_000.0 * metres_per_pixel,
                "required_lateral_resolution_m": REQUIRED_LATERAL_RESOLUTION_M,
                "pixels_per_required_displacement": pixels_per_displacement,
            },
            "framing": {"pixel_coordinates": projected, "in_frame": in_frame},
            "signal": signal,
            "frame": wrote_frame,
            "gate": {
                "applies": True,
                "minimum_pixels_per_displacement": MINIMUM_PIXELS_PER_DISPLACEMENT,
                "passed": passed,
                "rationale": (
                    "A displacement smaller than a pixel is absent signal, not a hard regression problem. "
                    "Framing is gated as well because a correctly scaled camera aimed at the wrong place "
                    "resolves nothing, and that cannot be caught by arithmetic."
                ),
            },
            "scope_and_limitations": [
                "Simulation only. This characterises the authored sensor, not a calibrated real camera.",
                "One frame at one reset. It says the interface is framed and resolvable, not that a policy can learn from it.",
            ],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["resolution"], indent=2))
        print(json.dumps(report["framing"]["in_frame"], indent=2))
        print(f"[{'PASS' if passed else 'FAIL'}] {args.report}")
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
