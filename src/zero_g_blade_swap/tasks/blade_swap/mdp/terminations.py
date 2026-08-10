"""Task termination predicates.

The blade-escape and non-finite predicates that lived here read two blades and a
phase state machine, so they went with the eight-phase swap task on 2026-08-10.
The insertion family carries its own equivalents in ``insertion.py``, scoped to
the single blade its scene actually spawns. This one survives because a
compliant robot mount is a property of the workcell, not of a task.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat, subtract_frame_transforms


def robot_mount_unstable(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("mount_anchor"),
    maximum_translation_axis: float = 0.0165,
    maximum_rotation_axis: float = 0.0383972435,
) -> torch.Tensor:
    """Fail if any D6 axis exceeds its authored limit plus solver tolerance."""

    robot = env.scene[asset_cfg.name]
    anchor = env.scene[anchor_cfg.name]
    relative_pos, relative_quat = subtract_frame_transforms(
        anchor.data.root_pos_w,
        anchor.data.root_quat_w,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )
    relative_rotation = axis_angle_from_quat(relative_quat)
    translation_trip = (relative_pos.abs() > maximum_translation_axis).any(dim=-1)
    rotation_trip = (relative_rotation.abs() > maximum_rotation_axis).any(dim=-1)
    return translation_trip | rotation_trip


__all__ = ["robot_mount_unstable"]
