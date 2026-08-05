"""Task termination predicates."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat, subtract_frame_transforms

from .commands import swap_complete_mask


def blade_inserted(env, command_name: str = "blade_goal") -> torch.Tensor:
    """Terminate on a stable insertion that occurred after extraction."""

    return swap_complete_mask(env, command_name)


def blade_out_of_workspace(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("blade"),
) -> torch.Tensor:
    """Stop irrecoverable blade escapes in zero gravity."""

    lost = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in (asset_cfg.name, "spare_blade"):
        blade = env.scene[name]
        position = blade.data.root_pos_w - env.scene.env_origins
        lost |= (
            (position[:, 0] < -0.10)
            | (position[:, 0] > 1.30)
            | (position[:, 1].abs() > 0.75)
            | (position[:, 2] < 0.20)
            | (position[:, 2] > 1.35)
        )
    return lost


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


def non_finite_state(env) -> torch.Tensor:
    """Terminate any environment containing a NaN or infinity."""

    invalid = ~torch.isfinite(env.scene["robot"].data.joint_pos).all(dim=-1)
    invalid |= ~torch.isfinite(env.scene["robot"].data.joint_vel).all(dim=-1)
    invalid |= ~torch.isfinite(env.scene["robot"].data.root_state_w).all(dim=-1)
    for name in ("blade", "spare_blade"):
        invalid |= ~torch.isfinite(env.scene[name].data.root_state_w).all(dim=-1)
    return invalid


__all__ = ["blade_inserted", "blade_out_of_workspace", "non_finite_state", "robot_mount_unstable"]
