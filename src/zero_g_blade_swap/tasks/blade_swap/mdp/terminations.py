"""Task termination predicates."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

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
    maximum_displacement: float = 0.018,
    maximum_rotation: float = 0.0436332313,
) -> torch.Tensor:
    """Fail loudly if the simulated mount leaves its intended compliance envelope."""

    robot = env.scene[asset_cfg.name]
    position = robot.data.root_pos_w - env.scene.env_origins
    displacement = torch.linalg.vector_norm(position - robot.data.default_root_state[:, :3], dim=-1)
    rotation = quat_error_magnitude(robot.data.root_quat_w, robot.data.default_root_state[:, 3:7])
    return (displacement > maximum_displacement) | (rotation > maximum_rotation)


def non_finite_state(env) -> torch.Tensor:
    """Terminate any environment containing a NaN or infinity."""

    invalid = ~torch.isfinite(env.scene["robot"].data.joint_pos).all(dim=-1)
    invalid |= ~torch.isfinite(env.scene["robot"].data.joint_vel).all(dim=-1)
    invalid |= ~torch.isfinite(env.scene["robot"].data.root_state_w).all(dim=-1)
    for name in ("blade", "spare_blade"):
        invalid |= ~torch.isfinite(env.scene[name].data.root_state_w).all(dim=-1)
    return invalid


__all__ = ["blade_inserted", "blade_out_of_workspace", "non_finite_state", "robot_mount_unstable"]
