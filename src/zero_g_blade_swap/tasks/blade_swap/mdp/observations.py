"""Vector, privileged, and image observations for blade swapping."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from ..assets import TOOL_OFFSET_POS, TOOL_OFFSET_ROT


def _single_body_id(asset, asset_cfg: SceneEntityCfg) -> int:
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, list) and len(body_ids) == 1:
        return body_ids[0]
    # SceneEntityCfg parameters owned by a manager are resolved at manager
    # construction. Helpers also call this function directly from the command
    # state machine, where the default config has not gone through that pass.
    if asset_cfg.body_names:
        resolved_ids, resolved_names = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
        if len(resolved_ids) == 1:
            return int(resolved_ids[0])
        raise ValueError(f"Observation expects exactly one body for '{asset_cfg.name}', resolved {resolved_names}.")
    raise ValueError(f"Observation expects exactly one body for '{asset_cfg.name}', resolved body IDs {body_ids}.")


def end_effector_pose_world(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Robotiq grasp-frame pose in world coordinates."""

    robot = env.scene[asset_cfg.name]
    body_id = _single_body_id(robot, asset_cfg)
    wrist_pos = robot.data.body_pos_w[:, body_id]
    wrist_rot = robot.data.body_quat_w[:, body_id]
    offset_pos_cfg = getattr(env.cfg, "tool_offset_pos", TOOL_OFFSET_POS)
    offset_pos = torch.tensor(offset_pos_cfg, device=env.device).expand(env.num_envs, -1)
    offset_rot = torch.tensor(TOOL_OFFSET_ROT, device=env.device).expand(env.num_envs, -1)
    return combine_frame_transforms(wrist_pos, wrist_rot, offset_pos, offset_rot)


def end_effector_pose_local(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
) -> torch.Tensor:
    """Return grasp-frame position and quaternion in the environment frame."""

    position, orientation = end_effector_pose_world(env, asset_cfg)
    return torch.cat((position - env.scene.env_origins, orientation), dim=-1)


def blade_pose_local(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("spare_blade"),
) -> torch.Tensor:
    """Return blade position and quaternion in the environment frame."""

    blade = env.scene[asset_cfg.name]
    return torch.cat((blade.data.root_pos_w - env.scene.env_origins, blade.data.root_quat_w), dim=-1)


def normalized_joint_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return joint velocity normalized by actuator soft limits."""

    robot = env.scene[asset_cfg.name]
    limits = robot.data.soft_joint_vel_limits[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    return (robot.data.joint_vel[:, asset_cfg.joint_ids] / limits).clamp(-1.0, 1.0)


def blade_pose_in_end_effector_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
    blade_cfg: SceneEntityCfg = SceneEntityCfg("spare_blade"),
) -> torch.Tensor:
    """Return the blade pose relative to the grasp frame."""

    ee_pos, ee_rot = end_effector_pose_world(env, robot_cfg)
    blade = env.scene[blade_cfg.name]
    rel_pos, rel_rot = subtract_frame_transforms(ee_pos, ee_rot, blade.data.root_pos_w, blade.data.root_quat_w)
    return torch.cat((rel_pos, rel_rot), dim=-1)


def robot_root_pose_local(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Expose true D6 base deflection to the asymmetric critic."""

    robot = env.scene[asset_cfg.name]
    return torch.cat((robot.data.root_pos_w - env.scene.env_origins, robot.data.root_quat_w), dim=-1)


def robot_mount_state(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("mount_anchor"),
) -> torch.Tensor:
    """Return D6 mount deflection pose plus root linear/angular velocity."""

    robot = env.scene[asset_cfg.name]
    anchor = env.scene[anchor_cfg.name]
    relative_pos, relative_quat = subtract_frame_transforms(
        anchor.data.root_pos_w,
        anchor.data.root_quat_w,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )
    return torch.cat((relative_pos, relative_quat, robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1)


def randomized_physics_parameters(env) -> torch.Tensor:
    """Expose normalized privileged dynamics parameters to the critic only.

    Six values for the one blade the insertion scenes spawn: its mass, the guide
    rail's static/dynamic friction and restitution, and its sampled breakaway
    and viscous rail resistance. Scales are the configured randomization ranges,
    so a critic input stays near the unit interval whether or not the level
    being trained actually randomizes that parameter.

    The rail material is read from the left guide rather than the slot floor,
    because the rigid-grasp profile disables the floor collider and randomizes
    only the guides.
    """

    mass = env.scene["spare_blade"].root_physx_view.get_masses()[:, :1].to(env.device) / 15.0
    material = env.scene["blade_slot_left_guide"].root_physx_view.get_material_properties()[:, 0, :].to(env.device)
    material_scale = material.new_tensor((2.0, 1.5, 0.05)).clamp_min(1.0e-6)
    material = material / material_scale
    zeros = torch.zeros(env.num_envs, device=env.device)
    breakaway, viscous = getattr(env, "_rail_stiction", {}).get("spare_blade", (zeros, zeros))
    stiction = torch.stack((breakaway / 120.0, viscous / 25.0), dim=-1)
    return torch.cat((mass, material, stiction), dim=-1)


def camera_rgb_with_radiation_noise(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    noise_std_range: tuple[float, float] = (0.01, 0.045),
) -> torch.Tensor:
    """Return normalized RGB with per-environment Gaussian radiation noise.

    Noise is generated directly on the simulation device.  No CPU image copies
    or per-camera Python loops are introduced.
    """

    image = env.scene.sensors[sensor_cfg.name].data.output[data_type]
    image = image[..., :3]
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32).mul_(1.0 / 255.0)
    else:
        image = image.to(dtype=torch.float32)

    low, high = noise_std_range
    std = torch.empty((env.num_envs, 1, 1, 1), device=image.device).uniform_(low, high)
    noisy = image + torch.randn_like(image) * std
    return noisy.clamp_(0.0, 1.0)


__all__ = [
    "TOOL_OFFSET_POS",
    "TOOL_OFFSET_ROT",
    "blade_pose_in_end_effector_frame",
    "blade_pose_local",
    "camera_rgb_with_radiation_noise",
    "end_effector_pose_local",
    "end_effector_pose_world",
    "normalized_joint_velocity",
    "randomized_physics_parameters",
    "robot_mount_state",
    "robot_root_pose_local",
]
