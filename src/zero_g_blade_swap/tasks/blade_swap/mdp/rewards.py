"""Dense shaping and milestone rewards for the eight-phase swap."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_error_magnitude,
    subtract_frame_transforms,
)

from ..assets import BLADE_HANDLE_OFFSET, BLADE_INSERTED_POS, SPARE_BLADE_POS
from .commands import (
    FAILED_EXTRACTION_X,
    PHASE_ACQUIRE_SPARE,
    PHASE_EXTRACT_FAILED,
    SPARE_EXTRACTION_X,
    gripper_handle_orientation_error,
    insertion_mask,
    rack_goal_pose_local,
    swap_complete_mask,
    update_swap_phase,
)
from .observations import end_effector_pose_world


def blade_grasp_pose_world(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    blade = env.scene[asset_cfg.name]
    offset = torch.tensor(BLADE_HANDLE_OFFSET, device=env.device).expand(env.num_envs, -1)
    return combine_frame_transforms(blade.data.root_pos_w, blade.data.root_quat_w, offset)


def _active_grasp_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    phase = update_swap_phase(env)
    failed_pos, failed_rot = blade_grasp_pose_world(env, SceneEntityCfg("blade"))
    spare_pos, spare_rot = blade_grasp_pose_world(env, SceneEntityCfg("spare_blade"))
    use_failed = (phase < PHASE_ACQUIRE_SPARE).unsqueeze(-1)
    return torch.where(use_failed, failed_pos, spare_pos), torch.where(use_failed, failed_rot, spare_rot)


def distance_to_blade(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
    blade_cfg: SceneEntityCfg | None = None,
    std: float = 0.18,
) -> torch.Tensor:
    del blade_cfg
    ee_pos, _ = end_effector_pose_world(env, robot_cfg)
    grasp_pos, _ = _active_grasp_pose(env)
    return torch.exp(-torch.linalg.vector_norm(ee_pos - grasp_pos, dim=-1) / std)


def alignment_error(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
    blade_cfg: SceneEntityCfg | None = None,
    proximity_std: float = 0.25,
) -> torch.Tensor:
    del blade_cfg
    ee_pos, ee_rot = end_effector_pose_world(env, robot_cfg)
    grasp_pos, grasp_rot = _active_grasp_pose(env)
    proximity = torch.exp(-torch.linalg.vector_norm(ee_pos - grasp_pos, dim=-1) / proximity_std)
    return gripper_handle_orientation_error(ee_rot, grasp_rot) * proximity


def alignment_reward(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
    position_std: float = 0.10,
    orientation_std: float = 0.25,
) -> torch.Tensor:
    """Reward simultaneous handle translation and orientation alignment."""

    ee_pos, ee_rot = end_effector_pose_world(env, robot_cfg)
    grasp_pos, grasp_rot = _active_grasp_pose(env)
    position_score = torch.exp(-torch.linalg.vector_norm(ee_pos - grasp_pos, dim=-1) / position_std)
    orientation_score = torch.exp(-gripper_handle_orientation_error(ee_rot, grasp_rot) / orientation_std)
    return position_score * orientation_score


def lateral_orientation_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["wrist_3_link"]),
) -> torch.Tensor:
    """Explicit near-target lateral and angular misalignment.

    Gating by proximity prevents a distant initial pose from accumulating a
    large negative return that makes deliberate early termination attractive.
    """

    ee_pos, ee_rot = end_effector_pose_world(env, robot_cfg)
    grasp_pos, grasp_rot = _active_grasp_pose(env)
    lateral = torch.linalg.vector_norm((ee_pos - grasp_pos)[:, 1:3], dim=-1) / 0.05
    orientation = gripper_handle_orientation_error(ee_rot, grasp_rot) / 0.35
    proximity = torch.exp(-torch.linalg.vector_norm(ee_pos - grasp_pos, dim=-1) / 0.30)
    return (lateral + orientation) * proximity


def extraction_progress(env, blade_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    del blade_cfg
    phase = update_swap_phase(env)
    failed_x = env.scene["blade"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    spare_x = env.scene["spare_blade"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    failed = ((BLADE_INSERTED_POS[0] - failed_x) / (BLADE_INSERTED_POS[0] - FAILED_EXTRACTION_X)).clamp(0, 1)
    spare = ((SPARE_BLADE_POS[0] - spare_x) / (SPARE_BLADE_POS[0] - SPARE_EXTRACTION_X)).clamp(0, 1)
    return torch.where(phase == PHASE_EXTRACT_FAILED, failed, torch.where(phase == PHASE_ACQUIRE_SPARE, spare, 0.0))


def phase_completion_reward(env) -> torch.Tensor:
    """One-shot reward for each newly completed phase."""

    phase = update_swap_phase(env)
    previous = getattr(env, "_last_rewarded_phase", None)
    if previous is None or previous.shape[0] != env.num_envs:
        previous = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._last_rewarded_phase = previous
    delta = (phase - previous).clamp(min=0).to(torch.float32)
    previous.copy_(torch.maximum(previous, phase))
    return delta


def stable_grasp_reward(env) -> torch.Tensor:
    """Reward a closed gripper that remains geometrically coupled to the active handle."""

    ee_pos, _ = end_effector_pose_world(env)
    grasp_pos, _ = _active_grasp_pose(env)
    gripper_closed = env.action_manager.get_term("gripper").raw_actions[:, 0] > 0.0
    phase = update_swap_phase(env)
    grasp_phase = (phase >= PHASE_EXTRACT_FAILED) & (phase <= 6)
    return (gripper_closed & grasp_phase & (torch.linalg.vector_norm(ee_pos - grasp_pos, dim=-1) <= 0.065)).to(
        torch.float32
    )


def extraction_complete(env) -> torch.Tensor:
    return phase_completion_reward(env)


def insertion_progress(
    env,
    blade_cfg: SceneEntityCfg = SceneEntityCfg("spare_blade"),
    command_name: str = "blade_goal",
    position_std: float = 0.15,
    rotation_std: float = 0.35,
) -> torch.Tensor:
    phase = update_swap_phase(env, command_name)
    blade = env.scene[blade_cfg.name]
    goal = rack_goal_pose_local(env, command_name)
    local_pos = blade.data.root_pos_w - env.scene.env_origins
    score = torch.exp(-torch.linalg.vector_norm(local_pos - goal[:, :3], dim=-1) / position_std)
    score *= torch.exp(-quat_error_magnitude(blade.data.root_quat_w, goal[:, 3:7]) / rotation_std)
    return score * (phase >= 5).to(torch.float32)


def insertion_success(env, command_name: str = "blade_goal") -> torch.Tensor:
    return insertion_mask(env, command_name).to(torch.float32)


def insertion_completion_reward(env, command_name: str = "blade_goal") -> torch.Tensor:
    """One-time reward when the replacement first reaches the seated phase."""

    inserted = insertion_mask(env, command_name)
    rewarded = getattr(env, "_insertion_rewarded", None)
    if rewarded is None or rewarded.shape[0] != env.num_envs:
        rewarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._insertion_rewarded = rewarded
    newly_inserted = inserted & ~rewarded
    rewarded.logical_or_(inserted)
    return newly_inserted.to(torch.float32)


def full_swap_reward(env, command_name: str = "blade_goal") -> torch.Tensor:
    return swap_complete_mask(env, command_name).to(torch.float32)


def blade_motion_penalty(env, blade_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    del blade_cfg
    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in ("blade", "spare_blade"):
        blade = env.scene[name]
        penalty += torch.sum(torch.square(blade.data.root_lin_vel_w), dim=-1)
        penalty += 0.1 * torch.sum(torch.square(blade.data.root_ang_vel_w), dim=-1)
    return penalty


def mount_deflection_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("mount_anchor"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    anchor = env.scene[anchor_cfg.name]
    relative_pos, relative_quat = subtract_frame_transforms(
        anchor.data.root_pos_w,
        anchor.data.root_quat_w,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )
    relative_rotation = axis_angle_from_quat(relative_quat)
    return torch.sum(torch.square(relative_pos), dim=-1) + torch.sum(torch.square(relative_rotation), dim=-1)


def undesired_termination_penalty(env, term_keys: tuple[str, ...]) -> torch.Tensor:
    """Return a one-time, timestep-independent penalty for unsafe endings."""

    failures = torch.zeros(env.num_envs, device=env.device)
    for term_name in term_keys:
        failures += env.termination_manager.get_term(term_name).to(torch.float32)
    return failures / float(env.step_dt)


def workspace_violation_penalty(env) -> torch.Tensor:
    """Soft penalty before either blade reaches the hard escape termination."""

    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in ("blade", "spare_blade"):
        position = env.scene[name].data.root_pos_w - env.scene.env_origins
        penalty += torch.relu(position[:, 0].abs() - 1.10)
        penalty += torch.relu(position[:, 1].abs() - 0.60)
        penalty += torch.relu(0.30 - position[:, 2]) + torch.relu(position[:, 2] - 1.25)
    return penalty


__all__ = [
    "alignment_error",
    "alignment_reward",
    "blade_grasp_pose_world",
    "blade_motion_penalty",
    "distance_to_blade",
    "extraction_complete",
    "extraction_progress",
    "insertion_progress",
    "insertion_completion_reward",
    "insertion_success",
    "full_swap_reward",
    "lateral_orientation_penalty",
    "mount_deflection_penalty",
    "undesired_termination_penalty",
    "phase_completion_reward",
    "stable_grasp_reward",
    "workspace_violation_penalty",
]
