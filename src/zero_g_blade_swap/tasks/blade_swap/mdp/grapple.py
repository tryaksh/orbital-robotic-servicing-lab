"""MDP terms for the three head-on grapple-pin skills.

Grasp, extract, and insert are separate skills with separate gates, so they get
separate terms rather than one shared reward with mode flags. What they have in
common is the frame they are scored in: the tool frame sits at the centre of the
measured 105-to-162 mm pad span, and the blade carries a matching point at the
centre of the length of pin the pads close on.

Two things here differ from the insertion terms in ``insertion.py`` and are easy
to get wrong.

*The blade is not welded to the tool.* Insertion's `tool_to_handle_error_m` is a
tautology on the rigid-grasp task, because a fixed joint holds the blade at the
frame the metric compares against. Here nothing holds the blade but contact, so
the same quantity is a real measurement of whether the grip is still there.

*The workspace predicate is inverted for extraction.* `insertion_failure` treats
a blade at x below 0.45 as an escape, which is exactly where a successful
extraction ends. Extraction therefore carries its own bounds.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms

from zero_g_blade_swap.grapple_geometry import (
    EXTRACTED_BLADE_CENTRE_X,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_PIN_GRIP_OFFSET,
    SLOT_MOUTH_X,
)

from .actions import ROBOTIQ_2F85_COUPLING_SIGNS, ROBOTIQ_2F85_JOINT_NAMES
from .insertion import attached_blade_pose_world, attached_blade_velocity
from .observations import end_effector_pose_world

# A grasp counts as formed only when the drive torque rises off its noise floor,
# which sits at 1e-5 N-m. This threshold is the same one grasp_diagnostics.py
# gates on, so the trained skill and the physics characterisation agree on what
# the word means.
GRIP_TORQUE_THRESHOLD_NM = 0.05


def grapple_grip_pose_error(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tool-to-grip-point vector in world axes and the orientation error.

    The orientation error is measured against the head-on capture attitude
    rather than against the blade, because the pin's axis is the blade's own
    x axis and the gripper has to arrive along it.
    """

    tool_position, tool_orientation = end_effector_pose_world(env)
    blade_position, blade_orientation = attached_blade_pose_world(env)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    # The capture attitude is expressed relative to the blade, so a blade that
    # has been knocked askew moves the target rather than hiding the error.
    desired = quat_mul(blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation))
    angle = torch.linalg.vector_norm(axis_angle_from_quat(quat_mul(desired, quat_inv(tool_orientation))), dim=-1)
    return tool_position - grip_position, angle


def grapple_grip_error_metrics(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tool-to-grip distance and capture-attitude error magnitudes."""

    vector, angle = grapple_grip_pose_error(env)
    return torch.linalg.vector_norm(vector, dim=-1), angle


def grapple_grip_error_observation(env) -> torch.Tensor:
    """Six values: the tool-to-grip offset in tool axes, and the attitude error."""

    tool_position, tool_orientation = end_effector_pose_world(env)
    blade_position, blade_orientation = attached_blade_pose_world(env)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    desired = quat_mul(blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation))
    relative_position, relative_quat = subtract_frame_transforms(
        tool_position, tool_orientation, grip_position, desired
    )
    return torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)


def _finger_joint_ids(env) -> list[int]:
    ids = getattr(env, "_grapple_finger_joint_ids", None)
    if ids is None:
        ids, names = env.scene["robot"].find_joints(list(ROBOTIQ_2F85_JOINT_NAMES), preserve_order=True)
        if tuple(names) != ROBOTIQ_2F85_JOINT_NAMES:
            raise RuntimeError(f"unexpected Robotiq joint order: {names}")
        env._grapple_finger_joint_ids = ids
    return ids


def grip_drive_torque(env) -> torch.Tensor:
    """Return the 2F-85 drive torque, the signal that a grip is actually loaded."""

    robot = env.scene["robot"]
    return robot.data.applied_torque[:, _finger_joint_ids(env)[0]].abs()


def gripper_state_observation(env) -> torch.Tensor:
    """Return the finger command's reached angle and the drive torque it develops.

    Both are needed: the angle alone cannot distinguish fingers closed on a pin
    from fingers closed on nothing, which is the failure this project spent
    three sessions not seeing.
    """

    robot = env.scene["robot"]
    joint_ids = _finger_joint_ids(env)
    angle = robot.data.joint_pos[:, joint_ids[0]].unsqueeze(-1)
    torque = grip_drive_torque(env).unsqueeze(-1)
    return torch.cat((angle / 0.8203, torque / 10.0), dim=-1)


def capture_established(
    env,
    torque_threshold: float = GRIP_TORQUE_THRESHOLD_NM,
    position_tolerance: float = 0.020,
    orientation_tolerance: float = 0.20,
) -> torch.Tensor:
    """True where the pads are loaded against the pin at the capture attitude."""

    position, orientation = grapple_grip_error_metrics(env)
    return (
        (grip_drive_torque(env) > torque_threshold)
        & (position <= position_tolerance)
        & (orientation <= orientation_tolerance)
    )


def _hold_counter(env, name: str, active: torch.Tensor, hold_time_s: float) -> torch.Tensor:
    """Count consecutive steps a condition has held, once per environment step."""

    counter = getattr(env, name, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, name, counter)
    step_name = f"{name}_step"
    current_step = int(env.common_step_counter)
    if getattr(env, step_name, -1) != current_step:
        counter.copy_(torch.where(active, counter + 1, torch.zeros_like(counter)))
        setattr(env, step_name, current_step)
    return counter >= max(1, int(round(hold_time_s / float(env.step_dt))))


def reset_grapple_progress(env, env_ids: torch.Tensor | None) -> None:
    """Clear every per-episode counter this module keeps on the environment."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    for name in ("_grapple_capture_hold", "_grapple_extract_hold", "_grapple_previous_cost"):
        value = getattr(env, name, None)
        if value is not None and value.shape[0] == env.num_envs:
            value[ids] = 0


def capture_success_mask(env, hold_time_s: float = 0.30) -> torch.Tensor:
    """Terminate a grasp episode once a loaded capture has been held."""

    return _hold_counter(env, "_grapple_capture_hold", capture_established(env), hold_time_s)


def capture_success_reward(env, hold_time_s: float = 0.30) -> torch.Tensor:
    return capture_success_mask(env, hold_time_s).to(torch.float32) / float(env.step_dt)


def capture_approach_reward(env, distance_scale: float = 0.050) -> torch.Tensor:
    """Pay for closing on the grip point; standing still earns nothing.

    Potential-based, so a policy cannot farm this by oscillating: the reward is
    the measured reduction in a pose cost since the previous step.
    """

    position, orientation = grapple_grip_error_metrics(env)
    cost = position / max(distance_scale, 1.0e-6) + 0.5 * orientation / 0.20
    previous = getattr(env, "_grapple_previous_cost", None)
    if previous is None or previous.shape[0] != env.num_envs:
        previous = cost.detach().clone()
        env._grapple_previous_cost = previous
        return torch.zeros_like(cost)
    current_step = int(env.common_step_counter)
    reward = getattr(env, "_grapple_approach_reward", None)
    if reward is None or reward.shape[0] != env.num_envs:
        reward = torch.zeros_like(cost)
        env._grapple_approach_reward = reward
    if getattr(env, "_grapple_approach_step", -1) != current_step:
        reward.copy_((previous - cost).clamp(-0.25, 0.25) / float(env.step_dt))
        previous.copy_(cost)
        env._grapple_approach_step = current_step
    return reward


def blade_disturbance_penalty(env, free_m: float = 0.005) -> torch.Tensor:
    """Penalize shoving the blade around while approaching it.

    In zero gravity a blade knocked off its rest pose does not come back, and a
    capture that has to chase a moving target is not a capture. The free band
    keeps ordinary settling unpenalized.
    """

    blade = env.scene["spare_blade"]
    start = getattr(env, "_grapple_blade_reset_pos", None)
    if start is None or start.shape[0] != env.num_envs:
        return torch.zeros(env.num_envs, device=env.device)
    displacement = torch.linalg.vector_norm(blade.data.root_pos_w - start, dim=-1)
    return ((displacement - free_m) / 0.010).clamp_min(0.0).square().clamp(max=25.0)


def record_blade_reset_pose(env, env_ids: torch.Tensor | None) -> None:
    """Remember where the blade started, so disturbance can be measured."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    blade = env.scene["spare_blade"]
    start = getattr(env, "_grapple_blade_reset_pos", None)
    if start is None or start.shape[0] != env.num_envs:
        start = blade.data.root_pos_w.clone()
        env._grapple_blade_reset_pos = start
    start[ids] = blade.data.root_pos_w[ids]


def capture_failure(
    env,
    position_limit: float = 0.120,
    orientation_limit: float = 0.60,
    blade_displacement_limit: float = 0.060,
) -> torch.Tensor:
    """End a grasp episode that can no longer succeed."""

    position, orientation = grapple_grip_error_metrics(env)
    blade = env.scene["spare_blade"]
    start = getattr(env, "_grapple_blade_reset_pos", None)
    displaced = (
        torch.linalg.vector_norm(blade.data.root_pos_w - start, dim=-1) > blade_displacement_limit
        if start is not None and start.shape[0] == env.num_envs
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    conditions = {
        "tool_lost": position > position_limit,
        "attitude_lost": orientation > orientation_limit,
        "blade_shoved": displaced,
    }
    env._grapple_latest_failure_conditions = conditions
    return torch.stack(tuple(conditions.values()), dim=-1).any(dim=-1)


def capture_failure_reward(env) -> torch.Tensor:
    return env.termination_manager.get_term("capture_failed").to(torch.float32) / float(env.step_dt)


def blade_centre_x(env) -> torch.Tensor:
    """Blade centre along the extraction axis, in the environment frame."""

    position, _ = attached_blade_pose_world(env)
    return position[:, 0] - env.scene.env_origins[:, 0]


def extraction_error(env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
    """How much further the blade still has to travel to clear the slot."""

    return (blade_centre_x(env) - target_x).clamp_min(0.0)


def extraction_progress_reward(env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
    """Pay for measured travel toward clear, not for holding position."""

    remaining = extraction_error(env, target_x)
    previous = getattr(env, "_grapple_previous_cost", None)
    if previous is None or previous.shape[0] != env.num_envs:
        previous = remaining.detach().clone()
        env._grapple_previous_cost = previous
        return torch.zeros_like(remaining)
    reward = getattr(env, "_grapple_approach_reward", None)
    if reward is None or reward.shape[0] != env.num_envs:
        reward = torch.zeros_like(remaining)
        env._grapple_approach_reward = reward
    current_step = int(env.common_step_counter)
    if getattr(env, "_grapple_approach_step", -1) != current_step:
        reward.copy_(((previous - remaining) / 0.030).clamp(-1.0, 1.0) / float(env.step_dt))
        previous.copy_(remaining)
        env._grapple_approach_step = current_step
    return reward


def extraction_success_mask(
    env,
    target_x: float = EXTRACTED_BLADE_CENTRE_X,
    hold_time_s: float = 0.20,
    linear_velocity_limit: float = 0.10,
    angular_velocity_limit: float = 0.30,
) -> torch.Tensor:
    """Clear of the slot, still gripped, and no longer moving fast."""

    velocity = attached_blade_velocity(env)
    active = (
        (blade_centre_x(env) <= target_x)
        & capture_established(env)
        & (torch.linalg.vector_norm(velocity[:, :3], dim=-1) <= linear_velocity_limit)
        & (torch.linalg.vector_norm(velocity[:, 3:], dim=-1) <= angular_velocity_limit)
    )
    return _hold_counter(env, "_grapple_extract_hold", active, hold_time_s)


def extraction_success_reward(env) -> torch.Tensor:
    return extraction_success_mask(env).to(torch.float32) / float(env.step_dt)


def extraction_failure(
    env,
    grip_position_limit: float = 0.030,
    grip_orientation_limit: float = 0.35,
) -> torch.Tensor:
    """End an extraction that has dropped the blade or left the workspace.

    The workspace bound along x is deliberately different from the insertion
    task's, which treats anything below 0.45 as an escape. That is where a
    successful extraction finishes.
    """

    position, orientation = grapple_grip_error_metrics(env)
    blade_position, _ = attached_blade_pose_world(env)
    local = blade_position - env.scene.env_origins
    conditions = {
        "grip_lost": position > grip_position_limit,
        "grip_attitude_lost": orientation > grip_orientation_limit,
        "workspace_x": (local[:, 0] < -0.10) | (local[:, 0] > 1.10),
        "workspace_yz": (local[:, 1].abs() > 0.30) | (local[:, 2] < 0.40) | (local[:, 2] > 1.10),
    }
    env._grapple_latest_failure_conditions = conditions
    return torch.stack(tuple(conditions.values()), dim=-1).any(dim=-1)


def extraction_failure_reward(env) -> torch.Tensor:
    return env.termination_manager.get_term("extraction_failed").to(torch.float32) / float(env.step_dt)


def grip_retention_penalty(env, free_m: float = 0.004, free_rad: float = 0.08) -> torch.Tensor:
    """Penalize the grip drifting, without paying a policy to stand still."""

    position, orientation = grapple_grip_error_metrics(env)
    position_excess = ((position - free_m) / 0.010).clamp_min(0.0)
    orientation_excess = ((orientation - free_rad) / 0.15).clamp_min(0.0)
    return (position_excess.square() + 0.25 * orientation_excess.square()).clamp(max=25.0)


def reset_grapple_blade_pose(
    env,
    env_ids: torch.Tensor | None,
    pose: tuple[float, ...],
    position_noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("spare_blade"),
) -> None:
    """Place the blade at one fixed pose, with optional isotropic position noise."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    blade = env.scene[asset_cfg.name]
    target = blade.data.root_state_w.new_tensor(pose).expand(len(ids), -1).clone()
    if position_noise > 0.0:
        target[:, :3] += (2.0 * torch.rand((len(ids), 3), device=env.device) - 1.0) * position_noise
    target[:, :3] += env.scene.env_origins[ids]
    blade.write_root_pose_to_sim(target, env_ids=ids)
    blade.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


def reset_grapple_fingers(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    finger_joint: float,
) -> None:
    """Write the finger joints to one command and hold them there."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    signs = robot.data.joint_pos.new_tensor(ROBOTIQ_2F85_COUPLING_SIGNS)
    targets = (signs * finger_joint).expand(len(ids), -1).clone()
    robot.write_joint_state_to_sim(targets, torch.zeros_like(targets), joint_ids=asset_cfg.joint_ids, env_ids=ids)
    robot.set_joint_position_target(targets, joint_ids=asset_cfg.joint_ids, env_ids=ids)


__all__ = [
    "EXTRACTED_BLADE_CENTRE_X",
    "GRIP_TORQUE_THRESHOLD_NM",
    "SLOT_MOUTH_X",
    "blade_centre_x",
    "blade_disturbance_penalty",
    "capture_approach_reward",
    "capture_established",
    "capture_failure",
    "capture_failure_reward",
    "capture_success_mask",
    "capture_success_reward",
    "extraction_error",
    "extraction_failure",
    "extraction_failure_reward",
    "extraction_progress_reward",
    "extraction_success_mask",
    "extraction_success_reward",
    "grapple_grip_error_metrics",
    "grapple_grip_error_observation",
    "grapple_grip_pose_error",
    "grip_drive_torque",
    "grip_retention_penalty",
    "gripper_state_observation",
    "record_blade_reset_pose",
    "reset_grapple_blade_pose",
    "reset_grapple_fingers",
    "reset_grapple_progress",
]
