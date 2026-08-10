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
from isaaclab.managers import ActionTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms

from zero_g_blade_swap.grapple_geometry import (
    EXTRACTED_BLADE_CENTRE_X,
    GRAPPLE_HEAD_ON_TOOL_ROT,
    GRAPPLE_PIN_GRIP_OFFSET,
    SLOT_MOUTH_X,
)

from .actions import (
    ROBOTIQ_2F85_COUPLING_SIGNS,
    ROBOTIQ_2F85_JOINT_NAMES,
    RobotiqBinaryAction,
    RobotiqBinaryActionCfg,
    robotiq_2f85_coupled_targets,
)
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


class TwoStageRobotiqAction(RobotiqBinaryAction):
    """Capture gently, then firm up once the grip is loaded.

    This is not a refinement, it is the difference between passing the gate and
    failing it. A wedge converts closing force into thrust along the pull axis,
    so a hard capture drives the payload away before it has been taken; but
    holding wants all the force the drive can produce. Measured axial capacity
    against a single command was 59 N; capturing at 0.48 rad and firming to
    0.68 once the grip is established gives 69 N, against a 66.4 N gate.

    The window is not wide, and it is asymmetric. Capturing at 0.44 gives 63 N
    and at 0.52 gives 68 N, but 0.56 collapses to 26 N. Bias low.
    See evidence/grapple_pin_capture_plateau.json.
    """

    cfg: TwoStageRobotiqActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Robotiq action must have shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}."
            )
        self._raw_actions.copy_(actions)
        closing = actions > self.cfg.threshold if self.cfg.close_on_positive else actions < self.cfg.threshold
        established = capture_established(self._env).unsqueeze(-1)
        target = torch.where(
            closing & established,
            torch.full_like(actions, self.cfg.hold_position),
            torch.where(
                closing,
                torch.full_like(actions, self.cfg.closed_position),
                torch.full_like(actions, self.cfg.open_position),
            ),
        )
        self._processed_actions.copy_(robotiq_2f85_coupled_targets(target))


@configclass
class TwoStageRobotiqActionCfg(RobotiqBinaryActionCfg):
    """Configuration for :class:`TwoStageRobotiqAction`."""

    class_type: type[ActionTerm] = TwoStageRobotiqAction
    #: Commanded once the grip is loaded. ``closed_position`` is the capture
    #: command, which is deliberately gentler.
    hold_position: float = 0.68


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


def _potential_reward(
    env, key: str, cost: torch.Tensor, scale: float, clamp: float, settle_s: float = 0.30
) -> torch.Tensor:
    """Pay for the measured reduction in a cost since the previous step.

    Potential-based, so a policy cannot farm it by oscillating: every metre
    gained is paid once and every metre lost is charged once.

    The validity flag is not optional. A reset moves the arm and the blade
    discontinuously, and without it the first step of every episode is charged
    the whole jump from the previous episode's final cost, which shows up as a
    large spurious reward of either sign. The insertion terms carry the same
    flag for the same reason.
    """

    previous = getattr(env, f"_grapple_prev_{key}", None)
    valid = getattr(env, f"_grapple_valid_{key}", None)
    reward = getattr(env, f"_grapple_reward_{key}", None)
    if previous is None or previous.shape[0] != env.num_envs:
        previous = torch.zeros_like(cost)
        valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        reward = torch.zeros_like(cost)
        setattr(env, f"_grapple_prev_{key}", previous)
        setattr(env, f"_grapple_valid_{key}", valid)
        setattr(env, f"_grapple_reward_{key}", reward)
    step_name = f"_grapple_step_{key}"
    current_step = int(env.common_step_counter)
    if getattr(env, step_name, -1) != current_step:
        # A reset writes joint positions but leaves the previous episode's
        # actuator targets in place, so the arm springs for a few steps before
        # it holds. Paying for that motion is paying for nothing the policy did.
        settled = env.episode_length_buf.to(torch.float32) * float(env.step_dt) >= settle_s
        reward.copy_(
            torch.where(valid & settled, ((previous - cost) / scale).clamp(-clamp, clamp) / float(env.step_dt), 0.0)
        )
        previous.copy_(cost)
        valid.fill_(True)
        setattr(env, step_name, current_step)
    return reward


def reset_grapple_progress(env, env_ids: torch.Tensor | None) -> None:
    """Clear every per-episode counter this module keeps on the environment."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    for name in ("_grapple_capture_hold", "_grapple_extract_hold"):
        value = getattr(env, name, None)
        if value is not None and value.shape[0] == env.num_envs:
            value[ids] = 0
    # Invalidate rather than zero the potentials: zeroing would charge the next
    # step the entire distance from the goal.
    for key in ("capture", "extract"):
        valid = getattr(env, f"_grapple_valid_{key}", None)
        if valid is not None and valid.shape[0] == env.num_envs:
            valid[ids] = False


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

    placed = _blade_reset_pose(env)
    if placed is None:
        return torch.zeros(env.num_envs, device=env.device)
    blade_position, blade_orientation = placed
    tool_position, tool_orientation = end_effector_pose_world(env)
    offset = blade_position.new_tensor(GRAPPLE_PIN_GRIP_OFFSET).expand(env.num_envs, -1)
    grip_position = blade_position + quat_apply(blade_orientation, offset)
    desired = quat_mul(
        blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation)
    )
    position = torch.linalg.vector_norm(tool_position - grip_position, dim=-1)
    orientation = torch.linalg.vector_norm(
        axis_angle_from_quat(quat_mul(desired, quat_inv(tool_orientation))), dim=-1
    )
    cost = position / max(distance_scale, 1.0e-6) + 0.5 * orientation / 0.20
    return _potential_reward(env, "capture", cost, scale=1.0, clamp=0.25)


def blade_disturbance_penalty(env, free_m: float = 0.005) -> torch.Tensor:
    """Penalize shoving the blade around while approaching it.

    In zero gravity a blade knocked off its rest pose does not come back, and a
    capture that has to chase a moving target is not a capture. The free band
    keeps ordinary settling unpenalized.
    """

    placed = _blade_reset_pose(env)
    if placed is None:
        return torch.zeros(env.num_envs, device=env.device)
    blade = env.scene["spare_blade"]
    displacement = torch.linalg.vector_norm(blade.data.root_pos_w - placed[0], dim=-1)
    return ((displacement - free_m) / 0.010).clamp_min(0.0).square().clamp(max=25.0)


def record_blade_reset_pose(env, env_ids: torch.Tensor | None) -> None:
    """Remember where the blade was placed.

    Two terms need this. Disturbance is measured against it, and so is the
    approach reward: scoring the approach against the *live* blade pays a policy
    for the blade drifting toward the tool, which in zero gravity it can cause
    by shoving it. Scoring against the placed pose means only the arm's own
    motion earns anything, and blade motion is charged separately.
    """

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    blade = env.scene["spare_blade"]
    pose = getattr(env, "_grapple_blade_reset_pose", None)
    if pose is None or pose.shape[0] != env.num_envs:
        pose = blade.data.root_state_w[:, :7].clone()
        env._grapple_blade_reset_pose = pose
    pose[ids] = blade.data.root_state_w[ids, :7]


def _blade_reset_pose(env) -> tuple[torch.Tensor, torch.Tensor] | None:
    pose = getattr(env, "_grapple_blade_reset_pose", None)
    if pose is None or pose.shape[0] != env.num_envs:
        return None
    return pose[:, :3], pose[:, 3:7]


def capture_failure(
    env,
    position_limit: float = 0.120,
    orientation_limit: float = 0.60,
    blade_displacement_limit: float = 0.060,
) -> torch.Tensor:
    """End a grasp episode that can no longer succeed."""

    position, orientation = grapple_grip_error_metrics(env)
    blade = env.scene["spare_blade"]
    placed = _blade_reset_pose(env)
    displaced = (
        torch.linalg.vector_norm(blade.data.root_pos_w - placed[0], dim=-1) > blade_displacement_limit
        if placed is not None
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


def extraction_remaining_observation(env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
    """The same distance as a column, because observation terms are 2-D."""

    return extraction_error(env, target_x).unsqueeze(-1)


def extraction_progress_reward(env, target_x: float = EXTRACTED_BLADE_CENTRE_X) -> torch.Tensor:
    """Pay for measured travel toward clear, not for holding position."""

    return _potential_reward(env, "extract", extraction_error(env, target_x), scale=0.030, clamp=1.0)


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


def hold_two_stage_grip(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    capture_positions: tuple[float, ...],
    hold_positions: tuple[float, ...],
) -> None:
    """Command the capture closure until the grip loads, then the holding one.

    Extract and insert start already captured, but "already captured" cannot be
    faked by writing the fingers to their seated angle: that places the pads
    around the wedge without pressing the pin against the collar, so the first
    pull travels the whole seating gap before anything takes load. Measured that
    way, a 40 mm pull moved the blade 0.1 mm and opened a 12.9 mm grip error.

    Letting the capture actually happen, inside the action term's settling
    window, produces the same preloaded state the pull gate measured 69 N on.
    """

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    capture = robot.data.joint_pos.new_tensor(capture_positions).expand(len(ids), -1)
    hold = robot.data.joint_pos.new_tensor(hold_positions).expand(len(ids), -1)
    established = capture_established(env)[ids].unsqueeze(-1)
    robot.set_joint_position_target(
        torch.where(established, hold, capture), joint_ids=asset_cfg.joint_ids, env_ids=ids
    )


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
    "TwoStageRobotiqAction",
    "TwoStageRobotiqActionCfg",
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
    "extraction_remaining_observation",
    "extraction_success_mask",
    "extraction_success_reward",
    "grapple_grip_error_metrics",
    "grapple_grip_error_observation",
    "grapple_grip_pose_error",
    "grip_drive_torque",
    "grip_retention_penalty",
    "gripper_state_observation",
    "hold_two_stage_grip",
    "record_blade_reset_pose",
    "reset_grapple_blade_pose",
    "reset_grapple_fingers",
    "reset_grapple_progress",
]
