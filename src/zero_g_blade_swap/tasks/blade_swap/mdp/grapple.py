"""Head-on grapple-pin capture: the terms the measured interface still needs.

This module once carried three hand-rolled skills (grasp, extract, insert).
They were deleted on 2026-08-10 along with their tasks, because all three failed
for reasons established work already solves and the project pivoted to
force-aware insertion under pose uncertainty. What survives is the machinery the
*interface* result depends on, which is measured and published in
``docs/service_interface_spec.md``:

*Capture and hold are two different commands.* A wedge converts closing force
into thrust along the pull axis, so a firm capture drives the payload away
before it has been taken, while holding wants everything the drive can produce.
Measured axial capacity against one command throughout is 59 N; capturing at
0.48 rad and firming to 0.68 once the grip is loaded holds 69 N against a 66.4 N
requirement. :class:`TwoStageRobotiqAction` and :func:`hold_two_stage_grip`
implement that split inside the environment rather than leaving it to a script.

*The blade is not welded to the tool here.* Insertion's ``tool_to_handle_error_m``
is a tautology on the rigid-grasp task, because a fixed joint holds the blade at
the frame the metric compares against. On this scene nothing holds the blade but
contact, so :func:`grapple_grip_error_metrics` is a real measurement of whether
the grip is still there.

Frame warning, because this has cost days: these terms are expressed in the
head-on convention, a quarter turn from the top-down ``GRIPPER_GRASP_ROT`` that
``insertion.py``'s grasp metrics use. Reading one with the other reports about
2.1 rad of error on a perfectly good grip.
"""

from __future__ import annotations

import torch
from isaaclab.managers import ActionTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms

from zero_g_blade_swap.grapple_geometry import GRAPPLE_HEAD_ON_TOOL_ROT, GRAPPLE_PIN_GRIP_OFFSET

from .actions import (
    ROBOTIQ_2F85_COUPLING_SIGNS,
    ROBOTIQ_2F85_JOINT_NAMES,
    RobotiqBinaryAction,
    RobotiqBinaryActionCfg,
    robotiq_2f85_coupled_targets,
)
from .insertion import attached_blade_pose_world
from .observations import end_effector_pose_world

# A grasp counts as formed only when the drive torque rises off its noise floor,
# which sits at 1e-5 N-m. This threshold is the same one grasp_diagnostics.py
# gates on, so the action term and the physics characterisation agree on what
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
    desired = quat_mul(
        blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation)
    )
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
    desired = quat_mul(
        blade_orientation, blade_orientation.new_tensor(GRAPPLE_HEAD_ON_TOOL_ROT).expand_as(blade_orientation)
    )
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
    from fingers closed on nothing, which is exactly the failure this project did
    not see for three sessions.
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
    """True where the pads are loaded against the pin at the capture attitude.

    This is the trigger for the capture-to-hold transition, not a success
    predicate; the skills that scored themselves on it are gone.
    """

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


def hold_two_stage_grip(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    capture_positions: tuple[float, ...],
    hold_positions: tuple[float, ...],
) -> None:
    """Command the capture closure until the grip loads, then the holding one.

    A task that starts "already captured" cannot fake it by writing the fingers
    to their seated angle: that places the pads around the wedge without
    pressing the pin against the collar, so the first pull travels the whole
    seating gap before anything takes load. Measured that way, a 40 mm pull moved
    the blade 0.1 mm and opened a 12.9 mm grip error.

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
    "GRIP_TORQUE_THRESHOLD_NM",
    "TwoStageRobotiqAction",
    "TwoStageRobotiqActionCfg",
    "capture_established",
    "grapple_grip_error_metrics",
    "grapple_grip_error_observation",
    "grapple_grip_pose_error",
    "grip_drive_torque",
    "gripper_state_observation",
    "hold_two_stage_grip",
    "reset_grapple_fingers",
]
