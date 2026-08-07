"""MDP terms for the first learned skill: state-based blade insertion."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from isaaclab.managers import (
    CommandTerm,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    compute_pose_error,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from zero_g_blade_swap.math_utils import insertion_curriculum_probabilities, update_curriculum_stage

from ..assets import (
    BLADE_HANDLE_OFFSET,
    BLADE_INSERTED_POS,
    GRIPPER_GRASP_ROT,
    INSERTION_STAGE_ARM_JOINT_POS,
    INSERTION_STAGE_BLADE_POSE,
)
from .commands import gripper_handle_orientation_error
from .observations import end_effector_pose_world
from .randomization import rail_stiction_resistance

INSERTION_SUCCESS_CONDITION_NAMES = (
    "axial_depth",
    "lateral_alignment",
    "orientation",
    "linear_velocity",
    "angular_velocity",
    "grasp_position",
    "grasp_orientation",
    "hold_duration",
)


def attached_blade_pose_world(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the dynamic blade pose constrained to the physical tool frame."""

    blade = env.scene["spare_blade"]
    return blade.data.root_pos_w, blade.data.root_quat_w


class InsertionGoalCommand(CommandTerm):
    """Fixed rack goal kept as a command so later stages can randomize it cleanly."""

    cfg: InsertionGoalCommandCfg

    def __init__(self, cfg: InsertionGoalCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 7), device=self.device)
        self._command[:, :3] = torch.tensor(cfg.goal_pos, device=self.device)
        self._command[:, 3:7] = torch.tensor(cfg.goal_rot, device=self.device)
        self.metrics["blade_position_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        self._command[env_ids, :3] = self._command.new_tensor(self.cfg.goal_pos)
        self._command[env_ids, 3:7] = self._command.new_tensor(self.cfg.goal_rot)

    def _update_command(self) -> None:
        return

    def _update_metrics(self) -> None:
        blade_position, _ = attached_blade_pose_world(self._env)
        local_position = blade_position - self._env.scene.env_origins
        self.metrics["blade_position_error"][:] = torch.linalg.vector_norm(
            local_position - self._command[:, :3], dim=-1
        )


@configclass
class InsertionGoalCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = InsertionGoalCommand
    resampling_time_range: tuple[float, float] = (12.0, 12.0)
    goal_pos: tuple[float, float, float] = BLADE_INSERTED_POS
    goal_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _goal(env, command_name: str = "insertion_goal") -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _errors(env, command_name: str = "insertion_goal") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blade_position, blade_orientation = attached_blade_pose_world(env)
    goal = _goal(env, command_name)
    local_position = blade_position - env.scene.env_origins
    delta = local_position - goal[:, :3]
    return (
        delta[:, 0].abs(),
        torch.linalg.vector_norm(delta[:, 1:3], dim=-1),
        quat_error_magnitude(blade_orientation, goal[:, 3:7]),
    )


def insertion_goal_error(env, command_name: str = "insertion_goal") -> torch.Tensor:
    """Goal minus blade pose as three metres and three rotation-vector components."""

    blade_position, blade_orientation = attached_blade_pose_world(env)
    goal = _goal(env, command_name)
    local_position = blade_position - env.scene.env_origins
    relative_position, relative_quat = subtract_frame_transforms(
        local_position,
        blade_orientation,
        goal[:, :3],
        goal[:, 3:7],
    )
    return torch.cat((relative_position, axis_angle_from_quat(relative_quat)), dim=-1)


def insertion_error_metrics(
    env, command_name: str = "insertion_goal"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return axial, lateral, and angular errors for evaluation reporting."""

    return _errors(env, command_name)


def secured_blade_pose_error(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the tool-to-handle position vector and orientation error."""

    tcp_position, tcp_orientation = end_effector_pose_world(env)
    blade_position, blade_orientation = attached_blade_pose_world(env)
    handle_offset_cfg = getattr(env.cfg.scene.spare_blade.spawn, "handle_offset", BLADE_HANDLE_OFFSET)
    handle_offset = blade_position.new_tensor(handle_offset_cfg).expand(env.num_envs, -1)
    handle_position = blade_position + quat_apply(blade_orientation, handle_offset)
    return tcp_position - handle_position, gripper_handle_orientation_error(tcp_orientation, blade_orientation)


def secured_blade_error_metrics(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tool-to-handle position and orientation magnitudes.

    These metrics make the Stage-1 grasp abstraction auditable: insertion is
    not counted as success if the compliant fixture lets the blade visibly
    separate from the gripper.
    """

    position_error, orientation_error = secured_blade_pose_error(env)
    return torch.linalg.vector_norm(position_error, dim=-1), orientation_error


def attached_blade_velocity(env) -> torch.Tensor:
    blade = env.scene["spare_blade"]
    return torch.cat((blade.data.root_lin_vel_w, blade.data.root_ang_vel_w), dim=-1)


class SecuredBladeConstraint(ManagerTermBase):
    """Compliant 6D grasp fixture; PPO still controls every robot motion."""

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        body_ids, body_names = env.scene["robot"].find_bodies(["wrist_3_link"], preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(f"Secured blade constraint expected wrist_3_link, resolved {body_names}")
        self._wrist_body_id = int(body_ids[0])
        self._mass_scaled_position_damping: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        ids = torch.arange(self._env.num_envs, device=self._env.device) if env_ids is None else env_ids
        zeros = torch.zeros((len(ids), 1, 3), device=self._env.device)
        self._env.scene["spare_blade"].permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            env_ids=ids,
            is_global=True,
        )

    @staticmethod
    def _clamp_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
        scale = (maximum / torch.linalg.vector_norm(vector, dim=-1).clamp_min(1.0e-6)).clamp(max=1.0)
        return vector * scale.unsqueeze(-1)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        position_stiffness: float,
        position_damping: float,
        rotation_stiffness: float,
        rotation_damping: float,
        maximum_force: float,
        maximum_torque: float,
        position_damping_ratio: float | None = None,
    ) -> None:
        del env_ids
        tcp_position, tcp_orientation = end_effector_pose_world(env)
        grasp_rotation = tcp_orientation.new_tensor(GRIPPER_GRASP_ROT).expand_as(tcp_orientation)
        desired_orientation = quat_mul(tcp_orientation, quat_inv(grasp_rotation))
        handle_offset = tcp_position.new_tensor(BLADE_HANDLE_OFFSET).expand(env.num_envs, -1)
        desired_position = tcp_position - quat_apply(desired_orientation, handle_offset)
        blade = env.scene["spare_blade"]
        position_error, rotation_error = compute_pose_error(
            blade.data.root_pos_w,
            blade.data.root_quat_w,
            desired_position,
            desired_orientation,
            rot_error_type="axis_angle",
        )
        # Damping belongs between the gripper and blade, not between the blade
        # and the stationary world.  World-relative damping made the fixture
        # oppose commanded insertion and caused light blades to lag/oscillate.
        robot = env.scene["robot"]
        wrist_position = robot.data.body_pos_w[:, self._wrist_body_id]
        wrist_linear_velocity = robot.data.body_lin_vel_w[:, self._wrist_body_id]
        wrist_angular_velocity = robot.data.body_ang_vel_w[:, self._wrist_body_id]
        wrist_to_blade = desired_position - wrist_position
        desired_linear_velocity = wrist_linear_velocity + torch.cross(
            wrist_angular_velocity,
            wrist_to_blade,
            dim=-1,
        )
        effective_position_damping: float | torch.Tensor = position_damping
        if position_damping_ratio is not None:
            if not 0.0 < position_damping_ratio <= 2.0:
                raise ValueError(f"position_damping_ratio must be in (0, 2], got {position_damping_ratio}")
            if self._mass_scaled_position_damping is None:
                masses = blade.root_physx_view.get_masses()[:, 0].to(device=env.device)
                self._mass_scaled_position_damping = (
                    2.0 * position_damping_ratio * torch.sqrt(position_stiffness * masses)
                ).unsqueeze(-1)
            effective_position_damping = self._mass_scaled_position_damping
        force = position_stiffness * position_error + effective_position_damping * (
            desired_linear_velocity - blade.data.root_lin_vel_w
        )
        # Phase 2 uses the same permanent-wrench channel for the compliant
        # grasp and rail resistance.  Compose the forces here so one manager
        # term cannot silently overwrite the other.
        ids = torch.arange(env.num_envs, device=env.device)
        force[:, 0] += rail_stiction_resistance(
            env,
            SceneEntityCfg("spare_blade"),
            ids,
            rail_center_y=0.0,
            rail_center_z=BLADE_INSERTED_POS[2],
            engagement_x_range=(0.50, 1.0),
        )
        torque = rotation_stiffness * rotation_error + rotation_damping * (
            wrist_angular_velocity - blade.data.root_ang_vel_w
        )
        blade.permanent_wrench_composer.set_forces_and_torques(
            forces=self._clamp_norm(force, maximum_force).unsqueeze(1),
            torques=self._clamp_norm(torque, maximum_torque).unsqueeze(1),
            is_global=True,
        )


class RailStictionForce(ManagerTermBase):
    """Apply rail resistance without supplying any gripper-to-blade force."""

    def reset(self, env_ids=None) -> None:
        ids = torch.arange(self._env.num_envs, device=self._env.device) if env_ids is None else env_ids
        zeros = torch.zeros((len(ids), 1, 3), device=self._env.device)
        self._env.scene["spare_blade"].permanent_wrench_composer.set_forces_and_torques(
            forces=zeros,
            torques=zeros,
            env_ids=ids,
            is_global=True,
        )

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rail_center_y: float = 0.0,
        rail_center_z: float = BLADE_INSERTED_POS[2],
        engagement_x_range: tuple[float, float] = (0.50, 1.0),
    ) -> None:
        del env_ids
        ids = torch.arange(env.num_envs, device=env.device)
        blade = env.scene[asset_cfg.name]
        force = torch.zeros((env.num_envs, 3), device=env.device)
        force[:, 0] = rail_stiction_resistance(
            env,
            asset_cfg,
            ids,
            rail_center_y=rail_center_y,
            rail_center_z=rail_center_z,
            engagement_x_range=engagement_x_range,
        )
        blade.permanent_wrench_composer.set_forces_and_torques(
            forces=force.unsqueeze(1),
            torques=torch.zeros_like(force).unsqueeze(1),
            is_global=True,
        )


def reset_insertion_progress(env, env_ids: torch.Tensor | None) -> None:
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    for name, shape, dtype in (
        ("_insertion_previous_cost", (env.num_envs,), torch.float32),
        ("_insertion_progress_reward", (env.num_envs,), torch.float32),
        ("_insertion_previous_axial", (env.num_envs,), torch.float32),
        ("_insertion_axial_progress_reward", (env.num_envs,), torch.float32),
        ("_insertion_axial_progress_valid", (env.num_envs,), torch.bool),
        ("_insertion_previous_valid", (env.num_envs,), torch.bool),
        ("_insertion_success_hold_steps", (env.num_envs,), torch.long),
    ):
        value = getattr(env, name, None)
        if value is None or value.shape != shape:
            value = torch.zeros(shape, dtype=dtype, device=env.device)
            setattr(env, name, value)
        value[ids] = 0


def reset_insertion_joints(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    noise_by_stage: tuple[float, ...] = (0.001, 0.002, 0.004),
    poses_by_stage: tuple[tuple[float, ...], ...] = INSERTION_STAGE_ARM_JOINT_POS,
) -> None:
    """Reset around the collision-free near-slot pose with curriculum noise."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    stages = getattr(env, "_insertion_curriculum_stage", None)
    if stages is None or stages.shape[0] != env.num_envs:
        stages = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._insertion_curriculum_stage = stages
    stage_noise = torch.tensor(noise_by_stage, device=env.device)[stages[ids].clamp_max(len(noise_by_stage) - 1)]
    nominal_by_stage = robot.data.joint_pos.new_tensor(poses_by_stage)
    nominal = nominal_by_stage[stages[ids].clamp_max(len(poses_by_stage) - 1)]
    noise = (2.0 * torch.rand_like(nominal) - 1.0) * stage_noise.unsqueeze(-1)
    position = nominal + noise
    velocity = torch.zeros_like(position)
    robot.write_joint_state_to_sim(position, velocity, joint_ids=asset_cfg.joint_ids, env_ids=ids)


def reset_insertion_blade(
    env,
    env_ids: torch.Tensor | None,
    poses_by_stage: tuple[tuple[float, ...], ...] = INSERTION_STAGE_BLADE_POSE,
) -> None:
    """Place the blade at the collision-valid pose for the curriculum stage."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    stages = getattr(env, "_insertion_curriculum_stage", None)
    if stages is None or stages.shape[0] != env.num_envs:
        stages = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._insertion_curriculum_stage = stages
    blade = env.scene["spare_blade"]
    poses = blade.data.root_state_w.new_tensor(poses_by_stage)
    pose = poses[stages[ids].clamp_max(len(poses_by_stage) - 1)].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    blade.write_root_pose_to_sim(pose, env_ids=ids)
    blade.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)


def hold_gripper_closed(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    closed_positions: tuple[float, ...] = (0.45, 0.45, -0.45, 0.45, -0.45, -0.45),
) -> None:
    """Keep the real Robotiq fingers closed while Stage 1 learns insertion only."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    targets = robot.data.joint_pos.new_tensor(closed_positions).expand(len(ids), -1)
    robot.set_joint_position_target(targets, joint_ids=asset_cfg.joint_ids, env_ids=ids)


def reset_contact_gripper(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pregrasp_positions: tuple[float, ...] = (0.45, 0.45, -0.45, 0.45, -0.45, -0.45),
    closed_positions: tuple[float, ...] = (0.51, 0.51, -0.51, 0.51, -0.51, -0.51),
) -> None:
    """Start just outside the handle and let the pads close through contact."""

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene[asset_cfg.name]
    positions = robot.data.joint_pos.new_tensor(pregrasp_positions).expand(len(ids), -1).clone()
    velocities = torch.zeros_like(positions)
    robot.write_joint_state_to_sim(positions, velocities, joint_ids=asset_cfg.joint_ids, env_ids=ids)
    targets = positions.new_tensor(closed_positions).expand(len(ids), -1)
    robot.set_joint_position_target(targets, joint_ids=asset_cfg.joint_ids, env_ids=ids)


def insertion_progress_reward(env, command_name: str = "insertion_goal") -> torch.Tensor:
    """Potential improvement; standing still earns exactly zero before time cost."""

    axial, lateral, orientation = _errors(env, command_name)
    cost = axial / 0.20 + 2.0 * lateral / 0.015 + 0.5 * orientation / 0.20
    previous = getattr(env, "_insertion_previous_cost", None)
    valid = getattr(env, "_insertion_previous_valid", None)
    reward = getattr(env, "_insertion_progress_reward", None)
    if previous is None or valid is None or reward is None:
        reset_insertion_progress(env, None)
        previous = env._insertion_previous_cost
        valid = env._insertion_previous_valid
        reward = env._insertion_progress_reward
    current_step = int(env.common_step_counter)
    if getattr(env, "_insertion_progress_last_step", -1) != current_step:
        reward.copy_(torch.where(valid, (previous - cost).clamp(-0.25, 0.25) / float(env.step_dt), 0.0))
        previous.copy_(cost)
        valid.fill_(True)
        env._insertion_progress_last_step = current_step
    return reward


def insertion_axial_progress_reward(
    env,
    command_name: str = "insertion_goal",
    distance_scale: float = 0.030,
) -> torch.Tensor:
    """Reward measured reduction in insertion depth; cyclic motion nets zero.

    This is a denser Stage-0 learning signal, not a scripted action. PPO still
    chooses all six Cartesian commands and is paid only after the simulated
    blade actually moves toward the target.
    """

    axial, _, _ = _errors(env, command_name)
    previous = getattr(env, "_insertion_previous_axial", None)
    valid = getattr(env, "_insertion_axial_progress_valid", None)
    reward = getattr(env, "_insertion_axial_progress_reward", None)
    if previous is None or valid is None or reward is None:
        reset_insertion_progress(env, None)
        previous = env._insertion_previous_axial
        valid = env._insertion_axial_progress_valid
        reward = env._insertion_axial_progress_reward
    current_step = int(env.common_step_counter)
    if getattr(env, "_insertion_axial_progress_last_step", -1) != current_step:
        improvement = (previous - axial) / max(float(distance_scale), 1.0e-6)
        reward.copy_(torch.where(valid, improvement.clamp(-1.0, 1.0) / float(env.step_dt), 0.0))
        previous.copy_(axial)
        valid.fill_(True)
        env._insertion_axial_progress_last_step = current_step
    return reward


def insertion_distance_reward(env, command_name: str = "insertion_goal") -> torch.Tensor:
    """Dense pose score that gives PPO a gradient before the first success."""

    axial, lateral, orientation = _errors(env, command_name)
    return torch.exp(-axial / 0.08 - lateral / 0.010 - orientation / 0.15)


def insertion_misalignment_penalty(env, command_name: str = "insertion_goal") -> torch.Tensor:
    axial, lateral, orientation = _errors(env, command_name)
    near_rack = (axial < 0.22).to(torch.float32)
    return near_rack * (torch.square(lateral / 0.010) + 0.25 * torch.square(orientation / 0.15))


def insertion_settling_penalty(env, command_name: str = "insertion_goal") -> torch.Tensor:
    """Penalize residual blade motion only during the final 5 cm approach."""

    axial, lateral, orientation = _errors(env, command_name)
    velocity = attached_blade_velocity(env)
    near_goal = (axial < 0.050) & (lateral < 0.010) & (orientation < 0.15)
    linear_cost = torch.square(torch.linalg.vector_norm(velocity[:, :3], dim=-1) / 0.030)
    angular_cost = 0.25 * torch.square(torch.linalg.vector_norm(velocity[:, 3:], dim=-1) / 0.080)
    return near_goal.to(torch.float32) * (linear_cost + angular_cost).clamp(max=25.0)


def insertion_success_conditions(
    env,
    command_name: str = "insertion_goal",
    grasp_position_tolerance: float = 0.020,
    grasp_orientation_tolerance: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Return each independently auditable insertion-success condition."""

    axial, lateral, orientation = _errors(env, command_name)
    grasp_position, grasp_orientation = secured_blade_error_metrics(env)
    velocity = attached_blade_velocity(env)
    return {
        "axial_depth": axial <= 0.012,
        "lateral_alignment": lateral <= 0.0025,
        "orientation": orientation <= 0.0523599,
        "linear_velocity": torch.linalg.vector_norm(velocity[:, :3], dim=-1) <= 0.030,
        "angular_velocity": torch.linalg.vector_norm(velocity[:, 3:], dim=-1) <= 0.080,
        "grasp_position": grasp_position <= grasp_position_tolerance,
        "grasp_orientation": grasp_orientation <= grasp_orientation_tolerance,
    }


def insertion_success_mask(
    env,
    command_name: str = "insertion_goal",
    hold_time_s: float = 0.20,
    grasp_position_tolerance: float = 0.020,
    grasp_orientation_tolerance: float = 0.15,
) -> torch.Tensor:
    conditions = insertion_success_conditions(
        env,
        command_name,
        grasp_position_tolerance=grasp_position_tolerance,
        grasp_orientation_tolerance=grasp_orientation_tolerance,
    )
    geometric = torch.stack(tuple(conditions.values()), dim=-1).all(dim=-1)
    hold = getattr(env, "_insertion_success_hold_steps", None)
    if hold is None:
        reset_insertion_progress(env, None)
        hold = env._insertion_success_hold_steps
    current_step = int(env.common_step_counter)
    if getattr(env, "_insertion_success_last_step", -1) != current_step:
        hold.copy_(torch.where(geometric, hold + 1, torch.zeros_like(hold)))
        env._insertion_success_last_step = current_step
    required = max(1, int(round(hold_time_s / float(env.step_dt))))
    conditions["hold_duration"] = hold >= required
    env._insertion_latest_success_conditions = conditions
    return conditions["hold_duration"]


def insertion_success_reward(env, command_name: str = "insertion_goal") -> torch.Tensor:
    return insertion_success_mask(env, command_name).to(torch.float32) / float(env.step_dt)


def contact_insertion_success_mask(
    env,
    command_name: str = "insertion_goal",
    hold_time_s: float = 0.20,
) -> torch.Tensor:
    """Require a visibly retained physical grasp as well as insertion."""

    return insertion_success_mask(
        env,
        command_name,
        hold_time_s=hold_time_s,
        grasp_position_tolerance=0.010,
        grasp_orientation_tolerance=0.10,
    )


def contact_insertion_success_reward(env, command_name: str = "insertion_goal") -> torch.Tensor:
    return contact_insertion_success_mask(env, command_name).to(torch.float32) / float(env.step_dt)


def grasp_retention_reward(
    env,
    position_std: float = 0.010,
    orientation_std: float = 0.10,
) -> torch.Tensor:
    """Reward the blade remaining centered between the physical finger pads."""

    position, orientation = secured_blade_error_metrics(env)
    return torch.exp(-position / position_std - orientation / orientation_std)


def grasp_slip_penalty(
    env,
    position_free: float = 0.010,
    orientation_free: float = 0.10,
    position_scale: float = 0.015,
    orientation_scale: float = 0.15,
) -> torch.Tensor:
    """Penalize emerging slip without rewarding a stationary grasp."""

    position, orientation = secured_blade_error_metrics(env)
    position_excess = ((position - position_free) / position_scale).clamp_min(0.0)
    orientation_excess = ((orientation - orientation_free) / orientation_scale).clamp_min(0.0)
    return (position_excess.square() + 0.25 * orientation_excess.square()).clamp(max=10.0)


def insertion_failure(
    env,
    command_name: str = "insertion_goal",
    grasp_position_limit: float = 0.060,
    grasp_orientation_limit: float = 0.50,
) -> torch.Tensor:
    blade_position, _ = attached_blade_pose_world(env)
    local_position = blade_position - env.scene.env_origins
    axial, lateral, orientation = _errors(env, command_name)
    grasp_position, grasp_orientation = secured_blade_error_metrics(env)
    conditions = {
        "workspace_x": (local_position[:, 0] < 0.45) | (local_position[:, 0] > 1.10),
        "workspace_z": (local_position[:, 2] < 0.30) | (local_position[:, 2] > 1.20),
        "lateral": lateral > 0.060,
        "blade_orientation": orientation > 0.60,
        "grasp_position": grasp_position > grasp_position_limit,
        "grasp_orientation": grasp_orientation > grasp_orientation_limit,
    }
    # Preserve the pre-reset values for smoke diagnostics. Gym resets a
    # terminated environment before the caller can inspect its live tensors.
    env._insertion_latest_failure_metrics = {
        "axial": axial.clone(),
        "lateral": lateral.clone(),
        "blade_orientation": orientation.clone(),
        "grasp_position": grasp_position.clone(),
        "grasp_orientation": grasp_orientation.clone(),
    }
    env._insertion_latest_failure_conditions = conditions
    return torch.stack(tuple(conditions.values()), dim=-1).any(dim=-1)


def insertion_failure_reward(env) -> torch.Tensor:
    return env.termination_manager.get_term("insertion_failed").to(torch.float32) / float(env.step_dt)


def insertion_timeout_error_penalty(
    env,
    command_name: str = "insertion_goal",
    distance_scale: float = 0.030,
) -> torch.Tensor:
    """Charge a terminal cost proportional to unfinished insertion distance."""

    axial, _, _ = _errors(env, command_name)
    timed_out = env.termination_manager.get_term("time_out").to(torch.float32)
    remaining = (axial / max(float(distance_scale), 1.0e-6)).clamp(0.0, 6.0)
    return timed_out * remaining / float(env.step_dt)


def insertion_non_finite_state(env) -> torch.Tensor:
    robot = env.scene["robot"]
    blade_position, blade_orientation = attached_blade_pose_world(env)
    values = (
        robot.data.root_state_w,
        robot.data.joint_pos,
        robot.data.joint_vel,
        env.scene["spare_blade"].data.root_state_w,
        blade_position,
        blade_orientation,
        attached_blade_velocity(env),
    )
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for value in values:
        invalid |= ~torch.isfinite(value).all(dim=-1)
    return invalid


def elapsed_time_penalty(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


class InsertionSuccessRateCurriculum(ManagerTermBase):
    """Unlock harder starts without discarding previously learned distances."""

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        max_stage = int(cfg.params["max_stage"])
        window_size = int(cfg.params["window_size"])
        self._histories = tuple(deque(maxlen=window_size) for _ in range(max_stage + 1))
        self._level = 0
        self._level_start_step = int(env.common_step_counter)
        self._forced_stage: int | None = None
        env._insertion_curriculum_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._insertion_timeout_failure_counts = {
            name: torch.zeros((), dtype=torch.long, device=env.device) for name in INSERTION_SUCCESS_CONDITION_NAMES
        }

    def force_reset_stage(self, stage: int) -> None:
        """Force one exact reset distance for deterministic evaluation only."""

        if not 0 <= stage < len(self._histories):
            raise ValueError(f"forced insertion stage must be in [0, {len(self._histories) - 1}]")
        self._forced_stage = stage
        self._level = stage
        self._level_start_step = int(self._env.common_step_counter)
        for history in self._histories:
            history.clear()
        self._env._insertion_curriculum_stage.fill_(stage)

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        success_term: str,
        threshold: float,
        window_size: int,
        max_stage: int,
        minimum_level_steps: int,
        stage_mixtures: tuple[tuple[float, ...], ...],
    ) -> dict[str, float]:
        ids = (
            torch.arange(env.num_envs, device=env.device)[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        )
        if len(ids) > 0:
            completed = env.episode_length_buf[ids] > 0
            successes = env.termination_manager.get_term(success_term)[ids][completed]
            completed_stages = env._insertion_curriculum_stage[ids][completed]
            for stage in range(max_stage + 1):
                stage_successes = successes[completed_stages == stage]
                self._histories[stage].extend(float(value) for value in stage_successes.detach().cpu())

            time_outs = getattr(env, "reset_time_outs", None)
            timed_out = torch.zeros_like(completed) if time_outs is None else completed & time_outs[ids]
            timed_out_ids = ids[timed_out]
            latest_conditions = getattr(env, "_insertion_latest_success_conditions", None)
            if len(timed_out_ids) > 0 and latest_conditions is not None:
                for name in INSERTION_SUCCESS_CONDITION_NAMES:
                    env._insertion_timeout_failure_counts[name].add_((~latest_conditions[name][timed_out_ids]).sum())

        current_step = int(env.common_step_counter)
        elapsed = current_step - self._level_start_step
        rolling = (
            float(sum(self._histories[self._level]) / len(self._histories[self._level]))
            if self._histories[self._level]
            else 0.0
        )
        if self._forced_stage is None:
            next_level, rolling, promoted = update_curriculum_stage(
                self._level,
                tuple(self._histories[self._level]),
                threshold=threshold,
                window_size=window_size,
                max_stage=max_stage,
                steps_elapsed=elapsed,
                minimum_steps=minimum_level_steps,
            )
            if promoted:
                self._level = next_level
                self._level_start_step = current_step
                elapsed = 0

            probabilities = insertion_curriculum_probabilities(self._level, stage_mixtures)
            weights = torch.as_tensor(probabilities, dtype=torch.float32, device=env.device)
            env._insertion_curriculum_stage[ids] = torch.multinomial(weights, len(ids), replacement=True)
        else:
            env._insertion_curriculum_stage[ids] = self._forced_stage

        bucket_success = [float(sum(history) / len(history)) if history else 0.0 for history in self._histories]
        return {
            "level": float(self._level),
            "active_rolling_success": float(rolling),
            "level_control_steps": float(elapsed),
            "near_success": bucket_success[0],
            "medium_success": bucket_success[1],
            "full_success": bucket_success[2],
        }


__all__ = [
    "InsertionGoalCommand",
    "InsertionGoalCommandCfg",
    "InsertionSuccessRateCurriculum",
    "RailStictionForce",
    "SecuredBladeConstraint",
    "attached_blade_pose_world",
    "attached_blade_velocity",
    "elapsed_time_penalty",
    "contact_insertion_success_mask",
    "contact_insertion_success_reward",
    "grasp_slip_penalty",
    "grasp_retention_reward",
    "hold_gripper_closed",
    "insertion_failure",
    "insertion_failure_reward",
    "insertion_error_metrics",
    "insertion_distance_reward",
    "insertion_goal_error",
    "insertion_misalignment_penalty",
    "insertion_non_finite_state",
    "insertion_progress_reward",
    "insertion_axial_progress_reward",
    "insertion_settling_penalty",
    "insertion_success_mask",
    "insertion_success_conditions",
    "insertion_success_reward",
    "insertion_timeout_error_penalty",
    "reset_insertion_joints",
    "reset_insertion_blade",
    "reset_insertion_progress",
    "reset_contact_gripper",
    "secured_blade_error_metrics",
    "secured_blade_pose_error",
]
