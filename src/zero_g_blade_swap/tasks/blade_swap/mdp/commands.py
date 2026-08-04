"""Batched eight-phase command/state machine for a complete blade swap."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude

from ..assets import BLADE_INSERTED_POS, BLADE_SIZE, SERVICE_CADDY_POS, SPARE_BLADE_POS

PHASE_APPROACH_FAILED = 0
PHASE_GRASP_FAILED = 1
PHASE_EXTRACT_FAILED = 2
PHASE_STOW_FAILED = 3
PHASE_ACQUIRE_SPARE = 4
PHASE_ALIGN_SPARE = 5
PHASE_INSERT_SPARE = 6
PHASE_RELEASE_RETREAT = 7
NUM_SWAP_PHASES = 8

FAILED_EXTRACTION_X = BLADE_INSERTED_POS[0] - 0.30
SPARE_EXTRACTION_X = SPARE_BLADE_POS[0] - 0.28
INSERTION_POSITION_TOLERANCE = 0.035
INSERTION_ORIENTATION_TOLERANCE = 0.12
INSERTION_SPEED_TOLERANCE = 0.10
BLADE_HANDLE_OFFSET = (-BLADE_SIZE[0] / 2.0 - 0.018, 0.0, 0.0)


def _phase_buffer(env) -> torch.Tensor:
    phase = getattr(env, "_swap_phase", None)
    if phase is None or phase.shape[0] != env.num_envs:
        phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._swap_phase = phase
    return phase


def _local_pose(env, asset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    asset = env.scene[asset_name]
    return asset.data.root_pos_w - env.scene.env_origins, asset.data.root_quat_w


def _blade_handle_pose_local(env, asset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    position, orientation = _local_pose(env, asset_name)
    offset = torch.tensor(BLADE_HANDLE_OFFSET, device=env.device).expand(env.num_envs, -1)
    return combine_frame_transforms(position, orientation, offset)


def _gripper_close_command(env) -> torch.Tensor:
    try:
        return env.action_manager.get_term("gripper").raw_actions[:, 0] > 0.0
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class BladeSwapCommand(CommandTerm):
    """Generate a phase-dependent task-space target in the environment frame."""

    cfg: BladeSwapCommandCfg

    def __init__(self, cfg: BladeSwapCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 7), device=self.device)
        self._rack_goal = torch.zeros_like(self._command)
        self._service_goal = torch.zeros_like(self._command)
        self._command[:, 3] = 1.0
        self._rack_goal[:, 3] = 1.0
        self._service_goal[:, 3] = 1.0
        self.metrics["active_target_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def rack_goal(self) -> torch.Tensor:
        return self._rack_goal

    @property
    def service_goal(self) -> torch.Tensor:
        return self._service_goal

    def _update_metrics(self) -> None:
        from .observations import end_effector_pose_world

        ee_pos, _ = end_effector_pose_world(self._env)
        active_target_world = self._command[:, :3] + self._env.scene.env_origins
        self.metrics["active_target_error"][:] = torch.linalg.vector_norm(ee_pos - active_target_world, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = len(env_ids)
        if count == 0:
            return
        jitter = torch.tensor(self.cfg.position_jitter, device=self.device)
        rack_offset = (2.0 * torch.rand((count, 3), device=self.device) - 1.0) * jitter
        service_offset = (2.0 * torch.rand((count, 3), device=self.device) - 1.0) * jitter
        self._rack_goal[env_ids, :3] = torch.tensor(self.cfg.rack_goal_pos, device=self.device) + rack_offset
        self._rack_goal[env_ids, 3:7] = torch.tensor(self.cfg.goal_rot, device=self.device)
        self._service_goal[env_ids, :3] = torch.tensor(self.cfg.service_goal_pos, device=self.device) + service_offset
        self._service_goal[env_ids, 3:7] = torch.tensor(self.cfg.goal_rot, device=self.device)
        self._update_command()

    def _update_command(self) -> None:
        phase = update_swap_phase(self._env)
        failed_handle_pos, failed_handle_rot = _blade_handle_pose_local(self._env, "blade")
        spare_handle_pos, spare_handle_rot = _blade_handle_pose_local(self._env, "spare_blade")

        self._command.copy_(self._rack_goal)
        failed_mask = phase <= PHASE_EXTRACT_FAILED
        self._command[failed_mask, :3] = failed_handle_pos[failed_mask]
        self._command[failed_mask, 3:7] = failed_handle_rot[failed_mask]

        stow_mask = phase == PHASE_STOW_FAILED
        self._command[stow_mask] = self._service_goal[stow_mask]

        acquire_mask = phase == PHASE_ACQUIRE_SPARE
        self._command[acquire_mask, :3] = spare_handle_pos[acquire_mask]
        self._command[acquire_mask, 3:7] = spare_handle_rot[acquire_mask]

        align_mask = phase == PHASE_ALIGN_SPARE
        self._command[align_mask] = self._rack_goal[align_mask]
        self._command[align_mask, 0] -= 0.24

        retreat_mask = phase == PHASE_RELEASE_RETREAT
        safe_pose = torch.tensor(self.cfg.safe_retreat_pose, device=self.device)
        self._command[retreat_mask] = safe_pose


@configclass
class BladeSwapCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = BladeSwapCommand
    resampling_time_range: tuple[float, float] = (45.0, 45.0)
    rack_goal_pos: tuple[float, float, float] = BLADE_INSERTED_POS
    service_goal_pos: tuple[float, float, float] = SERVICE_CADDY_POS
    goal_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    position_jitter: tuple[float, float, float] = (0.006, 0.004, 0.003)
    safe_retreat_pose: tuple[float, float, float, float, float, float, float] = (
        0.25,
        0.0,
        0.95,
        1.0,
        0.0,
        0.0,
        0.0,
    )


def goal_pose_local(env, command_name: str = "blade_goal") -> torch.Tensor:
    """Return the phase-dependent active target."""

    return env.command_manager.get_command(command_name)


def rack_goal_pose_local(env, command_name: str = "blade_goal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).rack_goal


def service_goal_pose_local(env, command_name: str = "blade_goal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).service_goal


def goal_position_world(env, command_name: str = "blade_goal") -> torch.Tensor:
    return goal_pose_local(env, command_name)[:, :3] + env.scene.env_origins


def update_swap_phase(env, command_name: str = "blade_goal") -> torch.Tensor:
    """Advance all environments through the eight geometric task phases."""

    phase = _phase_buffer(env)
    failed_pos, failed_rot = _local_pose(env, "blade")
    spare_pos, spare_rot = _local_pose(env, "spare_blade")
    failed_vel = env.scene["blade"].data.root_lin_vel_w
    spare_vel = env.scene["spare_blade"].data.root_lin_vel_w

    # During manager construction the command term may not yet be retrievable.
    try:
        rack_goal = rack_goal_pose_local(env, command_name)
        service_goal = service_goal_pose_local(env, command_name)
    except (AttributeError, KeyError):
        rack_goal = torch.tensor((*BLADE_INSERTED_POS, 1.0, 0.0, 0.0, 0.0), device=env.device).expand(env.num_envs, -1)
        service_goal = torch.tensor((*SERVICE_CADDY_POS, 1.0, 0.0, 0.0, 0.0), device=env.device).expand(
            env.num_envs, -1
        )

    from .observations import end_effector_pose_world

    ee_pos_w, ee_rot = end_effector_pose_world(env)
    ee_pos = ee_pos_w - env.scene.env_origins
    failed_handle_pos, failed_handle_rot = _blade_handle_pose_local(env, "blade")
    spare_handle_pos, spare_handle_rot = _blade_handle_pose_local(env, "spare_blade")
    close = _gripper_close_command(env)

    failed_reach = (torch.linalg.vector_norm(ee_pos - failed_handle_pos, dim=-1) < 0.075) & (
        quat_error_magnitude(ee_rot, failed_handle_rot) < 0.45
    )
    phase[(phase == PHASE_APPROACH_FAILED) & failed_reach] = PHASE_GRASP_FAILED
    phase[(phase == PHASE_GRASP_FAILED) & failed_reach & close] = PHASE_EXTRACT_FAILED

    failed_extracted = failed_pos[:, 0] <= FAILED_EXTRACTION_X
    phase[(phase == PHASE_EXTRACT_FAILED) & failed_extracted] = PHASE_STOW_FAILED

    failed_stowed = (
        (torch.linalg.vector_norm(failed_pos - service_goal[:, :3], dim=-1) < 0.055)
        & (torch.linalg.vector_norm(failed_vel, dim=-1) < 0.12)
        & ~close
    )
    phase[(phase == PHASE_STOW_FAILED) & failed_stowed] = PHASE_ACQUIRE_SPARE

    spare_near = torch.linalg.vector_norm(ee_pos - spare_handle_pos, dim=-1) < 0.080
    spare_extracted = spare_pos[:, 0] <= SPARE_EXTRACTION_X
    phase[(phase == PHASE_ACQUIRE_SPARE) & spare_near & close & spare_extracted] = PHASE_ALIGN_SPARE

    preinsert = rack_goal[:, :3].clone()
    preinsert[:, 0] -= 0.24
    spare_aligned = (torch.linalg.vector_norm(spare_pos - preinsert, dim=-1) < 0.075) & (
        quat_error_magnitude(spare_rot, rack_goal[:, 3:7]) < 0.30
    )
    phase[(phase == PHASE_ALIGN_SPARE) & spare_aligned] = PHASE_INSERT_SPARE

    inserted = (
        (torch.linalg.vector_norm(spare_pos - rack_goal[:, :3], dim=-1) < INSERTION_POSITION_TOLERANCE)
        & (quat_error_magnitude(spare_rot, rack_goal[:, 3:7]) < INSERTION_ORIENTATION_TOLERANCE)
        & (torch.linalg.vector_norm(spare_vel, dim=-1) < INSERTION_SPEED_TOLERANCE)
    )
    phase[(phase == PHASE_INSERT_SPARE) & inserted] = PHASE_RELEASE_RETREAT
    return phase


def extraction_state(env) -> torch.Tensor:
    """Compatibility helper: the failed blade has cleared the rack."""

    return update_swap_phase(env) >= PHASE_STOW_FAILED


def insertion_mask(env, command_name: str = "blade_goal") -> torch.Tensor:
    """Replacement blade is inserted; release/retreat may remain."""

    return update_swap_phase(env, command_name) >= PHASE_RELEASE_RETREAT


def swap_complete_mask(env, command_name: str = "blade_goal") -> torch.Tensor:
    """Return exact terminal success, including a continuous 0.5 second hold."""

    phase = update_swap_phase(env, command_name)
    rack_goal = rack_goal_pose_local(env, command_name)
    service_goal = service_goal_pose_local(env, command_name)
    failed_pos, failed_rot = _local_pose(env, "blade")
    spare_pos, spare_rot = _local_pose(env, "spare_blade")

    from .observations import end_effector_pose_world

    ee_pos_w, _ = end_effector_pose_world(env)
    ee_pos = ee_pos_w - env.scene.env_origins
    spare_handle, _ = _blade_handle_pose_local(env, "spare_blade")
    failed_stowed = (torch.linalg.vector_norm(failed_pos - service_goal[:, :3], dim=-1) <= 0.03) & (
        quat_error_magnitude(failed_rot, service_goal[:, 3:7]) <= math.radians(8.0)
    )
    spare_delta = spare_pos - rack_goal[:, :3]
    spare_seated = (
        (spare_delta[:, 0].abs() <= 0.02)
        & (torch.linalg.vector_norm(spare_delta[:, 1:3], dim=-1) <= 0.0025)
        & (quat_error_magnitude(spare_rot, rack_goal[:, 3:7]) <= math.radians(3.0))
        & (torch.linalg.vector_norm(env.scene["spare_blade"].data.root_lin_vel_w, dim=-1) <= 0.02)
        & (torch.linalg.vector_norm(env.scene["spare_blade"].data.root_ang_vel_w, dim=-1) <= 0.05)
    )
    geometric_success = (
        (phase == PHASE_RELEASE_RETREAT)
        & failed_stowed
        & spare_seated
        & ~_gripper_close_command(env)
        & (torch.linalg.vector_norm(ee_pos - spare_handle, dim=-1) >= 0.12)
    )

    hold_steps = getattr(env, "_swap_success_hold_steps", None)
    if hold_steps is None or hold_steps.shape[0] != env.num_envs:
        hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._swap_success_hold_steps = hold_steps
    current_step = int(env.common_step_counter)
    if getattr(env, "_swap_success_last_step", -1) != current_step:
        hold_steps.copy_(torch.where(geometric_success, hold_steps + 1, torch.zeros_like(hold_steps)))
        env._swap_success_last_step = current_step
    required_steps = max(1, math.ceil(0.5 / float(env.step_dt)))
    return hold_steps >= required_steps


def task_phase(env, command_name: str = "blade_goal") -> torch.Tensor:
    return update_swap_phase(env, command_name)


def task_phase_one_hot(env, command_name: str = "blade_goal") -> torch.Tensor:
    return torch.nn.functional.one_hot(task_phase(env, command_name), num_classes=NUM_SWAP_PHASES).to(
        dtype=torch.float32
    )


__all__ = [
    "BladeSwapCommand",
    "BladeSwapCommandCfg",
    "FAILED_EXTRACTION_X",
    "NUM_SWAP_PHASES",
    "PHASE_ACQUIRE_SPARE",
    "PHASE_ALIGN_SPARE",
    "PHASE_APPROACH_FAILED",
    "PHASE_EXTRACT_FAILED",
    "PHASE_GRASP_FAILED",
    "PHASE_INSERT_SPARE",
    "PHASE_RELEASE_RETREAT",
    "PHASE_STOW_FAILED",
    "SPARE_EXTRACTION_X",
    "extraction_state",
    "goal_pose_local",
    "goal_position_world",
    "insertion_mask",
    "rack_goal_pose_local",
    "service_goal_pose_local",
    "swap_complete_mask",
    "task_phase",
    "task_phase_one_hot",
    "update_swap_phase",
]
