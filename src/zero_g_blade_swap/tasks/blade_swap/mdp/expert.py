"""Deterministic Cartesian expert used to validate and demonstrate the task."""

from __future__ import annotations

import torch
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from ..assets import BLADE_HANDLE_OFFSET, TRANSFER_BLADE_X
from .commands import (
    GRIPPER_GRASP_ROT,
    PHASE_ACQUIRE_SPARE,
    PHASE_ALIGN_SPARE,
    PHASE_APPROACH_FAILED,
    PHASE_EXTRACT_FAILED,
    PHASE_GRASP_FAILED,
    PHASE_INSERT_SPARE,
    PHASE_RELEASE_RETREAT,
    PHASE_STOW_FAILED,
    gripper_grasp_orientation,
    gripper_handle_orientation_error,
)
from .observations import end_effector_pose_world


class ScriptedBladeSwapExpert:
    """Phase-aware Cartesian servo with real PhysX grasping.

    This is an engineering reference controller, not the learned policy.  It
    proves that the task geometry and controller can execute a sensible motion
    before PPO is asked to discover the same behavior from exploration.
    """

    def __init__(
        self,
        env,
        position_gain: float = 0.55,
        rotation_gain: float = 0.45,
        maximum_action: float = 0.30,
        grasp_dwell_steps: int = 45,
        grasp_assist: bool = True,
        kinematic_assist: bool = False,
    ) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.position_gain = position_gain
        self.rotation_gain = rotation_gain
        self.maximum_action = maximum_action
        self.grasp_dwell_steps = grasp_dwell_steps
        self.grasp_assist = grasp_assist
        self.kinematic_assist = kinematic_assist
        self._dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dwell_asset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._carrying_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._carrying_spare = torch.zeros_like(self._carrying_failed)
        self._preinsert_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._preinsert_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        arm = env.action_manager.get_term("arm")
        scale = torch.as_tensor(arm.cfg.scale, device=self.device, dtype=torch.float32)
        self._action_scale = scale.expand(self.num_envs, -1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self._dwell[ids] = 0
        self._dwell_asset[ids] = 0
        self._carrying_failed[ids] = False
        self._carrying_spare[ids] = False
        self._preinsert_hold[ids] = 0
        self._preinsert_ready[ids] = False
        zeros = torch.zeros((len(ids), 1, 3), device=self.device)
        for name in ("blade", "spare_blade"):
            self.env.scene[name].permanent_wrench_composer.set_forces_and_torques(
                forces=zeros,
                torques=zeros,
                env_ids=ids,
                is_global=True,
            )

    def _handle_pose_world(self, asset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        blade = self.env.scene[asset_name]
        offset = blade.data.root_pos_w.new_tensor(BLADE_HANDLE_OFFSET).expand(self.num_envs, -1)
        return combine_frame_transforms(blade.data.root_pos_w, blade.data.root_quat_w, offset)

    def _begin_grasp(self, mask: torch.Tensor, spare: bool) -> None:
        if not bool(mask.any()):
            return
        self._dwell[mask] = self.grasp_dwell_steps
        self._dwell_asset[mask] = int(spare)
        if spare:
            self._carrying_spare[mask] = True
        else:
            self._carrying_failed[mask] = True

    @staticmethod
    def _clamp_vector_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
        scale = (maximum / torch.linalg.vector_norm(vector, dim=-1).clamp_min(1.0e-6)).clamp(max=1.0)
        return vector * scale.unsqueeze(-1)

    def _apply_grasp_assist(self, ee_pos_w: torch.Tensor, ee_quat_w: torch.Tensor) -> None:
        """Apply a compliant 6D virtual fixture after verified gripper closure.

        The assist is deliberately limited to the deterministic demonstration
        baseline.  It is a transparent curriculum aid, not evidence that the
        current Robotiq contact model can retain every randomized blade.
        """

        if not self.grasp_assist:
            return
        grasp_rot = ee_quat_w.new_tensor(GRIPPER_GRASP_ROT).expand(self.num_envs, -1)
        desired_blade_quat = quat_mul(ee_quat_w, quat_inv(grasp_rot))
        handle_offset = ee_pos_w.new_tensor(BLADE_HANDLE_OFFSET).expand(self.num_envs, -1)
        desired_blade_pos = ee_pos_w - quat_apply(desired_blade_quat, handle_offset)
        for name, carrying in (
            ("blade", self._carrying_failed),
            ("spare_blade", self._carrying_spare),
        ):
            blade = self.env.scene[name]
            if self.kinematic_assist:
                pose = torch.cat((blade.data.root_pos_w, blade.data.root_quat_w), dim=-1)
                desired_pose = torch.cat((desired_blade_pos, desired_blade_quat), dim=-1)
                pose[carrying] = desired_pose[carrying]
                velocity = blade.data.root_vel_w.clone()
                velocity[carrying] = 0.0
                blade.write_root_pose_to_sim(pose)
                blade.write_root_velocity_to_sim(velocity)
                zeros = torch.zeros((self.num_envs, 1, 3), device=self.device)
                blade.permanent_wrench_composer.set_forces_and_torques(
                    forces=zeros,
                    torques=zeros,
                    is_global=True,
                )
                continue
            position_error, rotation_error = compute_pose_error(
                blade.data.root_pos_w,
                blade.data.root_quat_w,
                desired_blade_pos,
                desired_blade_quat,
                rot_error_type="axis_angle",
            )
            force = 800.0 * position_error - 180.0 * blade.data.root_lin_vel_w
            torque = 200.0 * rotation_error - 25.0 * blade.data.root_ang_vel_w
            force = self._clamp_vector_norm(force, 80.0) * carrying.unsqueeze(-1)
            torque = self._clamp_vector_norm(torque, 25.0) * carrying.unsqueeze(-1)
            blade.permanent_wrench_composer.set_forces_and_torques(
                forces=force.unsqueeze(1),
                torques=torque.unsqueeze(1),
                is_global=True,
            )

    def compute_actions(self) -> torch.Tensor:
        phase = self.env._swap_phase
        command = self.env.command_manager.get_command("blade_goal")
        target_pos_w = command[:, :3] + self.env.scene.env_origins
        ee_pos_w, ee_quat_w = end_effector_pose_world(self.env)
        target_quat_w = command[:, 3:7].clone()
        failed_handle_pos, failed_handle_quat = self._handle_pose_world("blade")
        spare_handle_pos, spare_handle_quat = self._handle_pose_world("spare_blade")
        failed_grasp_quat = gripper_grasp_orientation(failed_handle_quat)
        spare_grasp_quat = gripper_grasp_orientation(spare_handle_quat)

        need_failed = (
            ((phase == PHASE_GRASP_FAILED) | (phase == PHASE_EXTRACT_FAILED) | (phase == PHASE_STOW_FAILED))
            & ~self._carrying_failed
        )
        need_spare = (
            ((phase == PHASE_ACQUIRE_SPARE) | (phase == PHASE_ALIGN_SPARE) | (phase == PHASE_INSERT_SPARE))
            & ~self._carrying_spare
        )

        # Approach from the exposed -X side until the handle is close, then
        # move to its center and close.  This also repairs curriculum resets
        # that start at phases 3 or 6 without a blade already in the gripper.
        for need, handle_pos, handle_quat, grasp_quat, spare in (
            (need_failed, failed_handle_pos, failed_handle_quat, failed_grasp_quat, False),
            (need_spare, spare_handle_pos, spare_handle_quat, spare_grasp_quat, True),
        ):
            handle_distance = torch.linalg.vector_norm(ee_pos_w - handle_pos, dim=-1)
            handle_orientation_error = gripper_handle_orientation_error(ee_quat_w, handle_quat)
            standoff_offset = handle_pos.new_tensor((-0.060, 0.0, 0.0)).expand(self.num_envs, -1)
            standoff_pos, _ = combine_frame_transforms(handle_pos, handle_quat, standoff_offset)
            standoff_quat = grasp_quat
            # A six-axis arm can become orientation-singular at the handle if
            # translation is solved first.  Align the wrist at its current,
            # collision-free position before approaching the blade.
            align_wrist = need & (handle_orientation_error > 0.20)
            use_standoff = need & ~align_wrist & (handle_distance > 0.075)
            use_handle = need & ~align_wrist & ~use_standoff
            target_pos_w[align_wrist] = ee_pos_w[align_wrist]
            target_quat_w[align_wrist] = grasp_quat[align_wrist]
            target_pos_w[use_standoff] = standoff_pos[use_standoff]
            target_quat_w[use_standoff] = standoff_quat[use_standoff]
            target_pos_w[use_handle] = handle_pos[use_handle]
            target_quat_w[use_handle] = grasp_quat[use_handle]
            ready = (
                need
                & (handle_distance < 0.006)
                & (handle_orientation_error < 0.05)
            )
            self._begin_grasp(ready, spare)

        # Phase 0 deliberately stops 60 mm outside the handle.  Entering the
        # 75 mm state-machine threshold then switches to the precise grasp pose.
        approach_failed = phase == PHASE_APPROACH_FAILED
        approach_offset = failed_handle_pos.new_tensor((-0.060, 0.0, 0.0)).expand(self.num_envs, -1)
        approach_pos, _ = combine_frame_transforms(failed_handle_pos, failed_handle_quat, approach_offset)
        approach_quat = failed_grasp_quat
        target_pos_w[approach_failed] = approach_pos[approach_failed]
        target_quat_w[approach_failed] = approach_quat[approach_failed]

        # Hold the TCP on the handle while the physical fingers close instead
        # of pulling away as soon as the raw binary command changes phase.
        dwelling = self._dwell > 0
        dwell_on_spare = dwelling & (self._dwell_asset == 1)
        dwell_on_failed = dwelling & ~dwell_on_spare
        target_pos_w[dwell_on_failed] = failed_handle_pos[dwell_on_failed]
        target_quat_w[dwell_on_failed] = failed_grasp_quat[dwell_on_failed]
        target_pos_w[dwell_on_spare] = spare_handle_pos[dwell_on_spare]
        target_quat_w[dwell_on_spare] = spare_grasp_quat[dwell_on_spare]

        close = self._carrying_failed | self._carrying_spare | dwelling
        failed = self.env.scene["blade"]
        service_goal = self.env.command_manager.get_term("blade_goal").service_goal
        failed_local = failed.data.root_pos_w - self.env.scene.env_origins
        failed_stowed = (
            (torch.linalg.vector_norm(failed_local - service_goal[:, :3], dim=-1) < 0.025)
            & (quat_error_magnitude(failed.data.root_quat_w, service_goal[:, 3:7]) < 0.15)
            & (torch.linalg.vector_norm(failed.data.root_lin_vel_w, dim=-1) < 0.04)
        )
        release_failed = (phase == PHASE_STOW_FAILED) & failed_stowed & (self._dwell <= 0)
        close[release_failed] = False
        self._carrying_failed[release_failed] = False
        close[phase == PHASE_RELEASE_RETREAT] = False
        self._carrying_spare[phase == PHASE_RELEASE_RETREAT] = False

        # Tight-tolerance insertion must be split into alignment and axial
        # motion.  Hold the blade entirely outside the square guide entrance
        # until its lateral, vertical, angular, and velocity errors settle.
        rack_goal = self.env.command_manager.get_term("blade_goal").rack_goal
        spare = self.env.scene["spare_blade"]
        spare_local = spare.data.root_pos_w - self.env.scene.env_origins
        preinsert_center = rack_goal[:, :3].clone()
        preinsert_center[:, 0] = TRANSFER_BLADE_X
        preinsert_delta = spare_local - preinsert_center
        preinsert_aligned = (
            (preinsert_delta[:, 0].abs() < 0.006)
            & (torch.linalg.vector_norm(preinsert_delta[:, 1:3], dim=-1) < 0.0005)
            & (quat_error_magnitude(spare.data.root_quat_w, rack_goal[:, 3:7]) < 0.012)
            & (torch.linalg.vector_norm(spare.data.root_lin_vel_w, dim=-1) < 0.015)
            & (torch.linalg.vector_norm(spare.data.root_ang_vel_w, dim=-1) < 0.04)
        )
        settling = (phase == PHASE_INSERT_SPARE) & self._carrying_spare & ~self._preinsert_ready
        self._preinsert_hold = torch.where(
            settling & preinsert_aligned,
            self._preinsert_hold + 1,
            torch.zeros_like(self._preinsert_hold),
        )
        self._preinsert_ready |= self._preinsert_hold >= 20
        preinsert_tcp_pos, _ = combine_frame_transforms(
            preinsert_center,
            rack_goal[:, 3:7],
            preinsert_center.new_tensor(BLADE_HANDLE_OFFSET).expand(self.num_envs, -1),
        )
        target_pos_w[settling] = preinsert_tcp_pos[settling] + self.env.scene.env_origins[settling]
        target_quat_w[settling] = gripper_grasp_orientation(rack_goal[:, 3:7])[settling]
        self._apply_grasp_assist(ee_pos_w, ee_quat_w)

        robot = self.env.scene["robot"]
        target_pos_b, target_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            target_pos_w,
            target_quat_w,
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        position_error, rotation_error = compute_pose_error(
            ee_pos_b,
            ee_quat_b,
            target_pos_b,
            target_quat_b,
            rot_error_type="axis_angle",
        )
        pose_error = torch.cat(
            (self.position_gain * position_error, self.rotation_gain * rotation_error), dim=-1
        )
        arm_actions = (pose_error / self._action_scale).clamp(-self.maximum_action, self.maximum_action)
        actions = torch.cat((arm_actions, torch.where(close, 1.0, -1.0).unsqueeze(-1)), dim=-1)
        self._dwell.clamp_min_(0)
        self._dwell -= dwelling.to(torch.long)
        return actions


__all__ = ["ScriptedBladeSwapExpert"]
