"""Common20mm/s Cartesian reference bound with the frozen physical task."""
import torch

from assembly_recovery.approach_slew_v1 import clamp_impedance_target, slew_reference
from assembly_recovery.servo_impact_env_v1 import PegServoImpactEnvV1


class PegApproachSlewEnvV1(PegServoImpactEnvV1):
    def begin_jobs(self, prefix):
        if self.jobs_live:
            raise RuntimeError("Initialize reference state only between jobs")
        # Controller memory initialization; no simulator-state write and no
        # extra actor observation. The existing impedance loop already uses
        # current physical tool pose as ordinary proprioceptive feedback.
        self.slew_initial_reference = self.fingertip_midpoint_pos.clone()
        self.slew_reference = self.slew_initial_reference.clone()
        self.slew_updates = 0
        return super().begin_jobs(prefix)

    def generate_ctrl_signals(self, ctrl_target_fingertip_midpoint_pos,
                              ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos):
        if not self.jobs_live:
            return super().generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
                ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos)
        bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        desired = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise + self.actions[:, :3] * bounds
        self.slew_reference = slew_reference(self.slew_reference, desired)
        limited = clamp_impedance_target(self.slew_reference, self.fingertip_midpoint_pos, self.pos_threshold)
        self.slew_updates += 1
        self.requested_cartesian_reference = desired.clone()
        self.servo_tool_pos_before = self.fingertip_midpoint_pos.clone()
        self.clamped_impedance_target = limited.clone()
        # The superclass applies the existing terminal hold independently of
        # action targets. Record precisely that final position for replay.
        self.final_impedance_target = torch.where(self.finished_mask[:, None], self.hold_pos, limited).clone()
        self.incoming_impedance_quat = ctrl_target_fingertip_midpoint_quat.clone()
        self.final_impedance_quat = torch.where(
            self.finished_mask[:, None], self.hold_quat, ctrl_target_fingertip_midpoint_quat).clone()
        self.incoming_gripper_command = torch.as_tensor(
            ctrl_target_gripper_dof_pos, device=self.device).clone()
        # ForgeEnv already set delta_pos/delta_yaw from the original action.
        # Retain those reward bookkeeping values, the quaternion and gripper.
        return super().generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=limited,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos)

    def _sample_jobs(self):
        super()._sample_jobs()
        self.native_records[-1].update(
            requested_cartesian_reference=self.requested_cartesian_reference.clone(),
            slew_reference=self.slew_reference.clone(),
            servo_tool_pos_before=self.servo_tool_pos_before.clone(),
            clamped_impedance_target=self.clamped_impedance_target.clone(),
            final_impedance_target=self.final_impedance_target.clone(),
            reward_delta_pos=self.delta_pos.clone(),
            incoming_impedance_quat=self.incoming_impedance_quat.clone(),
            final_impedance_quat=self.final_impedance_quat.clone(),
            incoming_gripper_command=self.incoming_gripper_command.clone(),
            slew_updates_completed=self.slew_updates,
        )

