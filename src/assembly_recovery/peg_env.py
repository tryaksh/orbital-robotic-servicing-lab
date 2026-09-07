"""FORGE adapter for bounded evaluation. Import only after AppLauncher starts.

Training integration is deliberately separate: this adapter suppresses automatic
resets, samples each physics step, and keeps finalized jobs in a commanded hold.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import wraps

import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from assembly_recovery.evaluation import JobCriteria, JobEvaluator, PhysicsSample
from assembly_recovery.geometry import PegGeometry


class WithinJobMutationError(RuntimeError):
    pass


class PegStudyEnv(ForgeEnv):
    def __init__(self, cfg, *, criteria: JobCriteria, velocity_mode="corrected", **kwargs):
        if cfg.task.name != "peg_insert":
            raise ValueError("This adapter currently implements peg evaluation only")
        if velocity_mode not in {"upstream", "corrected"}:
            raise ValueError("Unknown velocity mode")
        self.criteria = criteria
        self.velocity_mode = velocity_mode
        self.jobs_live = False
        self.jobs: list[JobEvaluator] = []
        self.samples: list[list[dict]] = []
        self.initializations = 0
        self._study_timestamp = None
        self._previous_noisy_pos = None
        self._previous_noisy_quat = None
        self.velocity_errors: list[float] = []
        super().__init__(cfg, **kwargs)
        if abs(criteria.physics_dt - self.physics_dt) > 1e-12:
            raise ValueError("Evaluator sampling period differs from the simulator")
        self.geometry = PegGeometry(cfg.task.held_asset_cfg.diameter, cfg.task.fixed_asset_cfg.diameter,
                                    cfg.task.held_asset_cfg.height, cfg.task.fixed_asset_cfg.height,
                                    cfg.task.robot_cfg.franka_fingerpad_length, cfg.ctrl.pos_action_bounds[2],
                                    cfg.task.success_threshold)
        self.finished_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hold_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.commanded_down = False
        self.release_gripper = False
        # Catch project/upstream asset API pose and joint-state writes during jobs.
        # This does not sandbox arbitrary low-level PhysX/USD access.
        for asset in (self._robot, self._held_asset, self._fixed_asset):
            for method_name in ("write_root_pose_to_sim", "write_root_state_to_sim", "write_root_velocity_to_sim",
                                "write_joint_state_to_sim", "write_joint_position_to_sim", "write_joint_velocity_to_sim"):
                if hasattr(asset, method_name):
                    original = getattr(asset, method_name)
                    setattr(asset, method_name, self._guard_write(original, method_name))

    def _guard_write(self, original, name):
        @wraps(original)
        def guarded(*args, **kwargs):
            if self.jobs_live:
                for job in self.jobs:
                    job.forbidden_event("within_job_state_write")
                raise WithinJobMutationError(f"Blocked {name} during measured jobs")
            return original(*args, **kwargs)
        return guarded

    def _reset_idx(self, env_ids):
        if self.jobs_live:
            for job in self.jobs:
                job.forbidden_event("within_job_simulator_reset")
            raise WithinJobMutationError("A simulator reset cannot recover a measured job")
        if len(env_ids) != self.num_envs:
            raise ValueError("Pinned Factory reset logic requires a full-cohort initialization")
        self._previous_noisy_pos = None
        self._previous_noisy_quat = None
        self._study_timestamp = None
        super()._reset_idx(env_ids)
        self.initializations += 1

    def _compute_intermediate_values(self, dt):
        timestamp = self._robot._data._sim_timestamp
        if timestamp == self._study_timestamp:
            return
        previous_time = self._study_timestamp
        previous_pos = self._previous_noisy_pos
        previous_quat = self._previous_noisy_quat
        super()._compute_intermediate_values(dt)
        self._study_timestamp = timestamp
        current_quat = self.noisy_fingertip_quat / torch.linalg.vector_norm(self.noisy_fingertip_quat, dim=-1, keepdim=True).clamp_min(1e-12)
        if previous_pos is not None and previous_time is not None:
            elapsed = timestamp - previous_time
            expected_velocity = (self.noisy_fingertip_pos - previous_pos) / elapsed
            rotation = quat_mul(current_quat, quat_conjugate(previous_quat))
            rotation *= torch.where(rotation[:, :1] < 0, -1.0, 1.0)
            expected_angular = axis_angle_from_quat(rotation) / elapsed
            expected_angular[:, :2] = 0.0
            if self.velocity_mode == "corrected":
                self.ee_linvel_fd = expected_velocity
                self.ee_angvel_fd = expected_angular
            if self.jobs_live:
                self.velocity_errors.append(float((self.ee_linvel_fd - expected_velocity).abs().max()))
        self._previous_noisy_pos = self.noisy_fingertip_pos.clone()
        self._previous_noisy_quat = current_quat.clone()
        if self.jobs_live:
            self._sample_jobs()

    def _get_dones(self):
        self._compute_intermediate_values(dt=self.physics_dt)
        # The driver/evaluator owns job endings. No hidden automatic reset at 10s.
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done.clone()

    def generate_ctrl_signals(self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat,
                              ctrl_target_gripper_dof_pos):
        if self.jobs_live:
            ctrl_target_fingertip_midpoint_pos = torch.where(self.finished_mask[:, None], self.hold_pos, ctrl_target_fingertip_midpoint_pos)
            ctrl_target_fingertip_midpoint_quat = torch.where(self.finished_mask[:, None], self.hold_quat, ctrl_target_fingertip_midpoint_quat)
            if self.release_gripper:
                ctrl_target_gripper_dof_pos = 0.04
        return super().generate_ctrl_signals(ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
                                            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
                                            ctrl_target_gripper_dof_pos=ctrl_target_gripper_dof_pos)

    def begin_jobs(self, prefix):
        if self.jobs_live:
            raise RuntimeError("Finalize the previous cohort first")
        self.jobs = [JobEvaluator(f"{prefix}-{i}", self.criteria) for i in range(self.num_envs)]
        self.samples = [[] for _ in self.jobs]
        self.velocity_errors = []
        self.finished_mask[:] = False
        self.hold_pos[:] = self.fingertip_midpoint_pos
        self.hold_quat[:] = self.fingertip_midpoint_quat
        self.initial_tool_to_part = quat_apply_inverse(self.fingertip_midpoint_quat, self.held_pos - self.fingertip_midpoint_pos).clone()
        self.initial_part_pos = self.held_pos.clone()
        self.job_initializations = self.initializations
        self.jobs_live = True

    def _sample_jobs(self):
        upstream = self._get_curr_successes(self.cfg_task.success_threshold)
        # Peg base is its root; the hole's target base is at fixed_pos for this task.
        z_displacement = self.held_pos[:, 2] - self.fixed_pos[:, 2]
        seated = upstream & (z_displacement >= -self.geometry.success_height_tolerance_m)
        tool_to_part = quat_apply_inverse(self.fingertip_midpoint_quat, self.held_pos - self.fingertip_midpoint_pos)
        drift = torch.linalg.vector_norm(tool_to_part - self.initial_tool_to_part, dim=-1)
        center_offset = torch.zeros_like(self.held_pos)
        center_offset[:, 2] = self.geometry.peg_height_m / 2
        center = self.held_pos + quat_apply(self.held_quat, center_offset)
        separated = torch.linalg.vector_norm(center - self.fingertip_midpoint_pos, dim=-1) > self.geometry.gross_separation_distance_m
        raw_force = torch.linalg.vector_norm(self.force_sensor_world[:, :3], dim=-1)
        filtered_force = torch.linalg.vector_norm(self.force_sensor_smooth[:, :3], dim=-1)
        observed_height = (self.noisy_fingertip_pos - self.fixed_pos_obs_frame - self.init_fixed_pos_obs_noise)[:, 2]
        finite = (torch.isfinite(self.held_pos).all(dim=-1) & torch.isfinite(self.held_quat).all(dim=-1)
                  & torch.isfinite(self.noisy_fingertip_pos).all(dim=-1) & torch.isfinite(self.noisy_fingertip_quat).all(dim=-1)
                  & torch.isfinite(self.ee_linvel_fd).all(dim=-1) & torch.isfinite(self.ee_angvel_fd).all(dim=-1))
        rows = torch.stack((upstream, seated, raw_force, filtered_force, separated, drift, observed_height, finite), dim=1).cpu().tolist()
        for i, (job, values) in enumerate(zip(self.jobs, rows, strict=True)):
            if job.outcome is not None:
                continue
            sample = PhysicsSample(job.last_step + 1, bool(values[0]), bool(values[1]), values[2], values[3],
                                   bool(values[4]), values[5], values[6], self.commanded_down, bool(values[7]))
            self.samples[i].append(asdict(sample))
            job.observe(sample)
            if job.outcome is not None:
                self.finished_mask[i] = True
                self.hold_pos[i] = self.fingertip_midpoint_pos[i]
                self.hold_quat[i] = self.fingertip_midpoint_quat[i]

    def end_jobs(self, incomplete_reason="probe_end"):
        for job in self.jobs:
            job.finish(incomplete_reason)
        self.jobs_live = False
        return [job.result() for job in self.jobs]
