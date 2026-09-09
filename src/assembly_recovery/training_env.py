"""Project-owned finite-cohort FORGE path; import after AppLauncher."""
from __future__ import annotations

from functools import wraps

import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from assembly_recovery.contact_witness import TensorContactRecoveryWitness, WitnessCriteria
from assembly_recovery.fault_env import PegFaultEnv
from assembly_recovery.peg_env import WithinJobMutationError
from assembly_recovery.tensor_jobs import TensorJobs, endpoint_reward
from assembly_recovery.training_inputs import prepare_actions


class PegTrainingEnv(PegFaultEnv):
    def __init__(self, cfg, *, audit=False, **kwargs):
        cfg.is_finite_horizon = True
        cfg.episode_length_s = kwargs["criteria"].deadline_s
        self.initialization_physics_steps = 0
        self.audit = audit
        super().__init__(cfg, **kwargs)
        if cfg.task.action_grad_penalty_scale != 0 or cfg.task.action_penalty_ee_scale != 0:
            raise ValueError("Resolved peg action/change reward coefficients must stay zero")
        if cfg.action_noise_model or cfg.observation_noise_model:
            raise ValueError("Only the audited upstream sensor noise path is supported")
        if self.event_manager.active_terms.get("interval", []) != ["dead_zone_thresholds"]:
            raise ValueError("Unexpected interval event: only pinned FORGE dead-zone randomization is supported")
        self.tensor_jobs = TensorJobs(self.num_envs, self.criteria, self.device)
        self.success_latch = torch.zeros((), dtype=torch.bool, device=self.device)
        self.total_control_transitions = 0
        self.total_active_physics = torch.zeros((), dtype=torch.int64, device=self.device)
        self.total_active_controls = torch.zeros_like(self.total_active_physics)
        self.terminal_observations = None
        self._last_started_initialization = 0

    def _guard_write(self, original, name):
        @wraps(original)
        def guarded(*args, **kwargs):
            if self.jobs_live:
                self.tensor_jobs.forbidden_event("within_job_state_write")
                raise WithinJobMutationError(f"Blocked {name} during measured jobs")
            return original(*args, **kwargs)
        return guarded

    def _reset_idx(self, env_ids):
        if self.jobs_live:
            self.tensor_jobs.forbidden_event("within_job_simulator_reset")
            raise WithinJobMutationError("Only between-cohort initialization is permitted")
        super()._reset_idx(env_ids)

    def set_pos_inverse_kinematics(self, *args, **kwargs):
        if self.jobs_live:
            self.tensor_jobs.forbidden_event("within_job_state_write")
            raise WithinJobMutationError("Initialization IK cannot enter a measured job")
        return super().set_pos_inverse_kinematics(*args, **kwargs)

    def step_sim_no_action(self):
        if self.jobs_live:
            self.tensor_jobs.forbidden_event("within_job_state_write")
            raise WithinJobMutationError("Initialization physics cannot enter a measured job")
        self.initialization_physics_steps += 1
        return super().step_sim_no_action()

    def _compute_intermediate_values(self, dt):
        timestamp = self._robot._data._sim_timestamp
        if timestamp == self._study_timestamp:
            return
        previous_time, previous_pos, previous_quat = self._study_timestamp, self._previous_noisy_pos, self._previous_noisy_quat
        ForgeEnv._compute_intermediate_values(self, dt)
        self._study_timestamp = timestamp
        current_quat = self.noisy_fingertip_quat / self.noisy_fingertip_quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if previous_pos is not None and previous_time is not None:
            elapsed = timestamp - previous_time
            self.ee_linvel_fd = (self.noisy_fingertip_pos - previous_pos) / elapsed
            rotation = quat_mul(current_quat, quat_conjugate(previous_quat))
            rotation *= torch.where(rotation[:, :1] < 0, -1.0, 1.0)
            self.ee_angvel_fd = axis_angle_from_quat(rotation) / elapsed
            self.ee_angvel_fd[:, :2] = 0
        self._previous_noisy_pos = self.noisy_fingertip_pos.clone()
        self._previous_noisy_quat = current_quat.clone()
        if self.jobs_live:
            self._sample_jobs()

    def begin_jobs(self, prefix):
        if self.jobs_live:
            raise RuntimeError("Finalize the previous cohort first")
        if self.initializations <= self._last_started_initialization:
            raise RuntimeError("A fresh full-cohort initialization is required before starting new jobs")
        self._last_started_initialization = self.initializations
        self.job_prefix = prefix
        self.tensor_jobs.reset()
        self.contact_witness = TensorContactRecoveryWitness(self.num_envs, WitnessCriteria(
            physics_dt=self.criteria.physics_dt, stall_window_s=self.criteria.stall_window_s), self.device)
        self.finished_mask.zero_()
        self.hold_pos.copy_(self.fingertip_midpoint_pos)
        self.hold_quat.copy_(self.fingertip_midpoint_quat)
        self.initial_tool_to_part = quat_apply_inverse(self.fingertip_midpoint_quat, self.held_pos - self.fingertip_midpoint_pos).clone()
        self.control_step = 0
        self.action_input_finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.commanded_down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.audit:
            self.audit_buffer = torch.zeros((self.tensor_jobs.deadline, self.num_envs, 14), dtype=torch.float64, device=self.device)
        self.terminal_observations = {k: torch.zeros_like(v) for k, v in self._get_observations().items()}
        self.jobs_live = True

    def _sample_jobs(self):
        upstream = self._get_curr_successes(self.cfg_task.success_threshold)
        z = self.held_pos[:, 2] - self.fixed_pos[:, 2]
        seated = upstream & (z >= -self.geometry.success_height_tolerance_m)
        tool_to_part = quat_apply_inverse(self.fingertip_midpoint_quat, self.held_pos - self.fingertip_midpoint_pos)
        drift = (tool_to_part - self.initial_tool_to_part).norm(dim=-1)
        center_offset = torch.zeros_like(self.held_pos)
        center_offset[:, 2] = self.geometry.peg_height_m / 2
        center = self.held_pos + quat_apply(self.held_quat, center_offset)
        separated = (center - self.fingertip_midpoint_pos).norm(dim=-1) > self.geometry.gross_separation_distance_m
        raw = self.force_sensor_world[:, :3].norm(dim=-1)
        filtered = self.force_sensor_smooth[:, :3].norm(dim=-1)
        height = (self.noisy_fingertip_pos - self.fixed_pos_obs_frame - self.init_fixed_pos_obs_noise)[:, 2]
        finite = (torch.isfinite(self.held_pos).all(-1) & torch.isfinite(self.held_quat).all(-1)
                  & torch.isfinite(self.noisy_fingertip_pos).all(-1) & torch.isfinite(self.noisy_fingertip_quat).all(-1)
                  & torch.isfinite(self.ee_linvel_fd).all(-1) & torch.isfinite(self.ee_angvel_fd).all(-1))
        pair_force = self.fixture_contact.data.force_matrix_w
        if pair_force is None:
            raise RuntimeError("Contact positive control unavailable")
        forces = pair_force.norm(dim=-1).sum(dim=1)
        finite &= torch.isfinite(forces[:, 0]) & self.action_input_finite
        active_before = self.tensor_jobs.active
        self.total_active_physics += active_before.sum()
        new = self.tensor_jobs.observe(upstream=upstream, seated=seated, raw=raw, filtered=filtered,
                                       separated=separated, drift=drift, height=height, down=self.commanded_down,
                                       finite=finite, fingers=forces[:, 1:])
        self.contact_witness.observe(self.tensor_jobs.step, forces[:, 0], self.held_pos[:, 2],
                                     z - self.geometry.hole_height_m, self.tensor_jobs.first_stall_step, active_before)
        self.finished_mask.copy_(~self.tensor_jobs.active)
        self.hold_pos.copy_(torch.where(new[:, None], self.fingertip_midpoint_pos, self.hold_pos))
        self.hold_quat.copy_(torch.where(new[:, None], self.fingertip_midpoint_quat, self.hold_quat))
        for name, obs in self._get_observations().items():
            self.terminal_observations[name].copy_(torch.where(new[:, None], obs, self.terminal_observations[name]))
        if self.audit:
            self.audit_buffer[self.tensor_jobs.step - 1] = torch.stack(
                (upstream, seated, raw, filtered, separated, drift, height, self.commanded_down, finite,
                 forces[:, 1], forces[:, 2], forces[:, 0], self.held_pos[:, 2], z - self.geometry.hole_height_m), dim=-1).to(torch.float64)

    def _get_rewards(self):
        successes = self._get_curr_successes(self.cfg_task.success_threshold)
        terms, scales = FactoryEnv._get_factory_rew_dict(self, successes)
        self.reward_terms = {name: (value * scales[name]).expand(self.num_envs) for name, value in terms.items()}
        self.prev_actions = self.actions.clone()
        pos_error = self.delta_pos.norm(dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = self.delta_yaw.abs() / self.cfg.ctrl.rot_action_threshold[0]
        self.success_latch |= successes.float().mean() >= self.cfg_task.delay_until_ratio
        self.reward_terms.update(
            action_penalty_asset=-(pos_error + rot_error) * self.cfg_task.action_penalty_asset_scale,
            contact_penalty=-torch.relu(self.force_sensor_smooth[:, :3].norm(dim=-1) - self.contact_penalty_thresholds)
                            * self.cfg_task.contact_penalty_scale,
            success_pred_error=-(successes.float() - (self.actions[:, 6] + 1) / 2).abs() * self.success_latch)
        # Same endpoint ordering and persistent full-cohort success latch as FORGE.
        return sum(self.reward_terms.values())

    def step(self, action, *, commanded_down=None):
        if not self.jobs_live or self.control_step * self.cfg.decimation >= self.tensor_jobs.deadline:
            raise RuntimeError("Begin a new initialized cohort before further control steps")
        valid = self.tensor_jobs.active.clone()
        action, self.action_input_finite = prepare_actions(action.to(self.device), self.actions, valid)
        self.actions = self.ema_factor * action + (1 - self.ema_factor) * self.actions
        # Actor-observable intent from the applied target, not evaluator truth.
        target_z = self.actions[:, 2] * self.cfg.ctrl.pos_action_bounds[2]
        observed_z = (self.noisy_fingertip_pos - self.fixed_pos_obs_frame - self.init_fixed_pos_obs_noise)[:, 2]
        self.commanded_down = target_z < observed_z if commanded_down is None else commanded_down
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self._compute_intermediate_values(self.physics_dt)
        self.control_step += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1
        raw_reward = self._get_rewards()
        reward = endpoint_reward(raw_reward, valid, self.tensor_jobs.terminal_step, self.tensor_jobs.step)
        self.event_manager.apply(mode="interval", dt=self.step_dt)
        done = valid & ~self.tensor_jobs.active
        obs = {k: torch.where(self.finished_mask[:, None], self.terminal_observations[k], v)
               for k, v in self._get_observations().items()}
        self.total_control_transitions += self.num_envs
        self.total_active_controls += valid.sum()
        extras = {"valid": valid, "bootstrap": self.tensor_jobs.active.clone(),
                  "terminal_observation": self.terminal_observations, "raw_reward": raw_reward}
        return obs, reward, done, torch.zeros_like(done), extras

    def end_jobs(self, incomplete_reason="probe_end"):
        self.tensor_jobs.finish(self.tensor_jobs.active, incomplete_reason)
        self.jobs_live = False
        return self.tensor_jobs.results(self.job_prefix)

    def cost_report(self):
        physics = self.total_control_transitions * self.cfg.decimation
        initialization = self.initialization_physics_steps * self.num_envs
        return {"rollout_control_transitions": self.total_control_transitions,
                "active_control_transitions": int(self.total_active_controls),
                "absorbing_control_transitions": self.total_control_transitions - int(self.total_active_controls),
                "rollout_physics_env_steps": physics, "active_physics_env_steps": int(self.total_active_physics),
                "initialization_physics_env_steps": initialization,
                "charged_control_equivalent_transitions": self.total_control_transitions + initialization / self.cfg.decimation,
                "scope": "All rollout slots, including holds, plus initialization physics divided by decimation. Probes/mining charged by their own manifests."}
