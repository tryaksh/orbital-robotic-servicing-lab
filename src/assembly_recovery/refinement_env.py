"""Development physics refinement with 120 Hz servo/noise and native force checks.

Both resolutions use explicit independent sensor and initialization RNG streams.
This evaluation adapter leaves the existing training and upstream paths intact.
"""
from __future__ import annotations

import math

import isaacsim.core.utils.torch as torch_utils
import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.forge import forge_utils

from assembly_recovery.training_env import PegTrainingEnv


class PegRefinementEnv(PegTrainingEnv):
    def __init__(self, cfg, *, refinement, sensor_seed, **kwargs):
        if refinement not in (1, 2) or abs(cfg.sim.dt * refinement - 1 / 120) > 1e-12 or cfg.decimation != 8 * refinement:
            raise ValueError("Only registered 120/240 Hz physics with 15 Hz policy timing is supported")
        self.refinement = refinement
        self.sensor_seed = sensor_seed
        self.noise_generator = torch.Generator(device=cfg.sim.device).manual_seed(sensor_seed)
        self.sensor_updates = self.servo_updates = 0
        self.noise_records = []
        self.torque_records = []
        super().__init__(cfg, **kwargs)

    def begin_jobs(self, prefix):
        self.noise_generator.manual_seed(self.sensor_seed)
        self.noise_records = []
        self.torque_records = []
        self.sensor_updates = self.servo_updates = 0
        self._previous_noisy_pos = self._previous_noisy_quat = None
        self._previous_sensor_timestamp = None
        self.force_sensor_world_smooth.zero_()
        self._refresh_sensor_values()
        super().begin_jobs(prefix)

    def _refresh_sensor_values(self):
        n, device, g = self.num_envs, self.device, self.noise_generator
        pos_draw = torch.randn((n, 3), device=device, generator=g)
        axis_draw = torch.randn((n, 3), device=device, generator=g)
        angle_draw = torch.randn((n,), device=device, generator=g)
        force_draw = torch.randn((n, 3), device=device, generator=g)
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + pos_draw * self.cfg.obs_rand.fingertip_pos
        axis = axis_draw / axis_draw.norm(dim=-1, keepdim=True)
        angle = angle_draw * math.radians(self.cfg.obs_rand.fingertip_rot_deg)
        self.noisy_fingertip_quat = torch_utils.quat_mul(self.fingertip_midpoint_quat,
                                                                       torch_utils.quat_from_angle_axis(angle, axis))
        self.noisy_fingertip_quat[:, [0, 3]] = 0
        self.noisy_fingertip_quat *= self.flip_quats[:, None]
        current_quat = self.noisy_fingertip_quat / self.noisy_fingertip_quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        timestamp = self._robot._data._sim_timestamp
        if self._previous_noisy_pos is not None:
            elapsed = timestamp - self._previous_sensor_timestamp
            self.ee_linvel_fd = (self.noisy_fingertip_pos - self._previous_noisy_pos) / elapsed
            rotation = quat_mul(current_quat, quat_conjugate(self._previous_noisy_quat))
            rotation *= torch.where(rotation[:, :1] < 0, -1., 1.)
            self.ee_angvel_fd = axis_angle_from_quat(rotation) / elapsed
            self.ee_angvel_fd[:, :2] = 0
        else:
            self.ee_linvel_fd = torch.zeros_like(self.noisy_fingertip_pos)
            self.ee_angvel_fd = torch.zeros_like(self.noisy_fingertip_pos)
        self._previous_noisy_pos = self.noisy_fingertip_pos.clone()
        self._previous_noisy_quat = current_quat.clone()
        self._previous_sensor_timestamp = timestamp
        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth
        identity = torch.zeros((n, 4), device=device)
        identity[:, 0] = 1.
        smooth_force, smooth_torque = forge_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, :3], self.force_sensor_world_smooth[:, 3:],
            (identity, torch.zeros((n, 3), device=device)),
            (identity, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise))
        self.force_sensor_smooth = torch.cat((smooth_force, smooth_torque), dim=-1)
        self.noisy_force = smooth_force + force_draw * self.cfg.obs_rand.ft_force
        if self.jobs_live:
            self.sensor_updates += 1
            self.noise_records.append(torch.cat((pos_draw, axis_draw, angle_draw[:, None], force_draw), dim=-1))

    def _compute_intermediate_values(self, dt):
        timestamp = self._robot._data._sim_timestamp
        if timestamp == self._study_timestamp:
            return
        old_velocity = getattr(self, "ee_linvel_fd", None)
        old_angular = getattr(self, "ee_angvel_fd", None)
        FactoryEnv._compute_intermediate_values(self, dt)
        self._study_timestamp = timestamp
        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[:, self.force_sensor_body_idx]
        if not self.jobs_live or (self.tensor_jobs.step + 1) % self.refinement == 0:
            self._refresh_sensor_values()
        elif old_velocity is not None:
            self.ee_linvel_fd, self.ee_angvel_fd = old_velocity, old_angular
        if self.jobs_live:
            self._sample_jobs()

    def _sample_jobs(self):
        super()._sample_jobs()
        self.torque_records.append(self._robot.data.joint_effort_target.clone())

    def _apply_action(self):
        if self.jobs_live and self.tensor_jobs.step % self.refinement:
            return  # scene writes the previously commanded torques for this native substep
        if self.jobs_live:
            self.servo_updates += 1
        return super()._apply_action()
