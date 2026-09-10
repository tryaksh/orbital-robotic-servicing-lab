"""Frozen-policy physics sensitivity with the confirmed completion reward path.

Initialize every process at the original 120 Hz, without replaying or restoring
part states. Change the solver timestep once, between initialization and the job.
Keep the original sensor/filter/RNG state and the exact upstream sensor formulas.
"""
from __future__ import annotations

from dataclasses import replace

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from assembly_recovery.completion_env import PegCompletionEnv
from assembly_recovery.physics_validation_v1 import PhysicsTimingV1, physics_cost_v1
from assembly_recovery.tensor_jobs import TensorJobs


class PegLearnedRefinementEnvV1(PegCompletionEnv):
    def __init__(self, cfg, *, refinement, **kwargs):
        self.timing = PhysicsTimingV1(refinement)
        if cfg.sim.dt != 1 / 120 or cfg.decimation != 8:
            raise ValueError("Both resolutions require the unchanged 120 Hz initialization")
        self.resolution_prepared = False
        self.executed_job_steps = 0
        self.servo_updates = 0
        self.sensor_updates = 0
        self.native_records = []
        self.sensor_records = []
        self._native_timestamp = None
        self._sensor_tick = False
        super().__init__(cfg, **kwargs)

    def step_sim_no_action(self):
        if self.resolution_prepared:
            raise RuntimeError("No initialization physics after preparing job resolution")
        if self.initialization_physics_steps >= 256:
            raise RuntimeError("Registered initialization bound exceeded")
        return super().step_sim_no_action()

    def prepare_job_resolution(self):
        if self.jobs_live or self.resolution_prepared or self.initializations != 1:
            raise RuntimeError("Prepare resolution exactly once after initialization, before the job")
        # The supported setter changes the scene timestep, without reset or step.
        if self.timing.refinement == 2:
            self.sim.set_simulation_dt(physics_dt=self.timing.physics_dt, rendering_dt=self.step_dt)
            self.cfg.sim.dt = self.timing.physics_dt
            self.cfg.decimation = self.timing.decimation
            self.cfg.sim.render_interval = self.timing.decimation
            # ContactSensor caches the impulse-to-force dt at construction.
            # It must use the NEW native dt for subsequent live physics samples.
            self.fixture_contact._sim_physics_dt = self.timing.physics_dt
            self.criteria = replace(self.criteria, physics_dt=self.timing.physics_dt)
            self.tensor_jobs = TensorJobs(self.num_envs, self.criteria, self.device)
        if (self.sim.get_physics_dt() != self.timing.physics_dt
                or self.fixture_contact._sim_physics_dt != self.timing.physics_dt
                or self.physics_dt * self.cfg.decimation != 1 / 15):
            raise RuntimeError("Simulator, contact conversion and external timing disagree")
        self.resolution_prepared = True
        self._native_timestamp = self._study_timestamp

    def begin_jobs(self, prefix):
        if not self.resolution_prepared or self.executed_job_steps:
            raise RuntimeError("One prepared finite job cohort is permitted")
        # No refresh, reseed, filter zero, pose write, or extra random draw.
        self.job_start_sim_time = self.sim.current_time
        self.job_start_sensor_timestamp = self._study_timestamp
        return super().begin_jobs(prefix)

    def _compute_intermediate_values(self, dt):
        if not self.jobs_live:
            return super()._compute_intermediate_values(dt)
        timestamp = self._robot._data._sim_timestamp
        if timestamp == self._native_timestamp:
            return
        self._native_timestamp = timestamp
        self._sensor_tick = self.timing.sensor_tick(self.tensor_jobs.step + 1)
        if self._sensor_tick:
            # Same ForgeEnv formulas and global RNG, with the original corrected
            # noisy-pose derivative using the previous SENSOR timestamp.
            self.sensor_updates += 1
            return super()._compute_intermediate_values(dt)
        velocity, angular = self.ee_linvel_fd, self.ee_angvel_fd
        FactoryEnv._compute_intermediate_values(self, dt)
        self.ee_linvel_fd, self.ee_angvel_fd = velocity, angular
        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[:, self.force_sensor_body_idx]
        # Preserve the previous sensor timestamp/noise/filter on fine substeps.
        self._sample_jobs()

    def _apply_action(self):
        if self.jobs_live:
            if not self.timing.servo_tick(self.tensor_jobs.step):
                return
            self.servo_updates += 1
        return super()._apply_action()

    def _sample_jobs(self):
        self.executed_job_steps += 1
        super()._sample_jobs()
        matrix = self.fixture_contact.data.force_matrix_w
        impulse = self.fixture_contact.contact_physx_view.get_contact_force_matrix(dt=1.).view_as(matrix)
        self.native_records.append({
            "torques": self._robot.data.joint_effort_target.clone(),
            "gripper_targets": self._robot.data.joint_pos_target[:, 7:9].clone(),
            "raw_wrist": self.force_sensor_world.clone(),
            "contact_force": matrix.clone(), "contact_impulse": impulse.clone(),
            "sim_time": self.sim.current_time - self.job_start_sim_time,
            "sensor_tick": self._sensor_tick,
        })
        if self._sensor_tick:
            self.sensor_records.append({
                name: getattr(self, name).clone() for name in (
                    "noisy_fingertip_pos", "noisy_fingertip_quat", "ee_linvel_fd", "ee_angvel_fd",
                    "force_sensor_world_smooth", "force_sensor_smooth", "noisy_force",
                    "fingertip_midpoint_pos", "fingertip_midpoint_quat")
            } | {"timestamp": self._study_timestamp - self.job_start_sensor_timestamp})

    def cost_report(self):
        return physics_cost_v1(
            initialization_steps=self.initialization_physics_steps, job_steps=self.executed_job_steps,
            num_envs=self.num_envs, active_physics=int(self.total_active_physics),
            completed_controls=self.total_control_transitions // self.num_envs,
            active_controls=int(self.total_active_controls), refinement=self.timing.refinement)
