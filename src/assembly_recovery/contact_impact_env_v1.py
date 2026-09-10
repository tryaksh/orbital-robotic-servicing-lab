"""Native impact instrumentation with the frozen 120 Hz initialization path."""
from dataclasses import replace

from assembly_recovery.learned_refinement_env_v2 import PegLearnedRefinementEnvV2
from assembly_recovery.physics_validation_v1 import PhysicsTimingV1
from assembly_recovery.tensor_jobs import TensorJobs


class ContactImpactTimingV1(PhysicsTimingV1):
    def __post_init__(self):
        if type(self.refinement) is not int or self.refinement not in (1, 2, 4, 8):
            raise ValueError("Impact diagnostic supports 120/240/480/960 Hz only")


class PegContactImpactEnvV1(PegLearnedRefinementEnvV2):
    def __init__(self, cfg, *, refinement, **kwargs):
        timing = ContactImpactTimingV1(refinement)
        # All inherited initialization branches see the original timing.
        super().__init__(cfg, refinement=1, **kwargs)
        self.timing = timing

    def prepare_job_resolution(self):
        if self.jobs_live or self.resolution_prepared or self.initializations != 1:
            raise RuntimeError("Prepare exactly once before the initialized job")
        if self.sim.get_physics_dt() != 1 / 120:
            raise RuntimeError("Construction and initialization must remain at 120 Hz")
        self.cfg.sim.dt = self.timing.physics_dt
        self.cfg.decimation = self.timing.decimation
        self.cfg.sim.render_interval = self.timing.decimation
        self.fixture_contact._sim_physics_dt = self.timing.physics_dt
        self.criteria = replace(self.criteria, physics_dt=self.timing.physics_dt)
        self.tensor_jobs = TensorJobs(self.num_envs, self.criteria, self.device)
        context = self.sim.get_physics_context()
        original_step = context._step
        self.integration_dt_records = []

        def step_with_explicit_dt(current_time, update_fabric=False):
            if not self.jobs_live:
                return original_step(current_time=current_time, update_fabric=update_fabric)
            if update_fabric:
                raise ValueError("Registered impact steps require headless physics")
            context._physx_sim_interface.simulate(self.timing.physics_dt, current_time)
            context._physx_sim_interface.fetch_results()
            self.integration_dt_records.append(self.timing.physics_dt)

        context._step = step_with_explicit_dt
        self.resolution_prepared = True
        self._native_timestamp = self._study_timestamp

    def _sample_jobs(self):
        super()._sample_jobs()
        # Read-only native state. Incoming wrist force is in the child joint
        # frame (installed PhysX tensor API), despite the inherited world name.
        self.native_records[-1].update(
            gripper_joint_pos=self._robot.data.joint_pos[:, 7:9].clone(),
            gripper_joint_vel=self._robot.data.joint_vel[:, 7:9].clone(),
            robot_joint_pos=self._robot.data.joint_pos.clone(),
            robot_joint_vel=self._robot.data.joint_vel.clone(),
            held_pos=self.held_pos.clone(),
            held_quat=self.held_quat.clone(),
            held_linvel=self._held_asset.data.root_lin_vel_w.clone(),
            held_angvel=self._held_asset.data.root_ang_vel_w.clone(),
            tool_pos=self.fingertip_midpoint_pos.clone(),
            tool_quat=self.fingertip_midpoint_quat.clone(),
            tool_linvel=self.fingertip_midpoint_linvel.clone(),
            tool_angvel=self.fingertip_midpoint_angvel.clone(),
            force_sensor_body_quat=self._robot.data.body_quat_w[:, self.force_sensor_body_idx].clone(),
            active_after=self.tensor_jobs.active.clone(),
            native_timestamp=self._native_timestamp,
        )

    def cost_report(self):
        initialization = self.initialization_physics_steps * self.num_envs
        rollout = self.executed_job_steps * self.num_envs
        return {
            "initialization_physics_hz": 120,
            "job_physics_hz": 120 * self.timing.refinement,
            "initialization_physics_env_steps": initialization,
            "rollout_physics_env_steps": rollout,
            "active_physics_env_steps": int(self.total_active_physics),
            "absorbing_physics_env_steps": rollout - int(self.total_active_physics),
            "rollout_control_transitions": self.total_control_transitions,
            "active_control_transitions": int(self.total_active_controls),
            "absorbing_control_transitions": self.total_control_transitions - int(self.total_active_controls),
            "partial_control_physics_env_steps": self.executed_job_steps % self.timing.decimation * self.num_envs,
            "charged_control_equivalent_transitions": initialization / 8 + rollout / self.timing.decimation,
            "charged_reference_transitions": (initialization + rollout) / 8,
            "scope": "All native physics environment steps, including initialization and absorbing slots, charged at eight steps per reference transition. Four-second prefix diagnostic only.",
        }
