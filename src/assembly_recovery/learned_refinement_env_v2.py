"""Explicit integration dt; retain the original scene and its valid tensor views."""
from dataclasses import replace

from assembly_recovery.explicit_physics_step_v2 import explicit_physics_step_v2
from assembly_recovery.learned_refinement_env_v1 import PegLearnedRefinementEnvV1
from assembly_recovery.tensor_jobs import TensorJobs


class PegLearnedRefinementEnvV2(PegLearnedRefinementEnvV1):
    def prepare_job_resolution(self):
        if self.jobs_live or self.resolution_prepared or self.initializations != 1:
            raise RuntimeError("Prepare resolution exactly once before the initialized job")
        if self.sim.get_physics_dt() != 1 / 120:
            raise RuntimeError("Scene construction and initialization must remain at 120 Hz")
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
            explicit_physics_step_v2(context._physx_sim_interface, dt=self.timing.physics_dt,
                                     current_time=current_time, update_fabric=update_fabric)
            self.integration_dt_records.append(self.timing.physics_dt)

        # The installed PhysicsContext._step uses these same simulate/fetch calls
        # for render=False. Its usual dt comes from a USD getter. Change ONLY the
        # explicit integration argument; do not edit USD scene/timeline metadata,
        # recreate tensor views, step physics, reset, or restore any physical state.
        context._step = step_with_explicit_dt
        self.resolution_prepared = True
        self._native_timestamp = self._study_timestamp

    def _sample_jobs(self):
        super()._sample_jobs()
        self.native_records[-1]["integration_dt"] = self.integration_dt_records[-1]
