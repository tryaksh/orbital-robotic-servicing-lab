"""Reward-only v3 adapter; the frozen v2 physical job path stays unchanged."""
from assembly_recovery.completion_credit import completion_credit
from assembly_recovery.tensor_jobs import OUTCOMES
from assembly_recovery.training_env import PegTrainingEnv


class PegCompletionEnv(PegTrainingEnv):
    def step(self, action, *, commanded_down=None):
        obs, upstream_reward, done, truncated, extra = super().step(action, commanded_down=commanded_down)
        successful = done & (self.tensor_jobs.outcome == OUTCOMES.index("success"))
        credit = completion_credit(self.tensor_jobs.terminal_step, successful, self.control_step,
                                   controls=self.tensor_jobs.deadline // self.cfg.decimation, decimation=self.cfg.decimation)
        extra.update(upstream_job_reward=upstream_reward, completion_credit=credit)
        return obs, upstream_reward + credit, done, truncated, extra
