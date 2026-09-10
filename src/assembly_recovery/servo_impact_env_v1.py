"""External-servo-only variation of the frozen native-impact environment."""
from assembly_recovery.contact_impact_env_v1 import PegContactImpactEnvV1
from assembly_recovery.servo_impact_timing_v1 import ServoImpactTimingV1


class PegServoImpactEnvV1(PegContactImpactEnvV1):
    def __init__(self, cfg, *, refinement, **kwargs):
        timing = ServoImpactTimingV1(refinement)
        self._servo_applied_this_step = False
        super().__init__(cfg, refinement=refinement, **kwargs)
        # Sensor ticks, native dt, policy decimation and every physical/controller
        # parameter remain inherited. Only the external servo tick predicate
        # changes. Between-job reset still uses the original120Hz physics.
        self.timing = timing


    def _apply_action(self):
        before = self.servo_updates
        result = super()._apply_action()
        if self.jobs_live:
            self._servo_applied_this_step = self.servo_updates != before
        return result

    def _sample_jobs(self):
        super()._sample_jobs()
        # Record whether the inherited controller actually incremented its
        # command count before this native integration, independently of the
        # expected modulo grid used by the CPU timing check.
        self.native_records[-1].update(
            servo_tick=self._servo_applied_this_step,
            servo_updates_completed=self.servo_updates,
        )
