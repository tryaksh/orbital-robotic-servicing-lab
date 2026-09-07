"""Simulator-independent complete-job accounting, replayable from physics samples."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class JobCriteria:
    physics_dt: float = 1 / 120
    deadline_s: float = 30.0
    seated_dwell_s: float = 0.5
    force_budget_n: float = 20.0
    stall_window_s: float = 1.0
    stall_progress_m: float = 0.001
    contact_threshold_n: float = 5.0
    progress_filter_s: float = 0.1
    contact_occupancy: float = 0.8
    attempt_occupancy: float = 0.5
    finger_contact_threshold_n: float = 0.01
    grasp_contact_grace_s: float = 0.1

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Job criteria must be finite and positive")
        if self.seated_dwell_s > self.deadline_s:
            raise ValueError("The seating dwell cannot exceed the whole job deadline")
        if self.contact_occupancy > 1 or self.attempt_occupancy > 1:
            raise ValueError("Window occupancies cannot exceed one")


@dataclass(frozen=True)
class PhysicsSample:
    step: int
    upstream_success: bool
    seated: bool
    raw_force_n: float
    filtered_force_n: float
    grasp_separated: bool
    grasp_drift_m: float
    observed_height_m: float
    commanded_down: bool
    finite: bool = True
    finger_contact_forces_n: tuple[float, float] | None = None


class JobEvaluator:
    """One requested job. Terminal outcomes cannot be overwritten by later success."""

    def __init__(self, job_id: str, criteria: JobCriteria):
        self.job_id = job_id
        self.criteria = criteria
        self.last_step = 0
        self.dwell_steps = 0
        self.upstream_success_seen = False
        self.stall_seen = False
        self.contact_window: list[tuple[bool, bool]] = []
        self.height_window: list[float] = []
        self.best_height: float | None = None
        self.last_progress_step = 0
        self.first_stall_step: int | None = None
        self.peak_force_n = 0.0
        self.peak_filtered_force_n = 0.0
        self.max_grasp_drift_m = 0.0
        self.grasp_contact_missing_steps = 0
        self.grasp_contact_observed = False
        self.outcome: str | None = None
        self.terminal_step: int | None = None
        self.forbidden_events: list[dict] = []

    def finish(self, outcome: str) -> None:
        if self.outcome is None:
            self.outcome = outcome
            self.terminal_step = self.last_step

    def forbidden_event(self, event: str) -> None:
        self.forbidden_events.append({"step": self.last_step, "event": event})
        self.finish(event)

    def observe(self, sample: PhysicsSample) -> None:
        if self.outcome is not None:
            return
        if sample.step != self.last_step + 1:
            raise ValueError("Physics samples must be consecutive: missing samples invalidate force and dwell checks")
        self.last_step = sample.step
        numbers = (sample.raw_force_n, sample.filtered_force_n, sample.grasp_drift_m, sample.observed_height_m)
        if not sample.finite or not all(math.isfinite(x) for x in numbers):
            self.finish("nonfinite_state")
            return
        if any(x < 0 for x in numbers[:3]):
            raise ValueError("Force magnitudes and drift cannot be negative")
        self.peak_force_n = max(self.peak_force_n, sample.raw_force_n)
        self.peak_filtered_force_n = max(self.peak_filtered_force_n, sample.filtered_force_n)
        self.max_grasp_drift_m = max(self.max_grasp_drift_m, sample.grasp_drift_m)
        self.upstream_success_seen |= sample.upstream_success
        # A safety violation and success on the same step is a failed job.
        if sample.raw_force_n > self.criteria.force_budget_n:
            self.finish("force_abort")
            return
        if sample.finger_contact_forces_n is not None:
            forces = sample.finger_contact_forces_n
            if len(forces) != 2:
                raise ValueError("Expected exactly two finger-contact magnitudes")
            if not all(math.isfinite(x) for x in forces):
                self.finish("nonfinite_state")
                return
            if min(forces) < 0:
                raise ValueError("Finger-contact magnitudes cannot be negative")
            self.grasp_contact_observed = True
            bilateral = min(forces) > self.criteria.finger_contact_threshold_n
            self.grasp_contact_missing_steps = 0 if bilateral else self.grasp_contact_missing_steps + 1
        grace = math.ceil(self.criteria.grasp_contact_grace_s / self.criteria.physics_dt - 1e-9)
        if sample.grasp_separated or self.grasp_contact_missing_steps >= grace:
            self.finish("lost_grasp")
            return
        window_steps = math.ceil(self.criteria.stall_window_s / self.criteria.physics_dt - 1e-9)
        filter_steps = max(3, math.ceil(self.criteria.progress_filter_s / self.criteria.physics_dt - 1e-9))
        self.height_window.append(sample.observed_height_m)
        self.height_window = self.height_window[-filter_steps:]
        if len(self.height_window) == filter_steps:
            height = median(self.height_window)
            # Only a new depth renews progress. Repeated excursions to an old
            # depth cannot keep the timer alive by jiggling in place.
            if self.best_height is None or height < self.best_height - self.criteria.stall_progress_m:
                self.best_height = height
                self.last_progress_step = sample.step
        self.contact_window.append((sample.filtered_force_n >= self.criteria.contact_threshold_n, sample.commanded_down))
        self.contact_window = self.contact_window[-window_steps:]
        stalled = (
            not sample.seated and self.best_height is not None
            and len(self.contact_window) == window_steps
            and sample.step - self.last_progress_step >= window_steps
            and sum(contact for contact, _ in self.contact_window) / window_steps >= self.criteria.contact_occupancy
            and sum(attempt for _, attempt in self.contact_window) / window_steps >= self.criteria.attempt_occupancy
        )
        if stalled and not self.stall_seen:
            self.stall_seen = True
            self.first_stall_step = sample.step
        self.dwell_steps = self.dwell_steps + 1 if sample.seated else 0
        dwell = math.ceil(self.criteria.seated_dwell_s / self.criteria.physics_dt - 1e-9)
        deadline = math.ceil(self.criteria.deadline_s / self.criteria.physics_dt - 1e-9)
        if self.dwell_steps >= dwell and sample.step <= deadline:
            self.finish("success")
        elif sample.step >= deadline:
            self.finish("deadline")

    def result(self) -> dict:
        return {
            "job_id": self.job_id,
            "evaluator_version": 2,
            "grasp_contact_observed": self.grasp_contact_observed,
            "outcome": self.outcome or "incomplete",
            "finished": self.outcome is not None,
            "success": self.outcome == "success",
            "recovered_after_stall": self.outcome == "success" and self.stall_seen,
            "stall_seen": self.stall_seen,
            "first_stall_step": self.first_stall_step,
            "stall_detector_version": 2,
            "stall_scope": "Wrist-load and observed-progress proxy; not a part-to-fixture contact witness.",
            "upstream_success_seen": self.upstream_success_seen,
            "elapsed_s": (self.terminal_step if self.terminal_step is not None else self.last_step) * self.criteria.physics_dt,
            "peak_raw_wrist_force_n": self.peak_force_n,
            "peak_filtered_wrist_force_n": self.peak_filtered_force_n,
            "max_tool_to_part_translation_drift_m": self.max_grasp_drift_m,
            "forbidden_events": list(self.forbidden_events),
        }


def summarize_jobs(records: list[dict]) -> dict:
    if not records or any(not record["finished"] for record in records):
        raise ValueError("A cohort must contain only finalized requested jobs")
    successes = sum(record["success"] for record in records)
    return {"jobs": len(records), "successes": successes, "unfinished_jobs_per_100": 100 * (1 - successes / len(records)),
            "recoveries_after_stall": sum(record["recovered_after_stall"] for record in records),
            "mean_seconds_per_request": sum(record["elapsed_s"] for record in records) / len(records)}
