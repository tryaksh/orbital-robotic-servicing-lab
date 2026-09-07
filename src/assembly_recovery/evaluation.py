"""Simulator-independent complete-job accounting, replayable from physics samples."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class JobCriteria:
    physics_dt: float = 1 / 120
    deadline_s: float = 30.0
    seated_dwell_s: float = 0.5
    force_budget_n: float = 20.0
    stall_window_s: float = 1.0
    stall_progress_m: float = 0.001
    contact_threshold_n: float = 5.0

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Job criteria must be finite and positive")
        if self.seated_dwell_s > self.deadline_s:
            raise ValueError("The seating dwell cannot exceed the whole job deadline")


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


class JobEvaluator:
    """One requested job. Terminal outcomes cannot be overwritten by later success."""

    def __init__(self, job_id: str, criteria: JobCriteria):
        self.job_id = job_id
        self.criteria = criteria
        self.last_step = 0
        self.dwell_steps = 0
        self.upstream_success_seen = False
        self.stall_seen = False
        self.contact_window: list[tuple[int, float]] = []
        self.peak_force_n = 0.0
        self.peak_filtered_force_n = 0.0
        self.max_grasp_drift_m = 0.0
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
        if sample.grasp_separated:
            self.finish("lost_grasp")
            return
        window_steps = math.ceil(self.criteria.stall_window_s / self.criteria.physics_dt - 1e-9)
        if sample.commanded_down and sample.filtered_force_n >= self.criteria.contact_threshold_n and not sample.seated:
            self.contact_window.append((sample.step, sample.observed_height_m))
            self.contact_window = self.contact_window[-(window_steps + 1):]
            if len(self.contact_window) > window_steps:
                heights = [h for _, h in self.contact_window]
                self.stall_seen |= max(heights) - min(heights) < self.criteria.stall_progress_m
        else:
            self.contact_window.clear()
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
            "outcome": self.outcome or "incomplete",
            "finished": self.outcome is not None,
            "success": self.outcome == "success",
            "recovered_after_stall": self.outcome == "success" and self.stall_seen,
            "stall_seen": self.stall_seen,
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
