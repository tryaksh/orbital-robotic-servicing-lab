"""CPU-only timing and accounting contract for the frozen physics validation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsTimingV1:
    refinement: int

    def __post_init__(self):
        if type(self.refinement) is not int or self.refinement not in (1, 2):
            raise ValueError("Only registered 120/240 Hz physics is permitted")

    @property
    def physics_dt(self):
        return 1 / (120 * self.refinement)

    @property
    def decimation(self):
        return 8 * self.refinement

    @property
    def handoff_step(self):
        return 480 * self.refinement

    @property
    def deadline_steps(self):
        return 3600 * self.refinement

    def servo_tick(self, completed_native_steps):
        return completed_native_steps % self.refinement == 0

    def sensor_tick(self, completed_native_steps):
        return completed_native_steps % self.refinement == 0


def physics_cost_v1(*, initialization_steps, job_steps, num_envs, active_physics,
                    completed_controls, active_controls, refinement):
    timing = PhysicsTimingV1(refinement)
    if min(initialization_steps, job_steps, num_envs, active_physics, completed_controls, active_controls) < 0:
        raise ValueError("Negative simulator accounting")
    initialization = initialization_steps * num_envs
    rollout = job_steps * num_envs
    if active_physics > rollout or active_controls > completed_controls * num_envs:
        raise ValueError("Active work exceeds executed slots")
    return {
        "initialization_physics_hz": 120,
        "job_physics_hz": 120 * refinement,
        "initialization_physics_env_steps": initialization,
        "rollout_physics_env_steps": rollout,
        "active_physics_env_steps": active_physics,
        "absorbing_physics_env_steps": rollout - active_physics,
        "rollout_control_transitions": completed_controls * num_envs,
        "active_control_transitions": active_controls,
        "absorbing_control_transitions": completed_controls * num_envs - active_controls,
        "partial_control_physics_env_steps": (job_steps % timing.decimation) * num_envs,
        "charged_control_equivalent_transitions": initialization / 8 + rollout / timing.decimation,
        "charged_reference_transitions": (initialization + rollout) / 8,
        "scope": "Initialization runs at 120 Hz in both arms. Charge every executed native physics environment step, including absorbing slots and partial controls, at the fixed eight-step reference unit. Also report simulated-time control equivalents separately; fine native work is not discounted out of the cumulative ledger.",
    }


def whole_job_summary_v1(jobs):
    if not jobs:
        raise ValueError("Do not remove failed requests from the denominator")
    return {
        "requests": len(jobs), "completions": sum(j["success"] for j in jobs),
        "outcomes": dict(Counter(j["outcome"] for j in jobs)),
        "mean_seconds_per_request": sum(j["elapsed_s"] for j in jobs) / len(jobs),
        "upstream_success_seen": sum(j["upstream_success_seen"] for j in jobs),
        "forbidden_events": sum(len(j["forbidden_events"]) for j in jobs),
    }
