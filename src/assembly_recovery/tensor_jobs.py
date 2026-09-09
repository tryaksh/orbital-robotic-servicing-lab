"""Batched finite-job accounting. No simulator dependency or host reads in observe.

Float64 histories preserve the CPU evaluator's median/threshold semantics on
the same input samples. Export and validation are explicit boundary operations.
"""
from __future__ import annotations

import math

import torch

from assembly_recovery.evaluation import JobCriteria

OUTCOMES = ("incomplete", "success", "force_abort", "lost_grasp", "deadline", "nonfinite_state",
            "within_job_simulator_reset", "within_job_state_write", "initialization_invalid", "probe_end")


class TensorJobs:
    def __init__(self, num_envs: int, criteria: JobCriteria, device="cpu"):
        if num_envs < 1:
            raise ValueError("At least one requested job is required")
        self.n, self.criteria, self.device = num_envs, criteria, device
        steps = lambda seconds: math.ceil(seconds / criteria.physics_dt - 1e-9)  # noqa: E731
        self.deadline = steps(criteria.deadline_s)
        self.dwell = steps(criteria.seated_dwell_s)
        self.grace = steps(criteria.grasp_contact_grace_s)
        self.window = steps(criteria.stall_window_s)
        self.filter = max(3, steps(criteria.progress_filter_s))
        self.reset()

    def reset(self):
        self.step = 0
        self.forbidden_events = []
        for name in ("outcome", "last_step", "terminal_step", "dwell_steps", "last_progress_step",
                     "first_stall_step", "missing_steps", "history_count", "contact_count", "down_count",
                     "forbidden_code", "forbidden_step"):
            setattr(self, name, torch.zeros(self.n, dtype=torch.int64, device=self.device))
        for name in ("upstream_seen", "stall_seen", "grasp_observed"):
            setattr(self, name, torch.zeros(self.n, dtype=torch.bool, device=self.device))
        for name in ("peak_raw", "peak_filtered", "max_drift"):
            setattr(self, name, torch.zeros(self.n, dtype=torch.float64, device=self.device))
        self.best_height = torch.full((self.n,), float("inf"), dtype=torch.float64, device=self.device)
        self.heights = torch.zeros((self.n, self.filter), dtype=torch.float64, device=self.device)
        self.contacts = torch.zeros((self.n, self.window), dtype=torch.bool, device=self.device)
        self.downs = torch.zeros_like(self.contacts)

    @property
    def active(self):
        return self.outcome == 0

    def finish(self, mask, reason):
        selected = mask & self.active
        self.outcome = torch.where(selected, OUTCOMES.index(reason), self.outcome)
        self.terminal_step = torch.where(selected, self.last_step, self.terminal_step)
        return selected

    def forbidden_event(self, reason):
        self.forbidden_events.append((reason, self.last_step.clone()))
        code = OUTCOMES.index(reason)
        self.forbidden_code.fill_(code)
        self.forbidden_step.copy_(self.last_step)
        self.finish(self.active, reason)

    def observe(self, *, upstream, seated, raw, filtered, separated, drift, height, down, finite,
                fingers=None):
        """Consume one consecutive full-cohort physics sample; freeze ended slots.

        Inputs are device vectors (fingers: N x 2). Magnitudes come from norms;
        callers must validate nonnegative magnitudes at input/replay boundaries.
        The simulator path supplies norms, so negative magnitudes cannot arise.
        """
        self.step += 1
        before = self.active
        self.last_step = torch.where(before, self.step, self.last_step)
        raw, filtered, drift, height = (x.to(torch.float64) for x in (raw, filtered, drift, height))
        valid = finite & torch.isfinite(raw) & torch.isfinite(filtered) & torch.isfinite(drift) & torch.isfinite(height)
        self.finish(~valid, "nonfinite_state")
        valid = self.active
        self.peak_raw = torch.where(valid, torch.maximum(self.peak_raw, raw), self.peak_raw)
        self.peak_filtered = torch.where(valid, torch.maximum(self.peak_filtered, filtered), self.peak_filtered)
        self.max_drift = torch.where(valid, torch.maximum(self.max_drift, drift), self.max_drift)
        self.upstream_seen |= valid & upstream
        self.finish(raw > self.criteria.force_budget_n, "force_abort")
        if fingers is not None:
            self.finish(~torch.isfinite(fingers).all(-1), "nonfinite_state")
            valid = self.active
            self.grasp_observed |= valid
            bilateral = fingers.min(-1).values > self.criteria.finger_contact_threshold_n
            missing = torch.where(bilateral, 0, self.missing_steps + 1)
            self.missing_steps = torch.where(valid, missing, self.missing_steps)
        self.finish(separated | (self.missing_steps >= self.grace), "lost_grasp")
        valid = self.active
        col = (self.step - 1) % self.filter
        self.heights[:, col] = torch.where(valid, height, self.heights[:, col])
        self.history_count += valid
        ordered = self.heights.sort(dim=-1).values
        median = (ordered[:, (self.filter - 1) // 2] + ordered[:, self.filter // 2]) / 2
        progress = valid & (self.history_count >= self.filter) & (median < self.best_height - self.criteria.stall_progress_m)
        self.best_height = torch.where(progress, median, self.best_height)
        self.last_progress_step = torch.where(progress, self.step, self.last_progress_step)
        col = (self.step - 1) % self.window
        contact = filtered >= self.criteria.contact_threshold_n
        self.contact_count += valid * (contact.to(torch.int64) - self.contacts[:, col].to(torch.int64))
        self.down_count += valid * (down.to(torch.int64) - self.downs[:, col].to(torch.int64))
        self.contacts[:, col] = torch.where(valid, contact, self.contacts[:, col])
        self.downs[:, col] = torch.where(valid, down, self.downs[:, col])
        stalled = (valid & ~seated & torch.isfinite(self.best_height) & (self.history_count >= self.window)
                   & (self.step - self.last_progress_step >= self.window)
                   & (self.contact_count.to(torch.float64) / self.window >= self.criteria.contact_occupancy)
                   & (self.down_count.to(torch.float64) / self.window >= self.criteria.attempt_occupancy))
        self.first_stall_step = torch.where(stalled & ~self.stall_seen, self.step, self.first_stall_step)
        self.stall_seen |= stalled
        self.dwell_steps = torch.where(valid, torch.where(seated, self.dwell_steps + 1, 0), self.dwell_steps)
        self.finish((self.dwell_steps >= self.dwell) & (self.step <= self.deadline), "success")
        self.finish(torch.full_like(valid, self.step >= self.deadline), "deadline")
        return before & ~self.active

    def results(self, prefix="job"):
        names = ("outcome", "last_step", "terminal_step", "first_stall_step", "stall_seen", "grasp_observed",
                 "upstream_seen", "peak_raw", "peak_filtered", "max_drift", "forbidden_code", "forbidden_step")
        data = {name: getattr(self, name).cpu().tolist() for name in names}
        events = [(reason, steps.cpu().tolist()) for reason, steps in self.forbidden_events]
        result = []
        for i in range(self.n):
            outcome = OUTCOMES[data["outcome"][i]]
            job_events = [{"step": steps[i], "event": reason} for reason, steps in events]
            result.append({"job_id": f"{prefix}-{i}", "evaluator_version": 2,
                           "grasp_contact_observed": data["grasp_observed"][i], "outcome": outcome,
                           "finished": outcome != "incomplete", "success": outcome == "success",
                           "recovered_after_stall": outcome == "success" and data["stall_seen"][i],
                           "stall_seen": data["stall_seen"][i], "first_stall_step": data["first_stall_step"][i] or None,
                           "stall_detector_version": 2,
                           "stall_scope": "Wrist-load and observed-progress proxy; not a part-to-fixture contact witness.",
                           "upstream_success_seen": data["upstream_seen"][i],
                           "elapsed_s": data["last_step"][i] * self.criteria.physics_dt,
                           "peak_raw_wrist_force_n": data["peak_raw"][i],
                           "peak_filtered_wrist_force_n": data["peak_filtered"][i],
                           "max_tool_to_part_translation_drift_m": data["max_drift"][i], "forbidden_events": job_events})
        return result


def endpoint_reward(reward, active_before, terminal_step, endpoint_step):
    """Exact existing offline ledger: omit a terminal partial interval entirely."""
    include = active_before & ((terminal_step == 0) | (terminal_step >= endpoint_step))
    return torch.where(include, reward, 0.0)


def finite_job_gae(rewards, values, next_values, done, valid, gamma=0.995, tau=0.95):
    """Time-major rollout; chunk ends bootstrap only still-active jobs.

    terminal next observations may have arbitrary values, including NaN. Select
    zero before arithmetic rather than multiplying a NaN by a zero mask.
    """
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        bootstrap = torch.where(done[t] | ~valid[t], 0.0, next_values[t])
        delta = rewards[t] + gamma * bootstrap - values[t]
        carry = torch.where(valid[t], delta + gamma * tau * torch.where(done[t], 0.0, carry), 0.0)
        advantage[t] = carry
    return advantage, torch.where(valid, advantage + values, 0.0)
