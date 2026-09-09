"""Controller-independent contact/withdrawal witness, separate from job success.

Version 1 requires a CPU-evaluator stall with real fixture contact occupancy,
then a retained-part rise and sustained fixture clearance, then job completion.
It is additional development instrumentation, not a reinterpretation of the
historical scripted-phase witness in compare_recovery.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WitnessCriteria:
    physics_dt: float = 1 / 120
    stall_window_s: float = 1.
    fixture_contact_n: float = 0.1
    fixture_contact_occupancy: float = 0.8
    withdrawal_rise_m: float = 0.005
    withdrawal_clearance_m: float = 0.001
    fixture_clear_s: float = 0.2


DEFAULT_CRITERIA = WitnessCriteria()


class ContactRecoveryWitness:
    def __init__(self, criteria=DEFAULT_CRITERIA):
        self.criteria = criteria
        self.window = math.ceil(criteria.stall_window_s / criteria.physics_dt - 1e-9)
        self.clear_required = math.ceil(criteria.fixture_clear_s / criteria.physics_dt - 1e-9)
        self.contacts = []
        self.failure_step = self.withdrawal_step = 0
        self.failure_z = 0.
        self.clear_steps = 0

    def observe(self, step, contact, part_z, clearance, first_stall_step, active_before):
        if not active_before:
            return
        self.contacts.append(contact > self.criteria.fixture_contact_n)
        self.contacts = self.contacts[-self.window:]
        if (first_stall_step == step and len(self.contacts) == self.window
                and sum(self.contacts) / self.window >= self.criteria.fixture_contact_occupancy):
            self.failure_step = step
            self.failure_z = part_z
        if self.failure_step and step > self.failure_step:
            self.clear_steps = self.clear_steps + 1 if contact <= self.criteria.fixture_contact_n else 0
            if (not self.withdrawal_step and self.clear_steps >= self.clear_required
                    and part_z - self.failure_z >= self.criteria.withdrawal_rise_m
                    and clearance > self.criteria.withdrawal_clearance_m):
                self.withdrawal_step = step

    def result(self, job):
        recovered = (job["success"] and self.withdrawal_step > self.failure_step > 0
                     and self.withdrawal_step * self.criteria.physics_dt < job["elapsed_s"] and not job["forbidden_events"])
        return {"version": 1, "witnessed_failure_step": self.failure_step or None,
                "witnessed_withdrawal_step": self.withdrawal_step or None, "witnessed_complete_recovery": recovered}


class TensorContactRecoveryWitness:
    def __init__(self, num_envs, criteria=DEFAULT_CRITERIA, device="cpu"):
        self.n, self.criteria, self.device = num_envs, criteria, device
        self.window = math.ceil(criteria.stall_window_s / criteria.physics_dt - 1e-9)
        self.clear_required = math.ceil(criteria.fixture_clear_s / criteria.physics_dt - 1e-9)
        self.contacts = torch.zeros((num_envs, self.window), dtype=torch.bool, device=device)
        for name in ("count", "failure_step", "withdrawal_step", "clear_steps"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.int64, device=device))
        self.failure_z = torch.zeros(num_envs, dtype=torch.float64, device=device)

    def observe(self, step, contact, part_z, clearance, first_stall_step, active_before):
        c = self.criteria
        contact = contact.to(torch.float64)
        part_z, clearance = part_z.to(torch.float64), clearance.to(torch.float64)
        col = (step - 1) % self.window
        hit = contact > c.fixture_contact_n
        self.count += active_before * (hit.to(torch.int64) - self.contacts[:, col].to(torch.int64))
        self.contacts[:, col] = torch.where(active_before, hit, self.contacts[:, col])
        witnessed = ((first_stall_step == step) & active_before & (step >= self.window)
                     & (self.count.to(torch.float64) / self.window >= c.fixture_contact_occupancy))
        self.failure_step = torch.where(witnessed, step, self.failure_step)
        self.failure_z = torch.where(witnessed, part_z, self.failure_z)
        following = active_before & (self.failure_step > 0) & (step > self.failure_step)
        self.clear_steps = torch.where(following, torch.where(contact <= c.fixture_contact_n, self.clear_steps + 1, 0), self.clear_steps)
        withdrawn = (following & (self.withdrawal_step == 0) & (self.clear_steps >= self.clear_required)
                     & (part_z - self.failure_z >= c.withdrawal_rise_m) & (clearance > c.withdrawal_clearance_m))
        self.withdrawal_step = torch.where(withdrawn, step, self.withdrawal_step)

    def results(self, jobs):
        failure, withdrawal = self.failure_step.cpu().tolist(), self.withdrawal_step.cpu().tolist()
        return [{"version": 1, "witnessed_failure_step": f or None, "witnessed_withdrawal_step": w or None,
                 "witnessed_complete_recovery": bool(j["success"] and w > f > 0
                     and w * self.criteria.physics_dt < j["elapsed_s"] and not j["forbidden_events"])}
                for f, w, j in zip(failure, withdrawal, jobs, strict=True)]
