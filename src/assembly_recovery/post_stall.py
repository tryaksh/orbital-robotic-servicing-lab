"""Supplemental, mechanism-neutral endpoint for a fixed-prefix development assay.

Call only after native-physics replay has verified the frozen job evaluator and
contact witness. This never changes a controller, the withdrawal witness, or the
original competence gate. Failure exposure is determined before controller handoff.
"""
from __future__ import annotations

import math

from assembly_recovery.evaluation import JobCriteria


def post_stall_completion(job: dict, witness: dict, criteria: JobCriteria, handoff_step: int = 480) -> dict:
    if not isinstance(handoff_step, int) or handoff_step <= 0:
        raise ValueError("Handoff must be a positive native physics step")
    terminal_step = round(job["elapsed_s"] / criteria.physics_dt)
    if not math.isclose(terminal_step * criteria.physics_dt, job["elapsed_s"], abs_tol=1e-9):
        raise ValueError("Job terminal time must lie on the native physics grid")
    failure_step = witness["witnessed_failure_step"]
    if failure_step is not None and (
        not isinstance(failure_step, int) or not 0 < failure_step <= terminal_step
        or not job["stall_seen"] or failure_step != job["first_stall_step"]
    ):
        raise ValueError("Contact witness must coincide with the frozen evaluator's actual stall")
    prefix_failure = failure_step is not None and failure_step <= handoff_step
    active_at_handoff = terminal_step > handoff_step
    eligible = prefix_failure and active_at_handoff
    dwell_steps = math.ceil(criteria.seated_dwell_s / criteria.physics_dt - 1e-9)
    complete = bool(
        eligible and job["finished"] and job["success"] and job["outcome"] == "success"
        and terminal_step - dwell_steps + 1 > failure_step
        and job["elapsed_s"] <= criteria.deadline_s
        and job["peak_raw_wrist_force_n"] <= criteria.force_budget_n
        and job["grasp_contact_observed"] and not job["forbidden_events"]
    )
    withdrawal = witness["witnessed_withdrawal_step"]
    return {
        "version": "post_stall_completion_v1",
        "witnessed_failure_step": failure_step,
        "handoff_step": handoff_step,
        "prefix_terminal": not active_at_handoff,
        "witnessed_failure_by_handoff": prefix_failure,
        "active_at_handoff": active_at_handoff,
        "eligible_at_handoff": eligible,
        "post_stall_completion": complete,
        "original_withdrawal_recovery": witness["witnessed_complete_recovery"],
        "withdrawal_recovery_after_handoff": bool(
            eligible and witness["witnessed_complete_recovery"]
            and withdrawal is not None and withdrawal > handoff_step
        ),
    }


def summarize_endpoints(rows: list[dict], jobs: list[dict]) -> dict:
    if not rows or len(rows) != len(jobs):
        raise ValueError("Keep every requested job and one endpoint record per request")
    eligible = [i for i, row in enumerate(rows) if row["eligible_at_handoff"]]
    return {
        "all_requested_jobs": len(jobs),
        "whole_job_completions": sum(j["success"] for j in jobs),
        "prefix_script_completions": sum(r["prefix_terminal"] and j["success"] for r, j in zip(rows, jobs, strict=True)),
        "prefix_failed_jobs": sum(r["prefix_terminal"] and not j["success"] for r, j in zip(rows, jobs, strict=True)),
        "active_at_handoff": sum(r["active_at_handoff"] for r in rows),
        "witnessed_stalls_active_at_handoff": len(eligible),
        "post_stall_completions": sum(rows[i]["post_stall_completion"] for i in eligible),
        "post_stall_failures": sum(not rows[i]["post_stall_completion"] for i in eligible),
        "original_withdrawal_recoveries": sum(r["original_withdrawal_recovery"] for r in rows),
        "withdrawal_recoveries_after_handoff": sum(rows[i]["withdrawal_recovery_after_handoff"] for i in eligible),
        "post_stall_outcomes": {outcome: sum(jobs[i]["outcome"] == outcome for i in eligible)
                               for outcome in sorted({jobs[i]["outcome"] for i in eligible})},
    }
