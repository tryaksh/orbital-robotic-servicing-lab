import json
from pathlib import Path

import pytest

from assembly_recovery.contact_witness import ContactRecoveryWitness
from assembly_recovery.evaluation import JobCriteria, JobEvaluator, PhysicsSample
from assembly_recovery.post_stall import post_stall_completion, summarize_endpoints
from assembly_recovery.protocol import sha256


def replay_case(*, fixture_contact=2., terminal=600, failure=None, witness_after_handoff=False):
    criteria = JobCriteria()
    job = JobEvaluator("synthetic-contact-control", criteria)
    witness = ContactRecoveryWitness()
    for step in range(1, terminal + 1):
        active = job.outcome is None
        seated = failure is None and step > terminal - 60
        height = .03 - min(step, 480) * .0001 if witness_after_handoff else .03
        if failure == "forbidden_event" and step == 550:
            job.forbidden_event("pose_write")
        sample = PhysicsSample(step, seated, seated, 21. if failure == "force_abort" and step == 550 else 6.,
                               6., failure == "lost_grasp" and step == 550, 0., height, True,
                               finger_contact_forces_n=(1., 1.))
        job.observe(sample)
        witness.observe(step, fixture_contact, .01, -.015, job.first_stall_step or 0, active)
    if job.outcome is None:
        job.finish("incomplete_probe")
    record = job.result()
    return record, witness.result(record), criteria


def test_lateral_completion_qualifies_without_withdrawal_and_keeps_old_metric():
    job, witness, criteria = replay_case()
    result = post_stall_completion(job, witness, criteria)
    assert job["success"] and witness["witnessed_failure_step"] < 480
    assert result["eligible_at_handoff"] and result["post_stall_completion"]
    assert not result["original_withdrawal_recovery"]
    assert not result["withdrawal_recovery_after_handoff"]


def test_wrist_load_alone_cannot_establish_contact_stall_endpoint():
    job, witness, criteria = replay_case(fixture_contact=0.)
    assert job["success"] and job["stall_seen"]
    result = post_stall_completion(job, witness, criteria)
    assert not result["eligible_at_handoff"] and not result["post_stall_completion"]


@pytest.mark.parametrize("failure", ["force_abort", "lost_grasp", "forbidden_event", "deadline"])
def test_terminal_failures_remain_in_the_conditional_denominator(failure):
    job, witness, criteria = replay_case(failure=failure, terminal=3600 if failure == "deadline" else 600)
    result = post_stall_completion(job, witness, criteria)
    assert result["eligible_at_handoff"] and not result["post_stall_completion"]
    summary = summarize_endpoints([result], [job])
    assert summary["all_requested_jobs"] == summary["witnessed_stalls_active_at_handoff"] == 1
    assert summary["post_stall_failures"] == 1


@pytest.mark.parametrize("terminal", [360, 480])
def test_prefix_completions_belong_to_script_and_stay_in_whole_job_denominator(terminal):
    job, witness, criteria = replay_case(terminal=terminal)
    result = post_stall_completion(job, witness, criteria)
    assert job["success"] and not result["post_stall_completion"] and not result["eligible_at_handoff"]
    summary = summarize_endpoints([result], [job])
    assert summary["whole_job_completions"] == summary["prefix_script_completions"] == 1
    assert summary["witnessed_stalls_active_at_handoff"] == 0


def test_failure_first_witnessed_after_handoff_is_not_selected_retrospectively():
    job, witness, criteria = replay_case(terminal=900, witness_after_handoff=True)
    assert job["success"] and witness["witnessed_failure_step"] > 480
    assert not post_stall_completion(job, witness, criteria)["eligible_at_handoff"]


def test_discontinuous_dwell_does_not_complete_and_late_force_wins():
    criteria = JobCriteria()
    for force in (6., 21.):
        job = JobEvaluator("dwell-control", criteria)
        witness = ContactRecoveryWitness()
        for step in range(1, 601):
            seated = step > 540 and (force == 21. or step != 570)
            job.observe(PhysicsSample(step, seated, seated, force if step == 600 else 6., 6., False, 0., .03, True,
                                      finger_contact_forces_n=(1., 1.)))
            witness.observe(step, 2., .01, -.015, job.first_stall_step or 0, True)
        job.finish("incomplete_probe")
        record = job.result()
        endpoint = post_stall_completion(record, witness.result(record), criteria)
        assert endpoint["eligible_at_handoff"] and not endpoint["post_stall_completion"]


def test_endpoint_rejects_inconsistent_failure_record_and_missing_requests():
    job, witness, criteria = replay_case()
    with pytest.raises(ValueError, match="actual stall"):
        post_stall_completion(job, {**witness, "witnessed_failure_step": 1}, criteria)
    with pytest.raises(ValueError, match="every requested job"):
        summarize_endpoints([], [job])


def test_confirmation_registration_preserves_original_gate_and_frozen_sources():
    root = Path(__file__).resolve().parents[1]
    registration = json.loads((root / "configs/failure_prefix_confirmation_v2.json").read_text())
    assert registration["seeds"] == [10071, 10072]
    assert registration["expected_charged_transitions"] == 6 * 12834.5
    assert registration["original_competence_gate_passed"] is False
    assert registration["invariants"]["final_test_cases_used"] is False
    for path, digest in registration["diagnostic_source_sha256"].items():
        assert sha256(root / path) == digest, path
    for path, digest in registration["prior_evidence_sha256"].items():
        assert sha256(root / path) == digest, path
