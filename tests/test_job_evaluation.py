from dataclasses import replace

import pytest

from assembly_recovery.evaluation import JobCriteria, JobEvaluator, PhysicsSample, summarize_jobs


def sample(step, **kwargs):
    return replace(PhysicsSample(step, False, False, 0.0, 0.0, False, 0.0, 0.01, False), **kwargs)


def evaluator(**kwargs):
    return JobEvaluator("job", JobCriteria(**{"physics_dt": 0.1, "deadline_s": 2, "seated_dwell_s": 0.5, **kwargs}))


def test_success_requires_continuous_dwell_not_accumulated_good_frames():
    job = evaluator()
    for step in range(1, 5):
        job.observe(sample(step, seated=True, upstream_success=True))
    assert job.outcome is None
    job.observe(sample(5))
    for step in range(6, 10):
        job.observe(sample(step, seated=True, upstream_success=True))
    assert job.outcome is None
    job.observe(sample(10, seated=True, upstream_success=True))
    assert job.outcome == "success"


@pytest.mark.parametrize("violation,outcome", [
    ({"raw_force_n": 21}, "force_abort"),
    ({"grasp_separated": True}, "lost_grasp"),
    ({"raw_force_n": float("nan")}, "nonfinite_state"),
    ({"finite": False}, "nonfinite_state"),
])
def test_failure_on_final_seating_frame_wins_and_remains_terminal(violation, outcome):
    job = evaluator()
    for step in range(1, 5):
        job.observe(sample(step, seated=True))
    job.observe(sample(5, seated=True, **violation))
    assert job.outcome == outcome
    frozen = job.result()
    job.observe(sample(6, seated=True))
    assert job.result() == frozen


def test_raw_force_spike_cannot_be_hidden_by_filtering():
    job = evaluator()
    job.observe(sample(1, raw_force_n=30, filtered_force_n=2))
    assert job.outcome == "force_abort"
    assert job.result()["peak_raw_wrist_force_n"] == 30


def test_missing_physics_sample_invalidates_continuity():
    job = evaluator()
    job.observe(sample(1))
    with pytest.raises(ValueError, match="consecutive"):
        job.observe(sample(3))


def test_upstream_success_alone_does_not_establish_seating():
    job = evaluator(deadline_s=0.5)
    for step in range(1, 6):
        job.observe(sample(step, upstream_success=True, seated=False))
    assert job.outcome == "deadline"
    assert job.result()["upstream_success_seen"]


def test_forbidden_reset_cannot_be_followed_by_counted_success():
    job = evaluator()
    job.observe(sample(1))
    job.forbidden_event("within_job_simulator_reset")
    for step in range(2, 10):
        job.observe(sample(step, seated=True))
    assert not job.result()["success"]
    assert job.result()["forbidden_events"] == [{"step": 1, "event": "within_job_simulator_reset"}]


def test_recovery_requires_a_witnessed_stall():
    job = evaluator(deadline_s=3)
    for step in range(1, 15):
        job.observe(sample(step, commanded_down=True, filtered_force_n=6, raw_force_n=6))
    assert job.stall_seen
    for step in range(15, 20):
        job.observe(sample(step, seated=True, upstream_success=True))
    assert job.result()["recovered_after_stall"]
    nominal = evaluator()
    for step in range(1, 6):
        nominal.observe(sample(step, seated=True, upstream_success=True))
    assert nominal.result()["success"]
    assert not nominal.result()["recovered_after_stall"]


def test_failure_time_and_requests_remain_in_aggregate():
    success = evaluator()
    for step in range(1, 6):
        success.observe(sample(step, seated=True))
    failed = evaluator()
    failed.observe(sample(1, raw_force_n=30))
    result = summarize_jobs([success.result(), failed.result()])
    assert result["unfinished_jobs_per_100"] == 50
    assert result["mean_seconds_per_request"] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        summarize_jobs([evaluator().result()])


@pytest.mark.parametrize("oscillate,brief_gap", [(True, False), (False, True), (True, True)])
def test_jiggling_or_brief_signal_gaps_do_not_renew_progress(oscillate, brief_gap):
    job = evaluator(deadline_s=4)
    for step in range(1, 31):
        gap = brief_gap and step % 10 == 0
        height = 0.01 + (0.004 * (-1) ** step if oscillate else 0)
        job.observe(sample(step, commanded_down=not gap, filtered_force_n=0 if gap else 6,
                           raw_force_n=6, observed_height_m=height))
    assert job.stall_seen


def test_sustained_new_depth_is_progress():
    job = evaluator(deadline_s=4)
    for step in range(1, 31):
        job.observe(sample(step, commanded_down=True, filtered_force_n=6, raw_force_n=6,
                           observed_height_m=0.02 - step * 0.0004))
    assert not job.stall_seen


@pytest.mark.parametrize("force,attempt", [(0, True), (6, False)])
def test_free_space_or_pure_withdrawal_is_not_an_insertion_stall(force, attempt):
    job = evaluator(deadline_s=4)
    for step in range(1, 31):
        job.observe(sample(step, commanded_down=attempt, filtered_force_n=force, raw_force_n=force))
    assert not job.stall_seen



def test_part_near_hand_without_finger_contact_still_counts_as_lost():
    job = evaluator(grasp_contact_grace_s=0.3)
    for step in range(1, 4):
        job.observe(sample(step, finger_contact_forces_n=(0, 0), grasp_separated=False))
    assert job.outcome == "lost_grasp"
    assert job.result()["grasp_contact_observed"]


def test_brief_contact_dropout_does_not_discard_a_retained_grasp():
    job = evaluator(grasp_contact_grace_s=0.3)
    for step in range(1, 8):
        forces = (0, 0) if step % 3 else (15, 15)
        job.observe(sample(step, finger_contact_forces_n=forces))
    assert job.outcome is None


def test_one_finger_alone_does_not_establish_retained_pinch_grasp():
    job = evaluator(grasp_contact_grace_s=0.3)
    for step in range(1, 4):
        job.observe(sample(step, finger_contact_forces_n=(15, 0)))
    assert job.outcome == "lost_grasp"



def test_release_validation_rejects_the_observed_near_hand_false_negative():
    from scripts.validate_peg import expected_outcomes_match

    jobs = [{"outcome": "lost_grasp"}] * 3 + [{"outcome": "probe_end"}]
    assert not expected_outcomes_match("release", jobs, 6, 30)
    assert expected_outcomes_match("release", [{"outcome": "lost_grasp"}] * 4, 6, 30)


def test_hold_validation_cannot_pass_by_aborting_early():
    from scripts.validate_peg import expected_outcomes_match

    assert not expected_outcomes_match("hold", [{"outcome": "force_abort"}], 6, 30)
    assert expected_outcomes_match("hold", [{"outcome": "probe_end"}], 6, 30)
    assert expected_outcomes_match("hold", [{"outcome": "deadline"}], 31, 30)
