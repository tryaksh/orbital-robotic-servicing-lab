import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from assembly_recovery.contact_witness import ContactRecoveryWitness, WitnessCriteria
from assembly_recovery.evaluation import JobCriteria, JobEvaluator, PhysicsSample
from assembly_recovery.physics_validation_v1 import PhysicsTimingV1, physics_cost_v1, whole_job_summary_v1
from assembly_recovery.post_stall import post_stall_completion
from assembly_recovery.protocol import sha256
from scripts.verify_learned_physics_v1 import quaternion_product_v1, verify_actions_v1, verify_rewards_v1


@pytest.mark.parametrize("factor", [1, 2])
def test_external_timing_and_native_witness_conversion(factor):
    timing = PhysicsTimingV1(factor)
    assert timing.decimation * timing.physics_dt == 1 / 15
    assert timing.handoff_step * timing.physics_dt == 4.
    assert timing.deadline_steps * timing.physics_dt == 30.
    assert sum(timing.servo_tick(s) for s in range(timing.deadline_steps)) == 3600
    assert sum(timing.sensor_tick(s) for s in range(1, timing.deadline_steps + 1)) == 3600


@pytest.mark.parametrize("factor", [0, 3, True, 1.5])
def test_unregistered_physics_rejected(factor):
    with pytest.raises(ValueError):
        PhysicsTimingV1(factor)


@pytest.mark.parametrize("factor", [1, 2])
def test_native_post_stall_dwell_and_original_withdrawal_stay_separate(factor):
    timing = PhysicsTimingV1(factor)
    criteria = JobCriteria(physics_dt=timing.physics_dt)
    job = JobEvaluator("synthetic", criteria)
    witness = ContactRecoveryWitness(WitnessCriteria(physics_dt=timing.physics_dt))
    for step in range(1, 600 * factor + 1):
        active = job.outcome is None
        seated = step > 540 * factor
        job.observe(PhysicsSample(step, seated, seated, 6., 6., False, 0., .03, True,
                                  finger_contact_forces_n=(1., 1.)))
        witness.observe(step, 2., .01, -.015, job.first_stall_step or 0, active)
    record = job.result()
    endpoint = post_stall_completion(record, witness.result(record), criteria, timing.handoff_step)
    assert record["elapsed_s"] == 5.
    assert endpoint["eligible_at_handoff"] and endpoint["post_stall_completion"]
    assert not endpoint["original_withdrawal_recovery"]


def test_native_half_servo_spike_aborts_even_when_sensor_tick_would_miss_it():
    criteria = JobCriteria(physics_dt=1 / 240)
    job = JobEvaluator("half-servo-spike", criteria)
    for step, force in enumerate((21., 0.), 1):
        job.observe(PhysicsSample(step, False, False, force, 0., False, 0., .03, True,
                                  finger_contact_forces_n=(1., 1.)))
    assert job.result()["outcome"] == "force_abort"
    assert job.result()["elapsed_s"] == 1 / 240
    assert not PhysicsTimingV1(2).sensor_tick(1)


def test_dwell_needs_every_native_step_and_force_priority_is_unchanged():
    criteria = JobCriteria(physics_dt=1 / 240)
    for interruption in ("unseated", "force"):
        job = JobEvaluator("dwell", criteria)
        for step in range(1, 121):
            seated = not (interruption == "unseated" and step == 119)
            force = 21. if interruption == "force" and step == 120 else 0.
            job.observe(PhysicsSample(step, seated, seated, force, 0., False, 0., .03, False,
                                      finger_contact_forces_n=(1., 1.)))
        assert not job.result()["success"]
        assert job.outcome == ("force_abort" if interruption == "force" else None)


@pytest.mark.parametrize("factor,expected", [(1, 12834.5), (2, 25434.5)])
def test_native_work_charged_without_discounting_finer_solver(factor, expected):
    cost = physics_cost_v1(initialization_steps=67, job_steps=3600 * factor, num_envs=28,
        active_physics=1000, completed_controls=450, active_controls=200, refinement=factor)
    assert cost["charged_reference_transitions"] == expected
    assert cost["charged_control_equivalent_transitions"] == 12834.5
    assert cost["initialization_physics_env_steps"] == 1876
    assert cost["absorbing_physics_env_steps"] == 3600 * factor * 28 - 1000


def test_partial_failed_control_is_charged_and_impossible_counts_rejected():
    cost = physics_cost_v1(initialization_steps=67, job_steps=17, num_envs=28,
        active_physics=476, completed_controls=1, active_controls=28, refinement=2)
    assert cost["charged_reference_transitions"] == 294.
    assert cost["partial_control_physics_env_steps"] == 28
    with pytest.raises(ValueError):
        physics_cost_v1(initialization_steps=67, job_steps=17, num_envs=28,
            active_physics=477, completed_controls=1, active_controls=28, refinement=2)


def test_whole_job_failures_cannot_disappear():
    with pytest.raises(ValueError):
        whole_job_summary_v1([])
    base = {"success": False, "outcome": "force_abort", "elapsed_s": 4., "upstream_success_seen": False, "forbidden_events": []}
    result = whole_job_summary_v1([base, dict(base, success=True, outcome="success", elapsed_s=5.)])
    assert result["requests"] == 2 and result["completions"] == 1
    assert result["mean_seconds_per_request"] == 4.5


def test_quaternion_replay_handles_identity_and_noncommuting_rotations():
    identity = torch.tensor([[1., 0., 0., 0.]])
    x, y = torch.tensor([[0., 1., 0., 0.]]), torch.tensor([[0., 0., 1., 0.]])
    assert torch.equal(quaternion_product_v1(identity, x), x)
    assert torch.equal(quaternion_product_v1(x, y), torch.tensor([[0., 0., 0., 1.]]))
    assert torch.equal(quaternion_product_v1(y, x), torch.tensor([[0., 0., 0., -1.]]))


@pytest.mark.parametrize("factor", [1, 2])
def test_reward_replay_rejects_after_terminal_credit(factor):
    values = torch.zeros((450, 1))
    data = {k: values.clone() for k in ("completion_credit", "job_reward", "upstream_job_reward", "raw_reward")}
    data["reward_terms"] = {"zero": values.clone()}
    report = {"refinement": factor, "jobs": [{"elapsed_s": 4., "success": False}], "discounted_job_returns": [0.]}
    assert verify_rewards_v1(data, report)
    data["job_reward"][61] = 1.
    assert not verify_rewards_v1(data, report)


def test_registration_freezes_cases_cost_and_sources_without_local_artifacts():
    root = Path(__file__).resolve().parents[1]
    registration = json.loads((root / "configs/learned_physics_validation_v1.json").read_text())
    assert registration["seed"] == 10071 and registration["requests_per_run"] == 28
    assert registration["training_launches_allowed"] == 0
    assert registration["expected_charged_reference_transitions"] == 114807.
    assert sum(r["expected_charged_reference_transitions"] for r in registration["run_order"]) == 114807.
    assert registration["run_order"][:2] == [r for r in registration["run_order"] if r["compatibility_required"]]
    assert replace(JobCriteria(), physics_dt=1 / 240).force_budget_n == 20.
    for path, digest in registration["source_sha256"].items():
        assert sha256(root / path) == digest, path
    for path, digest in registration["preserved_evidence_sha256"].items():
        assert sha256(root / path) == digest, path


def test_finite_absorbing_commands_keep_confirmed_ema_convention():
    requests = torch.full((450, 1, 7), .8)
    previous = torch.zeros((1, 7))
    applied = []
    for request in requests:
        previous = .5 * request + .5 * previous
        applied.append(previous.clone())
    data = {"actor_before": torch.zeros((450, 1, 24)), "requested_action": requests,
            "mean_action": requests, "applied_action": torch.stack(applied),
            "active": torch.zeros((450, 1), dtype=torch.bool),
            "initial": {"actions": torch.zeros((1, 7)), "ema_factor": torch.tensor([[.5]])}}
    report = {"mode": "direct_learned", "controller_position_bounds": [.05]*3,
              "controller_seated_height_m": .02, "controller_settings": {}}
    assert verify_actions_v1(data, report)
    data["applied_action"][1:] = 0.
    assert not verify_actions_v1(data, report)
