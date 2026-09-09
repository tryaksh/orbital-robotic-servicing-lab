"""CPU parity tests use synthetic mixed cohorts, never local simulator artifacts."""
from dataclasses import replace

import pytest
import torch

from assembly_recovery.evaluation import JobCriteria, JobEvaluator, PhysicsSample
from assembly_recovery.tensor_jobs import TensorJobs, endpoint_reward, finite_job_gae

torch.set_num_threads(1)


def tensor_sample(samples):
    mapping = {"upstream": "upstream_success", "seated": "seated", "raw": "raw_force_n",
               "filtered": "filtered_force_n", "separated": "grasp_separated", "drift": "grasp_drift_m",
               "height": "observed_height_m", "down": "commanded_down", "finite": "finite"}
    boolean = {"upstream", "seated", "separated", "down", "finite"}
    result = {name: torch.tensor([getattr(s, field) for s in samples],
                                dtype=torch.bool if name in boolean else torch.float64)
              for name, field in mapping.items()}
    if samples[0].finger_contact_forces_n is not None:
        result["fingers"] = torch.tensor([s.finger_contact_forces_n for s in samples], dtype=torch.float64)
    return result


def blank(step, **kw):
    return replace(PhysicsSample(step, False, False, 0., 0., False, 0., 0.02, False, True, (10., 10.)), **kw)


def assert_parity(cpu, batch):
    assert batch.results() == [job.result() for job in cpu]
    assert batch.dwell_steps.tolist() == [j.dwell_steps for j in cpu]
    assert batch.last_progress_step.tolist() == [j.last_progress_step for j in cpu]
    assert batch.missing_steps.tolist() == [j.grasp_contact_missing_steps for j in cpu]


@pytest.mark.parametrize("dt", [0.1, 1 / 120, 1 / 240])
def test_mixed_terminal_and_stall_cohort_exact_parity_every_sample(dt):
    criteria = JobCriteria(physics_dt=dt, deadline_s=3)
    batch = TensorJobs(12, criteria)
    cpu = [JobEvaluator(f"job-{i}", criteria) for i in range(12)]
    for step in range(1, batch.deadline + 9):
        seconds = step * dt
        variants = [
            {"seated": seconds > 0.2, "upstream_success": True},
            {"seated": seconds % 0.6 < 0.4},
            {"raw_force_n": 21. if seconds >= 0.5 else 0., "seated": True},
            {"grasp_separated": seconds >= 0.5, "seated": True},
            {"finger_contact_forces_n": (10., 0.)},
            {"finger_contact_forces_n": (10., 0.) if step % 3 else (10., 10.)},
            {"filtered_force_n": 6., "raw_force_n": 6., "commanded_down": True,
             "observed_height_m": 0.02 + (0.004 if step % 2 else -0.004)},
            {"filtered_force_n": 6., "raw_force_n": 6., "commanded_down": True,
             "observed_height_m": 0.02 - seconds * 0.003},
            {"upstream_success": True},
            {"raw_force_n": float("nan") if seconds > 0.2 else 0., "seated": True},
            {"raw_force_n": 21., "finger_contact_forces_n": (float("nan"), 10.)},
            {"finger_contact_forces_n": (float("nan"), 10.), "grasp_separated": True},
        ]
        samples = [blank(step, **v) for v in variants]
        for job, sample in zip(cpu, samples, strict=True):
            job.observe(sample)
        batch.observe(**tensor_sample(samples))
        assert_parity(cpu, batch)


def test_even_window_median_averages_middle_and_strict_progress_threshold():
    criteria = JobCriteria(physics_dt=0.025, progress_filter_s=0.1, deadline_s=2)
    batch, cpu = TensorJobs(1, criteria), JobEvaluator("job-0", criteria)
    for step, height in enumerate([0., 2., 0., 2., 1., 1., 1., 1.], 1):
        s = blank(step, observed_height_m=height)
        cpu.observe(s)
        batch.observe(**tensor_sample([s]))
        assert_parity([cpu], batch)
    assert batch.best_height.item() == 1.0  # torch.median would initially choose 0


def test_dwell_at_deadline_wins_but_late_dwell_is_deadline():
    c = JobCriteria(physics_dt=0.1, deadline_s=1, seated_dwell_s=0.5)
    batch = TensorJobs(2, c)
    cpu = [JobEvaluator(f"job-{i}", c) for i in range(2)]
    for step in range(1, 11):
        samples = [blank(step, seated=step >= 6), blank(step, seated=step >= 7)]
        batch.observe(**tensor_sample(samples))
        for job, s in zip(cpu, samples, strict=True):
            job.observe(s)
    assert_parity(cpu, batch)
    assert [r["outcome"] for r in batch.results()] == ["success", "deadline"]


@pytest.mark.parametrize("event", ["within_job_state_write", "within_job_simulator_reset"])
def test_forbidden_event_fails_live_jobs_and_cannot_rewrite_ended_job(event):
    batch = TensorJobs(2, JobCriteria())
    cpu = [JobEvaluator(f"job-{i}", batch.criteria) for i in range(2)]
    samples = [blank(1), blank(1, raw_force_n=21.)]
    batch.observe(**tensor_sample(samples))
    for job, s in zip(cpu, samples, strict=True):
        job.observe(s)
        job.forbidden_event(event)
    batch.forbidden_event(event)
    assert_parity(cpu, batch)


@pytest.mark.parametrize("terminal", [1, 7, 8, 9, 16])
def test_partial_terminal_endpoint_reward_matches_preserved_ledger(terminal):
    from assembly_recovery.reward_ledger import discounted_job_return

    rows = [{"step": step, "reward": [2. * step], "reward_terms": {"one": [2. * step]}} for step in range(1, 5)]
    rewards = []
    for step in range(1, 5):
        active = torch.tensor([(step - 1) * 8 < terminal])
        term = torch.tensor([terminal if terminal <= step * 8 else 0])
        rewards.append(endpoint_reward(torch.tensor([2. * step]), active, term, step * 8).item())
    total = sum(0.995 ** i * r for i, r in enumerate(rewards))
    assert total == pytest.approx(discounted_job_return(rows, 0, terminal, 8)["discounted_return"])


def test_gae_terminal_no_bootstrap_absorbing_nan_and_chunk_continuation():
    reward = torch.tensor([[1., 1.], [2., 2.], [float("nan"), 3.]])
    value = torch.tensor([[10., 10.], [20., 20.], [float("nan"), 30.]])
    next_value = torch.tensor([[20., 20.], [float("nan"), 30.], [float("nan"), 40.]])
    done = torch.tensor([[False, False], [True, False], [False, False]])
    valid = torch.tensor([[True, True], [True, True], [False, True]])
    adv, returns = finite_job_gae(reward, value, next_value, done, valid, gamma=1., tau=1.)
    assert torch.equal(returns, torch.tensor([[3., 46.], [2., 45.], [0., 43.]]))
    assert torch.equal(adv, torch.tensor([[-7., 36.], [-18., 25.], [0., 13.]]))
