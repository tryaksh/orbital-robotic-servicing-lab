"""Guard recovery exposure against evaluation leakage and off-policy credit."""
import random

import pytest
import torch

from assembly_recovery.recovery_teaching_v1 import (
    exposure_mask,
    mask_scripted_rollout,
    policy_transition_valid,
    scripted_prefix_mask,
)
from assembly_recovery.tensor_jobs import finite_job_gae
from assembly_recovery.training_cases import uniform_training_cases


def cases_for_bins(counts):
    return [{"case_id": f"{bin_id}-{i}", "seed": 170, "split": "training", "bin_id": bin_id,
             "family": "nominal" if bin_id == "nominal" else "approach_offset"}
            for bin_id, count in counts.items() for i in range(count)]


def choose(cases, **kwargs):
    return exposure_mask(cases, seed=170, cohort=0, training_seeds=[170, 271, 372], **kwargs)


def test_stratified_half_faults_preserves_nominal_and_rng():
    cases = cases_for_bins({"nominal": 256, **{f"fault-{i}": 128 for i in range(6)}})
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    mask = choose(cases)
    assert sum(mask) == 384
    assert not any(mask[:256])
    assert [sum(mask[256 + i * 128:256 + (i + 1) * 128]) for i in range(6)] == [64] * 6
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert mask == choose(cases)
    selected_ids = {case["case_id"] for case, selected in zip(cases, mask, strict=True) if selected}
    reverse_ids = {case["case_id"] for case, selected in zip(reversed(cases), choose(list(reversed(cases))), strict=True) if selected}
    assert selected_ids == reverse_ids
    assert mask != exposure_mask(cases, seed=170, cohort=1, training_seeds=[170])


def test_odd_quotas_and_empty_fault_support():
    cases = cases_for_bins({"nominal": 2, "a": 3, "b": 3, "c": 1})
    assert sum(choose(cases)) == 4
    assert choose(cases_for_bins({"nominal": 4})) == [False] * 4
    assert not any(choose(cases, fraction=0))
    assert choose(cases, fraction=1) == [False] * 2 + [True] * 7


@pytest.mark.parametrize("mutation", [
    {"split": "development"}, {"split": "test"}, {"seed": 10070},
    {"case_id": ""}, {"bin_id": None}, {"family": ""},
])
def test_exposure_rejects_nontraining_or_malformed_cases(mutation):
    cases = cases_for_bins({"a": 1})
    cases[0].update(mutation)
    with pytest.raises(ValueError):
        choose(cases)


def test_exposure_rejects_undeclared_seed_and_duplicates():
    cases = cases_for_bins({"a": 2})
    with pytest.raises(ValueError, match="declared training seed"):
        exposure_mask(cases, seed=10070, cohort=0, training_seeds=[170])
    with pytest.raises(ValueError, match="unique"):
        choose([cases[0], cases[0]])


@pytest.mark.parametrize("fraction", [-1, 1.1, float("nan"), float("inf"), True])
def test_bad_exposure_fraction(fraction):
    with pytest.raises(ValueError):
        choose(cases_for_bins({"a": 2}), fraction=fraction)


def test_handoff_and_terminal_action_ownership():
    exposure = torch.tensor([True, False, True])
    assert torch.equal(scripted_prefix_mask(exposure, step=59, prefix_controls=60), exposure)
    assert not scripted_prefix_mask(exposure, step=60, prefix_controls=60).any()
    active = torch.tensor([True, True, False])
    assert policy_transition_valid(active, exposure).tolist() == [False, True, False]
    assert policy_transition_valid(active, torch.zeros_like(exposure)).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="match exactly"):
        policy_transition_valid(active[:, None], exposure)
    with pytest.raises(ValueError, match="boolean"):
        policy_transition_valid(active.float(), exposure)


def test_prefix_rewards_cannot_credit_learned_actions_or_script_completions():
    active = torch.tensor([[True, True], [True, False], [True, False], [True, False]])
    scripted = torch.tensor([[True, True], [True, True], [False, False], [False, False]])
    reward = torch.tensor([[1e9, 1e9], [1e9, 0.0], [1.0, 0.0], [10.0, 0.0]])
    done = torch.tensor([[False, True], [False, False], [False, False], [True, False]])
    original = {"valid": active, "reward": reward}
    result = mask_scripted_rollout(original, scripted)
    zeros = torch.zeros_like(reward)
    adv, returns = finite_job_gae(reward, zeros, zeros, done, result["valid"])
    _, suffix_returns = finite_job_gae(reward[2:, :1], zeros[2:, :1], zeros[2:, :1], done[2:, :1], active[2:, :1])
    assert not returns[:2].any()
    assert not returns[:, 1].any()
    assert torch.equal(returns[2:, :1], suffix_returns)
    assert torch.isfinite(adv).all()
    assert result["reward"] is reward
    assert original["valid"] is active
    assert active.sum() == 5


def test_later_script_intervention_is_rejected():
    active = torch.ones((3, 1), dtype=torch.bool)
    scripted = torch.tensor([[True], [False], [True]])
    with pytest.raises(ValueError, match="precede learned"):
        mask_scripted_rollout({"valid": active}, scripted)


def test_ppo_ignores_prefix_in_losses_and_normalizers():
    from copy import deepcopy

    from assembly_recovery.study_ppo import StudyPolicy
    from assembly_recovery.unclipped_ppo import ppo_update

    torch.manual_seed(981)
    model = StudyPolicy(3, 4, action_size=2, widths=(8,))
    raw = {'policy': torch.randn(4, 2, 3), 'critic': torch.randn(4, 2, 4)}
    with torch.no_grad():
        action, log_prob, value, normalized = model.act(raw)
    rollout = {'raw_actor': raw['policy'], 'raw_critic': raw['critic'],
               'actor': normalized['policy'], 'critic': normalized['critic'],
               'action': action, 'log_prob': log_prob, 'value': value,
               'next_value': value + 1, 'reward': torch.arange(8).reshape(4, 2).float(),
               'valid': torch.ones(4, 2, dtype=torch.bool),
               'done': torch.tensor([[False, False], [False, False], [False, False], [True, True]])}
    reference_rollout = {key: tensor[2:].clone() for key, tensor in rollout.items()}
    # Prefix exclusion must cover the critic and statistics, not just ratios.
    for tensor in rollout.values():
        if tensor.is_floating_point():
            tensor[:2] = float('nan')
    scripted = torch.tensor([[True, True], [True, True], [False, False], [False, False]])
    masked = mask_scripted_rollout(rollout, scripted)
    reference = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4)
    torch.manual_seed(718)
    metric = ppo_update(model, optimizer, masked, mini_epochs=2, minibatch=3)
    torch.manual_seed(718)
    reference_metric = ppo_update(reference, reference_optimizer, reference_rollout, mini_epochs=2, minibatch=3)
    assert metric == reference_metric
    assert metric['active_samples'] == 4
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in model.state_dict().items())


def test_selector_accepts_actual_source_only_training_cases():
    import json
    from pathlib import Path

    study = json.loads((Path(__file__).resolve().parents[1] / "configs/study.json").read_text())
    cases = uniform_training_cases(study, 170, 2, 1024)
    mask = exposure_mask(cases, seed=170, cohort=2, training_seeds=study["training"]["seeds"])
    assert sum(mask) == 384
    assert sum(case["family"] == "nominal" for case in cases) == 256
    assert all(not chosen for case, chosen in zip(cases, mask, strict=True) if case["family"] == "nominal")


@pytest.mark.parametrize("retry", [False, True])
def test_vectorized_prefix_matches_frozen_controller_for_every_control(retry):
    from assembly_recovery.recovery_teaching_v1 import make_prefix_action_targets, prefix_action_at_step
    from assembly_recovery.retry_controller import ActorRetryController, RetrySettings

    generator = torch.Generator().manual_seed(529)
    initial = torch.randn((9, 24), generator=generator)
    bounds, seated_height, dt = [0.05, 0.05, 0.05], 0.010531, 1 / 15
    hold, insert = make_prefix_action_targets(initial, position_bounds=bounds, seated_height_m=seated_height)
    controllers = [ActorRetryController(row, step_dt=dt, position_bounds=bounds,
                   seated_height_m=seated_height, retry=retry, settings=RetrySettings())
                   for row in initial.tolist()]
    for step in range(60):
        # Strongly changing forces/poses alter witness history but cannot alter
        # the scripted first-attempt actions before the frozen handoff.
        observation = torch.randn((9, 24), generator=generator) * 20
        expected = torch.tensor([controller.act(row, step * dt)[0]
                                 for controller, row in zip(controllers, observation.tolist(), strict=True)])
        actual = prefix_action_at_step(hold, insert, step=step, step_dt=dt)
        assert torch.equal(actual, expected)
    assert torch.equal(hold, initial[:, -7:].clamp(-1, 1))
    with pytest.raises(ValueError, match="after handoff"):
        prefix_action_at_step(hold, insert, step=60, step_dt=dt)
