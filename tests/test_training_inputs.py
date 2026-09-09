import torch

from assembly_recovery.training_inputs import prepare_actions


def test_absorbing_nan_actions_cannot_pollute_global_zero_scaled_reward_term():
    previous = torch.zeros(2, 7)
    requested = torch.tensor([[2.] * 7, [float("nan")] * 7])
    safe, finite = prepare_actions(requested, previous, torch.tensor([True, False]))
    assert torch.equal(safe, torch.tensor([[1.] * 7, [0.] * 7]))
    assert torch.isfinite(torch.norm(safe) * 0.)
    assert finite.tolist() == [True, True]


def test_nonfinite_live_request_is_held_and_flagged_for_failure():
    previous = torch.full((1, 7), 0.2)
    requested = torch.full((1, 7), float("inf"))
    safe, finite = prepare_actions(requested, previous, torch.tensor([True]))
    assert torch.equal(safe, previous)
    assert not finite.item()


def test_initial_action_projection_preserves_in_range_and_rejectable_nonfinite():
    from assembly_recovery.training_inputs import bounded_initial_actions

    actions = torch.tensor([[-1., -0.5, 0., 1., 1.01111054, float("nan"), float("inf")]])
    actual = bounded_initial_actions(actions)
    assert torch.equal(actual[:, :4], actions[:, :4])
    assert actual[0, 4] == 1
    assert torch.isnan(actual[0, 5]) and torch.isinf(actual[0, 6])
    assert actions[0, 4] > 1  # no mutation of the recorded preprojection history
