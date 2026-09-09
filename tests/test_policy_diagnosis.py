import torch

from assembly_recovery.policy_diagnosis import capture_normal_draws, independent_action
from assembly_recovery.study_ppo import StudyPolicy


def test_policy_rng_is_independent_and_reproducible():
    torch.manual_seed(72)
    model = StudyPolicy(3, 5, widths=(8,))
    observation = {"policy": torch.randn(4, 3), "critic": torch.randn(4, 5)}
    before = torch.get_rng_state().clone()
    action, mean = independent_action(model, observation, torch.Generator().manual_seed(123))
    assert torch.equal(before, torch.get_rng_state())
    again, _ = independent_action(model, observation, torch.Generator().manual_seed(123))
    assert torch.equal(action, again)
    assert torch.equal(mean, model.act(observation, deterministic=True)[0])
    assert not torch.equal(action, mean)


def test_sensor_capture_does_not_change_rng_or_mutate_saved_draws():
    torch.manual_seed(19)
    expected = torch.randn(3, 4)
    tail = torch.get_rng_state().clone()
    torch.manual_seed(19)
    records = []
    original = torch.randn
    with capture_normal_draws(records):
        actual = torch.randn(3, 4)
        actual.zero_()
    assert torch.equal(records[0], expected)
    assert torch.equal(tail, torch.get_rng_state())
    assert torch.randn is original
