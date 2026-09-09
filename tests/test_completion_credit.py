import pytest
import torch

from assembly_recovery.completion_credit import GAMMA, IDEAL_STEP_REWARD, completion_credit


@pytest.mark.parametrize("terminal", [1, 7, 8, 9, 330, 336, 3599, 3600])
def test_credit_exactly_replaces_only_omitted_discounted_endpoints(terminal):
    control = (terminal + 7) // 8
    value = completion_credit(torch.tensor([terminal]), torch.tensor([True]), control).item()
    actual = GAMMA ** (control - 1) * value
    expected = IDEAL_STEP_REWARD * sum(GAMMA ** k for k in range(terminal // 8, 450))
    assert actual == pytest.approx(expected, rel=2e-5, abs=2e-5)


def test_no_success_predicate_failure_or_absorbing_credit():
    value = completion_credit(torch.tensor([40, 40, 40]), torch.tensor([False, True, False]), 5)
    assert value[0] == value[2] == 0
    assert value[1] > 0
    assert completion_credit(torch.tensor([3600]), torch.tensor([True]), 450).item() == 0
