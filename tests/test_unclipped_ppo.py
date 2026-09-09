import copy

import torch
from test_study_ppo import rollout

from assembly_recovery.study_ppo import StudyPolicy
from assembly_recovery.study_ppo import ppo_update as clipped_update
from assembly_recovery.unclipped_ppo import ppo_update


def test_value_gradient_survives_old_absolute_clip_plateau():
    torch.manual_seed(20)
    model = StudyPolicy(4, 8, 2, (8, 4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.value_head.bias.fill_(1.)
    data = rollout(model)
    data["value"].zero_()
    data["reward"].fill_(100.)
    data["done"].fill_(True)
    old = copy.deepcopy(model)
    clipped_update(old, torch.optim.SGD(old.parameters(), lr=.1), data, mini_epochs=1)
    ppo_update(model, torch.optim.SGD(model.parameters(), lr=.1), data, mini_epochs=1)
    assert old.value_head.bias.item() == 1.
    assert model.value_head.bias.item() > 1.09
    assert all(torch.equal(p, q) for p, q in zip(model.actor.parameters(), old.actor.parameters(), strict=True))


def test_unclipped_update_excludes_absorbing_nonfinite_rows():
    torch.manual_seed(7)
    first = StudyPolicy(4, 8, 2, (8, 4))
    second = copy.deepcopy(first)
    torch.manual_seed(8)
    clean = rollout(first)
    torch.manual_seed(8)
    padded = rollout(second, absorbing=True)
    torch.manual_seed(9)
    a = ppo_update(first, torch.optim.Adam(first.parameters(), lr=1e-4), clean, mini_epochs=1)
    torch.manual_seed(9)
    b = ppo_update(second, torch.optim.Adam(second.parameters(), lr=1e-4), padded, mini_epochs=1)
    assert a == b
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())


def test_unclipped_empty_rollout_skips_optimizer():
    model = StudyPolicy(4, 8, 2, (8, 4))
    data = rollout(model)
    data["valid"].zero_()
    original = copy.deepcopy(model.state_dict())
    assert ppo_update(model, torch.optim.Adam(model.parameters()), data) == {"updated": False, "active_samples": 0}
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
