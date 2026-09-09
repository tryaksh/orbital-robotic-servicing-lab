import copy

import torch

from assembly_recovery.study_ppo import RunningNorm, StudyPolicy, ppo_update


def test_privileged_critic_inputs_cannot_change_actor_or_actor_normalization():
    model = StudyPolicy(4, 8, 2, (8, 4))
    obs = {"policy": torch.randn(5, 4), "critic": torch.randn(5, 8)}
    before = model.act(obs, deterministic=True)[0]
    changed = {**obs, "critic": obs["critic"] + 100}
    assert torch.equal(before, model.act(changed, deterministic=True)[0])
    actor_params = list(model.actor.parameters()) + list(model.mean_head.parameters())
    value = model.value(model.critic_norm(changed["critic"])).sum()
    grads = torch.autograd.grad(value, actor_params, allow_unused=True)
    assert all(g is None for g in grads)


def test_normalizer_excludes_absorbing_nonfinite_values_and_empty_update():
    norm = RunningNorm(2)
    x = torch.tensor([[1., 3.], [3., 5.], [float("nan"), float("inf")]])
    norm.update(x, torch.tensor([True, True, False]))
    assert torch.allclose(norm.mean, torch.tensor([2., 4.]), atol=0.001)
    assert torch.allclose(norm.variance, torch.ones(2), atol=0.001)
    previous = copy.deepcopy(norm.state_dict())
    norm.update(x, torch.zeros(3, dtype=torch.bool))
    assert all(torch.equal(v, norm.state_dict()[k]) for k, v in previous.items())


def rollout(model, absorbing=False):
    obs = {"policy": torch.randn(3, 4), "critic": torch.randn(3, 8)}
    with torch.no_grad():
        action, log, value, normalized = model.act(obs)
    row = {"actor": normalized["policy"], "critic": normalized["critic"], "action": action,
           "raw_actor": obs["policy"], "raw_critic": obs["critic"], "log_prob": log,
           "value": value, "next_value": value, "reward": torch.tensor([1., 2., 3.]),
           "done": torch.tensor([False, True, False]), "valid": torch.ones(3, dtype=torch.bool)}
    if absorbing:
        for k, v in row.items():
            if k == "valid" or v.dtype == torch.bool:
                row[k] = torch.cat((v, torch.zeros(2, dtype=torch.bool)))
            else:
                row[k] = torch.cat((v, torch.full((2,) + v.shape[1:], float("nan"))))
    return {k: torch.stack([v, v]) for k, v in row.items()}


def test_absorbing_rows_cannot_influence_ppo_or_normalization():
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


def test_all_absorbing_rollout_skips_optimizer():
    model = StudyPolicy(4, 8, 2, (8, 4))
    data = rollout(model)
    data["valid"].zero_()
    original = copy.deepcopy(model.state_dict())
    result = ppo_update(model, torch.optim.Adam(model.parameters()), data)
    assert result == {"updated": False, "active_samples": 0}
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
