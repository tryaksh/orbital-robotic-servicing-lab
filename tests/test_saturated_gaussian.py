import math

import torch

from assembly_recovery.saturated_gaussian import expected_clipped_action


def test_expected_clipped_action_matches_independent_numeric_integration():
    mean = torch.tensor([-.9, -.3, 0., .3, .9, 2.], dtype=torch.float64)
    std = torch.tensor([.1, .5, 1., .5, 1., .3], dtype=torch.float64)
    x = torch.linspace(-1, 1, 100001, dtype=torch.float64)[:, None]
    normal = torch.distributions.Normal(mean, std)
    continuous = torch.trapezoid(x * normal.log_prob(x).exp(), x, dim=0)
    expected = continuous - normal.cdf(torch.tensor(-1.)) + 1 - normal.cdf(torch.tensor(1.))
    assert torch.allclose(expected_clipped_action(mean, std), expected, atol=1e-8, rtol=0)


def test_extraction_symmetry_bounds_and_no_rng_consumption():
    mean = torch.linspace(-5, 5, 101)
    std = torch.full_like(mean, math.exp(-.05))
    before = torch.get_rng_state().clone()
    action = expected_clipped_action(mean, std)
    assert torch.equal(before, torch.get_rng_state())
    assert bool((action.abs() <= 1).all())
    assert torch.allclose(action, -expected_clipped_action(-mean, std), atol=1e-6)
    assert action[50].abs() < 1e-7
