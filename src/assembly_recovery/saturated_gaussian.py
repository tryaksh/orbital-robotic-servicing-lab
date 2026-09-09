"""Read-only Gaussian action extraction for a registered development diagnostic."""
import math

import torch


def expected_clipped_action(mean, std):
    """Return E[clip(N(mean, std), -1, 1)] using only actor distribution values.

    This is a distinct deterministic diagnostic controller. It is not an
    estimate of expected motion through the nonlinear simulator or a claim
    that this extraction outperforms the registered Gaussian-mean controller.
    """
    lower, upper = (-1 - mean) / std, (1 - mean) / std
    cdf_lower = .5 * (1 + torch.erf(lower / math.sqrt(2)))
    cdf_upper = .5 * (1 + torch.erf(upper / math.sqrt(2)))
    density_lower = torch.exp(-.5 * lower.square()) / math.sqrt(2 * math.pi)
    density_upper = torch.exp(-.5 * upper.square()) / math.sqrt(2 * math.pi)
    result = mean * (cdf_upper - cdf_lower) + std * (density_lower - density_upper) - cdf_lower + 1 - cdf_upper
    return result.clamp(-1, 1)
