"""Read-only diagnostic sampling and exact sensor-draw capture."""
from contextlib import contextmanager

import torch


def independent_action(model, observation, generator):
    distribution = model.distribution(model.actor_norm(observation["policy"]))
    draw = torch.randn(distribution.mean.shape, device=distribution.mean.device,
                       dtype=distribution.mean.dtype, generator=generator)
    return distribution.mean + distribution.stddev * draw, distribution.mean


@contextmanager
def capture_normal_draws(records):
    """Capture the unchanged upstream torch.randn draws during env.step only.

    Clone before the caller can modify a draw in place. Never reseed, replace
    values, or sample additional noise. The caller excludes policy sampling.
    """
    original = torch.randn

    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        records.append(value.detach().clone())
        return value

    torch.randn = record
    try:
        yield
    finally:
        torch.randn = original
