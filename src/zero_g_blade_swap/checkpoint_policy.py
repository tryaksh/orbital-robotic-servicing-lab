"""One RL-Games actor, loaded without RL-Games.

Lifted out of ``scripts/run_workflow_demo.py`` unchanged, because a second
caller now needs it: the chained-insert training task runs the frozen capture
policy *inside* the environment. Two copies of a policy loader would be two
chances to disagree about observation clipping or normaliser handling, and this
repository has paid for restated constants often enough.

The policies are loaded straight from their checkpoints rather than through
RL-Games, because several players in one process would each need their own
vector environment. The network is a three-layer MLP and its observation
normaliser is in the same file, so running it directly is both simpler and
easier to audit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


class CheckpointPolicy:
    """One RL-Games actor, loaded without RL-Games.

    Reproduces exactly what a deterministic player does: clip the observation to
    the configured range, apply the running mean/variance normaliser the policy
    was trained with, run the trunk, and take the mean action.
    """

    def __init__(self, path: Path, device: str, clip_observations: float = 10.0) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        weights = checkpoint["model"]
        self.device = device
        self.clip_observations = clip_observations
        self.mean = weights["running_mean_std.running_mean"].to(device).float()
        self.variance = weights["running_mean_std.running_var"].to(device).float()
        self.observation_dim = int(self.mean.shape[0])

        layers: list[nn.Module] = []
        index = 0
        while f"a2c_network.trunk.{index}.weight" in weights:
            weight = weights[f"a2c_network.trunk.{index}.weight"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(weights[f"a2c_network.trunk.{index}.bias"])
            layers.extend((layer, nn.ELU()))
            index += 2
        self.trunk = nn.Sequential(*layers).to(device).eval()

        mu_weight = weights["a2c_network.mu.weight"]
        self.mu = nn.Linear(mu_weight.shape[1], mu_weight.shape[0]).to(device)
        self.mu.weight.data.copy_(mu_weight)
        self.mu.bias.data.copy_(weights["a2c_network.mu.bias"])
        self.mu.eval()
        self.action_dim = int(mu_weight.shape[0])
        self.epoch = int(checkpoint.get("epoch", -1))
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest().upper()

    @torch.inference_mode()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise RuntimeError(
                f"{self.path.name} expects {self.observation_dim} observation values, "
                f"received {observation.shape[-1]}. The observation group does not match the policy."
            )
        clipped = observation.clamp(-self.clip_observations, self.clip_observations)
        normalized = ((clipped - self.mean) / torch.sqrt(self.variance + 1.0e-5)).clamp(-5.0, 5.0)
        return self.mu(self.trunk(normalized)).clamp(-1.0, 1.0)


__all__ = ["CheckpointPolicy"]
