"""The module-pose regressor, as a plain PyTorch network.

This module imports nothing from Isaac Lab, deliberately and for the same reason
``grapple_geometry.py`` does not: the head is trained offline from recorded
frames, the training script must run on a machine with no simulator, and the
test suite has to be able to reach it on every commit rather than only where
Isaac Sim is installed.

What it predicts is the **module's own pose** — position in its environment's
frame and orientation as an axis-angle vector. Not the tool-relative grip
vector, which was the first design and was wrong: the servicing camera cannot
see the gripper from its mount, so a network asked for a tool-relative quantity
would have to invent the half of the answer that forward kinematics already
supplies exactly. That error was caught by rendering a frame and projecting the
targets into it before any data was collected, which is the cheapest place to
catch it.

The network is small on purpose. A 64x64 tile is the throughput budget this
project measured and kept, and the target is a smooth six-dimensional function
of one rigid body's pose rather than a semantic problem. Four strided
convolutions take 64x64 to 4x4.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

#: Position (3) and axis-angle orientation (3), all properties of the module
#: alone, so every one of them is genuinely present in the image.
MODULE_POSE_DIM = 6


class ModulePoseHead(nn.Module):
    """64x64 RGB to the module's six-value pose.

    Label normalisation statistics are registered buffers and therefore travel
    inside the checkpoint. A head fed differently at evaluation than at training
    is the easiest way to produce a plausible number that means nothing, and
    keeping the statistics with the weights makes that impossible to get wrong.
    """

    def __init__(self, output_dim: int = MODULE_POSE_DIM) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )
        self.register_buffer("label_mean", torch.zeros(output_dim))
        self.register_buffer("label_std", torch.ones(output_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image`` is (N, H, W, 3) in [0, 1]; returns (N, 6) in metres and radians."""

        normalized = self.regressor(self.features(image.permute(0, 3, 1, 2)))
        return normalized * self.label_std + self.label_mean


def load_pose_head(checkpoint: str | Path, device: torch.device | str) -> ModulePoseHead:
    """Load a trained head, weights and label statistics together."""

    state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    head = ModulePoseHead(output_dim=int(state["label_mean"].numel()))
    head.load_state_dict(state)
    head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


__all__ = ["MODULE_POSE_DIM", "ModulePoseHead", "load_pose_head"]
