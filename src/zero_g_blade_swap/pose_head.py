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

The network is small on purpose: the target is a smooth six-dimensional
function of one rigid body's pose rather than a semantic problem. Four strided
convolutions are followed by parameter-free adaptive pooling. Architecture v1
keeps the legacy 4x4 feature grid; v2 preserves an 8x8 grid for the 256x256
overview instead of averaging away the spatial precision it added.
"""

from __future__ import annotations

import hashlib
import math
import secrets
from pathlib import Path

import torch
from torch import nn

#: Position (3) and axis-angle orientation (3), all properties of the module
#: alone, so every one of them is genuinely present in the image.
MODULE_POSE_DIM = 6

#: Bays the occupancy branch reports on, when it is asked for at all.
#:
#: With one slot the only question the camera can answer is *where the module
#: is*. With two, the question a servicer actually asks first is *which bay is
#: occupied* --- and that is a different claim: reading the state of the rack
#: rather than locating a part. It is a classification, not a regression, so it
#: gets its own branch and its own loss rather than being smuggled into the pose
#: vector, where a normalised MSE would report it in millimetres.
SECOND_SLOT_OCCUPANCY_SLOTS = 2

POSE_HEAD_ARCHITECTURE_V1 = 1
POSE_HEAD_ARCHITECTURE_V2 = 2
POSE_HEAD_LEGACY_GRID_SIZE = 4
POSE_HEAD_OVERVIEW_GRID_SIZE = 8


def _architecture_version_for_grid(feature_grid_size: int) -> int:
    versions = {
        POSE_HEAD_LEGACY_GRID_SIZE: POSE_HEAD_ARCHITECTURE_V1,
        POSE_HEAD_OVERVIEW_GRID_SIZE: POSE_HEAD_ARCHITECTURE_V2,
    }
    try:
        return versions[int(feature_grid_size)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported pose-head feature grid {feature_grid_size}; expected one of {sorted(versions)}"
        ) from exc


def checkpoint_sha256(path: str | Path) -> str:
    """Hash a checkpoint for evidence/runtime identity checks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_matches_sha256(path: str | Path, recorded_sha256: str) -> bool:
    """Return whether an evidence digest identifies these exact weights."""

    normalized = recorded_sha256.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        return False
    return secrets.compare_digest(checkpoint_sha256(path), normalized)


class ModulePoseHead(nn.Module):
    """RGB to the module's six-value pose, and optionally which bay it is in.

    Label normalisation statistics are registered buffers and therefore travel
    inside the checkpoint. A head fed differently at evaluation than at training
    is the easiest way to produce a plausible number that means nothing, and
    keeping the statistics with the weights makes that impossible to get wrong.

    ``occupancy_slots`` adds a second branch off the shared trunk, producing one
    logit per bay. It defaults to zero, so a head trained before this existed
    loads and behaves exactly as it did: ``forward`` still returns the pose alone
    and every existing call site is unchanged. The occupancy is read through
    ``forward_with_occupancy`` or ``occupancy_probabilities``, which raise rather
    than invent an answer on a head that has no such branch.
    """

    def __init__(
        self,
        output_dim: int = MODULE_POSE_DIM,
        occupancy_slots: int = 0,
        feature_grid_size: int = POSE_HEAD_LEGACY_GRID_SIZE,
        architecture_version: int | None = None,
    ) -> None:
        super().__init__()
        feature_grid_size = int(feature_grid_size)
        expected_version = _architecture_version_for_grid(feature_grid_size)
        if architecture_version is None:
            architecture_version = expected_version
        if int(architecture_version) != expected_version:
            raise ValueError(
                f"pose-head architecture v{architecture_version} cannot use a {feature_grid_size}x"
                f"{feature_grid_size} feature grid; expected v{expected_version}"
            )
        self.register_buffer("_architecture_version", torch.tensor(int(architecture_version), dtype=torch.int64))
        self.register_buffer("_feature_grid_size", torch.tensor(feature_grid_size, dtype=torch.int64))
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((feature_grid_size, feature_grid_size)),
        )
        flattened_features = 128 * feature_grid_size * feature_grid_size
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )
        # Off the same trunk on purpose: "which bay holds the module" and "where
        # is the module" are the same visual evidence read two ways, and a second
        # set of convolutions would double the inference budget this project
        # measured and kept for an answer the first set already contains.
        self.occupancy_slots = int(occupancy_slots)
        self.occupancy = (
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.occupancy_slots),
            )
            if self.occupancy_slots
            else None
        )
        self.register_buffer("label_mean", torch.zeros(output_dim))
        self.register_buffer("label_std", torch.ones(output_dim))

    @property
    def architecture_version(self) -> int:
        return int(self._architecture_version)

    @property
    def feature_grid_size(self) -> int:
        return int(self._feature_grid_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image`` is (N, H, W, 3) in [0, 1]; returns (N, 6) in metres and radians."""

        normalized = self.regressor(self.features(image.permute(0, 3, 1, 2)))
        return normalized * self.label_std + self.label_mean

    def forward_with_occupancy(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pose, occupancy_logits)`` from one pass of the shared trunk."""

        if self.occupancy is None:
            raise AttributeError(
                "This ModulePoseHead was built without an occupancy branch. Train one with "
                "occupancy_slots > 0 rather than reading a bay from a head that never saw two."
            )
        features = self.features(image.permute(0, 3, 1, 2))
        pose = self.regressor(features) * self.label_std + self.label_mean
        return pose, self.occupancy(features)

    def occupancy_probabilities(self, image: torch.Tensor) -> torch.Tensor:
        """Per-bay probability that the bay holds the module."""

        return torch.sigmoid(self.forward_with_occupancy(image)[1])


def load_pose_head(checkpoint: str | Path, device: torch.device | str) -> ModulePoseHead:
    """Load a trained head, weights and label statistics together.

    Output widths and feature-grid architecture are recovered from the state
    dict rather than passed in, for the same reason the label statistics travel
    inside the checkpoint: the caller cannot then describe a head differently
    than it was trained. Metadata-free legacy weights are inferred as v1 from
    their regressor width.
    """

    state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    flattened_features = int(state["regressor.1.weight"].shape[1])
    inferred_grid = math.isqrt(flattened_features // 128)
    if 128 * inferred_grid * inferred_grid != flattened_features:
        raise ValueError(
            f"checkpoint regressor width {flattened_features} is not a square 128-channel feature grid"
        )
    recorded_grid = state.get("_feature_grid_size")
    feature_grid_size = inferred_grid if recorded_grid is None else int(recorded_grid)
    if feature_grid_size != inferred_grid:
        raise ValueError(
            f"checkpoint records feature grid {feature_grid_size}, but regressor weights require {inferred_grid}"
        )
    inferred_version = _architecture_version_for_grid(feature_grid_size)
    recorded_version = state.get("_architecture_version")
    architecture_version = inferred_version if recorded_version is None else int(recorded_version)
    if architecture_version != inferred_version:
        raise ValueError(
            f"checkpoint records architecture v{architecture_version}, but grid {feature_grid_size} "
            f"requires v{inferred_version}"
        )
    occupancy_weight = state.get("occupancy.3.weight")
    head = ModulePoseHead(
        output_dim=int(state["label_mean"].numel()),
        occupancy_slots=0 if occupancy_weight is None else int(occupancy_weight.shape[0]),
        feature_grid_size=feature_grid_size,
        architecture_version=architecture_version,
    )
    # Checkpoints created before architecture metadata existed are v1 by
    # construction. Add inferred buffers only in memory so strict loading still
    # validates every learned tensor without rewriting legacy files.
    state.setdefault("_feature_grid_size", head._feature_grid_size.clone())
    state.setdefault("_architecture_version", head._architecture_version.clone())
    if occupancy_weight is not None and int(state["occupancy.1.weight"].shape[1]) != flattened_features:
        raise ValueError("checkpoint occupancy and pose branches use different feature-grid widths")
    head.load_state_dict(state)
    head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


__all__ = [
    "MODULE_POSE_DIM",
    "POSE_HEAD_ARCHITECTURE_V1",
    "POSE_HEAD_ARCHITECTURE_V2",
    "POSE_HEAD_LEGACY_GRID_SIZE",
    "POSE_HEAD_OVERVIEW_GRID_SIZE",
    "SECOND_SLOT_OCCUPANCY_SLOTS",
    "ModulePoseHead",
    "checkpoint_matches_sha256",
    "checkpoint_sha256",
    "load_pose_head",
]
