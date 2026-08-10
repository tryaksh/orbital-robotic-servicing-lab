"""Top-down grasp-frame conventions shared by the insertion tasks.

These three helpers are what is left of the eight-phase swap task's command
module, which was deleted on 2026-08-10. They survive because the insertion
family measures its grasp abstraction against the top-down Robotiq attitude, in
``insertion.py``'s :func:`secured_blade_pose_error`.

**This is not the head-on convention.** ``mdp/grapple.py`` measures a head-on
capture, which sits a quarter turn from ``GRIPPER_GRASP_ROT``. Applying
:func:`gripper_handle_orientation_error` to a head-on grip reports about
2.1 rad of error on a grip that is working exactly as designed, and that bug has
already appeared four times in this repository. Check which frame a number is in
before trusting it.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_error_magnitude, quat_mul

from ..assets import GRIPPER_GRASP_ROT


def gripper_grasp_orientation(handle_orientation: torch.Tensor) -> torch.Tensor:
    """Convert a blade/handle orientation to Isaac Lab's 2F-85 grasp frame."""

    offset = handle_orientation.new_tensor(GRIPPER_GRASP_ROT).expand_as(handle_orientation)
    return quat_mul(handle_orientation, offset)


def equivalent_gripper_orientation(handle_orientation: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return the nearest Robotiq orientation, accounting for finger symmetry.

    Rotating the 2F-85 by 180 degrees about the blade insertion axis swaps the
    two fingers but produces the same physical grasp.
    """

    flip_x = handle_orientation.new_tensor((0.0, 1.0, 0.0, 0.0)).expand_as(handle_orientation)
    primary = gripper_grasp_orientation(handle_orientation)
    flipped_handle = quat_mul(handle_orientation, flip_x)
    secondary = gripper_grasp_orientation(flipped_handle)
    use_secondary = quat_error_magnitude(reference, secondary) < quat_error_magnitude(reference, primary)
    return torch.where(use_secondary.unsqueeze(-1), secondary, primary)


def gripper_handle_orientation_error(gripper_orientation: torch.Tensor, handle_orientation: torch.Tensor) -> torch.Tensor:
    """Smallest angular error across the two equivalent Robotiq grasps."""

    equivalent = equivalent_gripper_orientation(handle_orientation, gripper_orientation)
    return quat_error_magnitude(gripper_orientation, equivalent)


__all__ = [
    "GRIPPER_GRASP_ROT",
    "equivalent_gripper_orientation",
    "gripper_grasp_orientation",
    "gripper_handle_orientation_error",
]
