"""Reward terms that belong to the workcell rather than to one task.

The dense shaping and milestone rewards for the eight-phase swap lived here and
were deleted with that task on 2026-08-10. These two are kept because the
insertion family uses them and neither mentions a phase: one measures how far
the arm's compliant mount has been pushed off its anchor, and the other converts
any named termination into a one-time, timestep-independent penalty.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat, subtract_frame_transforms


def mount_deflection_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("mount_anchor"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    anchor = env.scene[anchor_cfg.name]
    relative_pos, relative_quat = subtract_frame_transforms(
        anchor.data.root_pos_w,
        anchor.data.root_quat_w,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )
    relative_rotation = axis_angle_from_quat(relative_quat)
    return torch.sum(torch.square(relative_pos), dim=-1) + torch.sum(torch.square(relative_rotation), dim=-1)


def undesired_termination_penalty(env, term_keys: tuple[str, ...]) -> torch.Tensor:
    """Return a one-time, timestep-independent penalty for unsafe endings."""

    failures = torch.zeros(env.num_envs, device=env.device)
    for term_name in term_keys:
        failures += env.termination_manager.get_term(term_name).to(torch.float32)
    return failures / float(env.step_dt)


__all__ = [
    "mount_deflection_penalty",
    "undesired_termination_penalty",
]
