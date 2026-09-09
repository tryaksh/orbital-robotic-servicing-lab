"""Finite action preparation shared by project training/evaluation drivers."""
from __future__ import annotations

import torch


def prepare_actions(requested, previous, active):
    """Discard arbitrary absorbing-slot outputs before any shared reward term.

    A nonfinite live request is replaced by the previous safe command and must
    terminate that job at its next physics sample via the returned live-input validity mask. Finite absorbing commands retain
    the reference convention; their servo targets and rewards are still held/masked.
    """
    finite = torch.isfinite(requested).all(-1)
    safe = torch.where(finite[:, None], requested.clamp(-1, 1), previous)
    return safe, finite | ~active


def bounded_initial_actions(actions):
    """Project finite initial EMA history into the legal action box.

    A noisy target frame can place a physically valid initial pose just outside
    the representable no-motion action. Preserve nonfinite values for rejection.
    This changes action history only; it never writes simulator state.
    """
    return torch.where(torch.isfinite(actions), actions.clamp(-1, 1), actions)
