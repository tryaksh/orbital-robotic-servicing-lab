"""Finite-horizon completion credit, derived from the frozen reward upper bound."""
import torch

GAMMA = 0.995
IDEAL_STEP_REWARD = 35 / 12  # 1/6 + 1/4 + 1/2 proximity, engagement, upstream success.


def completion_credit(terminal_step, newly_successful, control_step, *, controls=450, decimation=8):
    """Credit ideal remaining endpoints once, at the completed dwell notification.

    The original finite reward omits endpoints after native termination. Restore
    their discounted ideal-success value algebraically, without simulated
    continuation, resets, or bootstrapping. No credit for an upstream success
    predicate alone, force/grasp failure, or an absorbing slot.
    """
    included = torch.div(terminal_step, decimation, rounding_mode="floor")
    remaining = (controls - included).clamp_min(0)
    # Aligned termination: first omitted endpoint is one control later.
    # Partial termination: notification endpoint itself was omitted upstream.
    offset = (included - (control_step - 1)).clamp(0, 1)
    credit = IDEAL_STEP_REWARD * GAMMA ** offset * (1 - GAMMA ** remaining) / (1 - GAMMA)
    return torch.where(newly_successful, credit, 0.)
