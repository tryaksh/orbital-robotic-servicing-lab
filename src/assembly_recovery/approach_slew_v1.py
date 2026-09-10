"""CPU-testable Cartesian reference interpolation; no simulator dependency."""
import math

import torch


def slew_reference(previous, desired, *, speed_m_s=0.020, servo_dt=1 / 480):
    """Limit one XYZ reference increment by Euclidean norm.

    This limits a requested reference, not robot velocity or contact force.
    Physical position-error clamps are applied afterward by the controller.
    """
    if previous.shape != desired.shape or previous.ndim != 2 or previous.shape[-1] != 3:
        raise ValueError("Expected matching environment-by-XYZ reference tensors")
    if not math.isfinite(speed_m_s) or not math.isfinite(servo_dt) or min(speed_m_s, servo_dt) <= 0:
        raise ValueError("Reference speed and servo interval must be finite and positive")
    difference = desired - previous
    distance = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
    scale = (speed_m_s * servo_dt / distance.clamp_min(torch.finfo(distance.dtype).tiny)).clamp(max=1)
    return previous + scale * difference


def clamp_impedance_target(reference, tool_position, position_error_threshold):
    """Retain the original per-axis FORGE position-error clamp."""
    return tool_position + torch.clamp(reference - tool_position,
                                       min=-position_error_threshold, max=position_error_threshold)

