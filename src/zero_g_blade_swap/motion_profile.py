"""Simulator-independent minimum-jerk setpoint timing.

One scalar blend drives both a straight translation and the shortest rotation
between two poses. The duration bounds the *setpoint's* Euclidean linear speed,
linear acceleration, angular speed and angular acceleration. It does not bound
tracking error, joint acceleration, contact force, or hardware loading; those
need separate measurements of the robot that executes the trajectory.

The defaults are initial engineering limits for a simulation comparison, not
qualified hardware limits. This module requires neither Isaac Sim nor Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Exact maxima of ds/du and abs(d2s/du2) for s(u) = 10u^3 - 15u^4 + 6u^5.
QUINTIC_PEAK_VELOCITY = 15.0 / 8.0
QUINTIC_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)


@dataclass(frozen=True, slots=True)
class MotionLimits:
    """Positive finite Cartesian setpoint limits, in SI units."""

    linear_velocity_mps: float = 0.10
    linear_acceleration_mps2: float = 0.25
    angular_velocity_radps: float = 0.30
    angular_acceleration_radps2: float = 0.80

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")


DEFAULT_MOTION_LIMITS = MotionLimits()


def _progress(progress: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(progress, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("progress must be finite")
    return np.clip(value, 0.0, 1.0)


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    return float(value) if value.ndim == 0 else value


def minimum_jerk_blend(progress: ArrayLike) -> float | NDArray[np.float64]:
    """Return s(u), holding at 0 before the motion and 1 after it.

    ``progress`` is elapsed time divided by duration. Endpoint velocity and
    acceleration are zero, so joining this profile to a stationary hold is C2
    continuous. Jerk is finite within the move but not continuous at the holds.
    """

    u = _progress(progress)
    return _scalar_or_array(u**3 * (10.0 + u * (-15.0 + 6.0 * u)))


def minimum_jerk_velocity(progress: ArrayLike) -> float | NDArray[np.float64]:
    """Return dimensionless ds/du; multiply by travel / duration for speed."""

    u = _progress(progress)
    return _scalar_or_array(30.0 * u**2 * (1.0 - u)**2)


def minimum_jerk_acceleration(progress: ArrayLike) -> float | NDArray[np.float64]:
    """Return dimensionless d2s/du2; scale by travel / duration squared."""

    u = _progress(progress)
    return _scalar_or_array(60.0 * u * (1.0 - u) * (1.0 - 2.0 * u))


def profile_duration(
    translation_distance_m: float,
    rotation_angle_rad: float = 0.0,
    limits: MotionLimits = DEFAULT_MOTION_LIMITS,
) -> float:
    """Minimum duration satisfying all four limits for this quintic profile.

    Pass the Euclidean translation distance and a nonnegative rotation angle
    along the chosen arc, normally the shortest quaternion arc. Both motions
    use the same blend and arrive together. A zero-distance, zero-angle move
    returns 0; callers should hold the existing pose without dividing by it.
    """

    for name, value in (
        ("translation_distance_m", translation_distance_m),
        ("rotation_angle_rad", rotation_angle_rad),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")

    duration = max(
        QUINTIC_PEAK_VELOCITY * translation_distance_m / limits.linear_velocity_mps,
        math.sqrt(QUINTIC_PEAK_ACCELERATION * translation_distance_m / limits.linear_acceleration_mps2),
        QUINTIC_PEAK_VELOCITY * rotation_angle_rad / limits.angular_velocity_radps,
        math.sqrt(QUINTIC_PEAK_ACCELERATION * rotation_angle_rad / limits.angular_acceleration_radps2),
    )
    if not math.isfinite(duration):
        raise ValueError("motion and limits produce a nonfinite duration")
    return duration


JOINT_TRIM_INTEGRAL_GAIN_PER_S = 1.5
JOINT_TRIM_RATE_LIMIT_RAD_PER_S = 0.03
JOINT_TRIM_BIAS_LIMIT_RAD = 0.06


def update_joint_trim(error, bias, dt: float, active):
    """Return the next bias; inactive entries keep their previous value.

    ``error`` and ``bias`` have identical (..., joints) shapes and contain finite
    floating-point values. ``active`` is a boolean scalar, a per-joint mask, or
    a per-environment mask matching the leading array dimensions. For Torch,
    array arguments share a device. No input is mutated or converted to NumPy.

    Rate limiting acts before the bias cap. Clamping the integrator state itself
    avoids accumulating hidden windup while a sustained error saturates the trim.
    This trim cannot eliminate offsets requiring more torque than its bias cap
    provides; that remains a physical controller limitation.
    """
    seconds = float(dt)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Joint-trim dt must be positive and finite")
    if len(error.shape) < 1 or error.shape != bias.shape:
        raise ValueError("Joint-trim error and bias must have identical (..., joints) shapes")
    if getattr(active, "shape", None) == bias.shape[:-1] and len(bias.shape) > 1:
        active = active[..., None]
    max_step = JOINT_TRIM_RATE_LIMIT_RAD_PER_S * seconds
    increment = (error * (JOINT_TRIM_INTEGRAL_GAIN_PER_S * seconds)).clip(-max_step, max_step)
    candidate = (bias + increment).clip(-JOINT_TRIM_BIAS_LIMIT_RAD, JOINT_TRIM_BIAS_LIMIT_RAD)
    return bias + active * (candidate - bias)


