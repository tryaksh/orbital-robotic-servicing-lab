"""Runtime-independent task math used by tests and analysis tools.

The Isaac environment uses equivalent batched Torch operations in its manager
terms. Keeping these NumPy reference functions independent of Kit makes reward
and threshold behavior testable on ordinary GitHub runners.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

GRIPPER_COUPLING = np.asarray((1.0, 1.0, -1.0, 1.0, -1.0, -1.0), dtype=np.float32)
INSERTION_CURRICULUM_MIXTURES = (
    (1.00, 0.00, 0.00),
    (0.25, 0.75, 0.00),
    (0.20, 0.20, 0.60),
)


def transform_points(
    position: ArrayLike,
    quaternion: ArrayLike,
    points: ArrayLike,
) -> NDArray[np.float64]:
    """Apply scalar-first quaternion rotation and translation to 3-D points."""

    translation = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(quaternion, dtype=np.float64)
    source = np.asarray(points, dtype=np.float64)
    if translation.shape[-1] != 3 or source.shape[-1] != 3 or rotation.shape[-1] != 4:
        raise ValueError("position/points must end in 3 values and quaternion in 4")
    norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("quaternion must have non-zero magnitude")
    rotation = rotation / norm
    vector = rotation[..., 1:]
    twice_cross = 2.0 * np.cross(vector, source)
    rotated = source + rotation[..., :1] * twice_cross + np.cross(vector, twice_cross)
    return translation + rotated


def exponential_distance_reward(distance: ArrayLike, scale: float = 10.0) -> NDArray[np.float64]:
    """Return ``exp(-scale * distance)`` for non-negative metric distance."""

    values = np.asarray(distance, dtype=np.float64)
    if np.any(values < 0.0):
        raise ValueError("distance must be non-negative")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return np.exp(-scale * values)


def quaternion_angular_error(q_a: ArrayLike, q_b: ArrayLike) -> NDArray[np.float64]:
    """Shortest angular distance in radians for scalar-first unit quaternions."""

    first = np.asarray(q_a, dtype=np.float64)
    second = np.asarray(q_b, dtype=np.float64)
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("quaternion arrays must have identical (..., 4) shapes")
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1.0e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1.0e-12)
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def robotiq_2f85_coupled_positions(command: ArrayLike) -> NDArray[np.float32]:
    """Map a scalar 2F-85 closure command to its six coupled joint targets."""

    value = np.asarray(command, dtype=np.float32)
    if np.any((value < 0.0) | (value > 0.8)):
        raise ValueError("2F-85 command must be in [0.0, 0.8] radians")
    return value[..., None] * GRIPPER_COUPLING


def axial_stiction_force(
    axial_velocity: ArrayLike,
    breakaway_force: ArrayLike,
    viscous_drag: ArrayLike,
    velocity_epsilon: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Opposing axial guide force once the blade is moving inside a slot."""

    velocity = np.asarray(axial_velocity, dtype=np.float64)
    breakaway = np.asarray(breakaway_force, dtype=np.float64)
    viscous = np.asarray(viscous_drag, dtype=np.float64)
    if np.any(breakaway < 0.0) or np.any(viscous < 0.0):
        raise ValueError("stiction parameters must be non-negative")
    moving = np.abs(velocity) > velocity_epsilon
    magnitude = breakaway + viscous * np.abs(velocity)
    return np.where(moving, -np.sign(velocity) * magnitude, 0.0)


def first_order_filter_alpha(time_step_s: float, time_constant_s: float) -> float:
    """Return the exponential-filter coefficient for one control step.

    A real wrist force/torque signal is low-pass filtered before a controller
    ever sees it, so the policy's force observation is filtered too. Expressing
    the coefficient as a time constant keeps it meaningful if the control rate
    changes, which a raw per-step alpha would not.
    """

    if time_step_s <= 0.0 or time_constant_s <= 0.0:
        raise ValueError("time_step_s and time_constant_s must be positive")
    return float(1.0 - np.exp(-time_step_s / time_constant_s))


def exponential_moving_average(previous: ArrayLike, sample: ArrayLike, alpha: float) -> NDArray[np.float64]:
    """Blend one new sample into a running exponential average."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    history = np.asarray(previous, dtype=np.float64)
    latest = np.asarray(sample, dtype=np.float64)
    return history + alpha * (latest - history)


def gaussian_camera_noise(
    image: ArrayLike,
    sigma: float = 0.025,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Add reproducible Gaussian radiation noise and clamp normalized RGB."""

    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    source = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noisy = source + rng.normal(0.0, sigma, size=source.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def update_curriculum_stage(
    stage: int,
    completed_successes: ArrayLike,
    *,
    threshold: float = 0.70,
    window_size: int = 100,
    max_stage: int = 3,
    steps_elapsed: int = 0,
    minimum_steps: int = 0,
) -> tuple[int, float, bool]:
    """Return the next curriculum stage from the latest completed episodes.

    Promotion requires a full rolling window and is inclusive at the requested
    success threshold. The rolling statistic resets to zero after promotion.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if steps_elapsed < 0 or minimum_steps < 0:
        raise ValueError("steps_elapsed and minimum_steps must be non-negative")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0 <= stage <= max_stage:
        raise ValueError("stage must be between zero and max_stage")
    outcomes = np.asarray(completed_successes, dtype=np.float64).reshape(-1)[-window_size:]
    if np.any((outcomes < 0.0) | (outcomes > 1.0)):
        raise ValueError("completed_successes must contain values in [0, 1]")
    rolling = float(outcomes.mean()) if outcomes.size else 0.0
    promoted = (
        outcomes.size == window_size and rolling >= threshold and steps_elapsed >= minimum_steps and stage < max_stage
    )
    return (stage + 1, 0.0, True) if promoted else (stage, rolling, False)


def insertion_curriculum_probabilities(
    level: int,
    mixtures: ArrayLike = INSERTION_CURRICULUM_MIXTURES,
) -> NDArray[np.float64]:
    """Return a validated reset-stage distribution for a curriculum level.

    A level may retain easier reset stages, but it may not sample a harder stage
    that has not yet been unlocked.
    """

    weights = np.asarray(mixtures, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("curriculum mixtures must be a square level-by-stage matrix")
    if not 0 <= level < weights.shape[0]:
        raise ValueError("curriculum level is outside the mixture matrix")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("curriculum mixture weights must be finite and non-negative")
    selected = weights[level]
    if selected[level + 1 :].sum() > 0.0:
        raise ValueError("a curriculum level cannot sample locked harder stages")
    total = float(selected.sum())
    if total <= 0.0 or selected[level] <= 0.0:
        raise ValueError("a curriculum level must sample itself with positive weight")
    return selected / total
