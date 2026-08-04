"""Runtime-independent task math used by tests and analysis tools.

The Isaac environment uses equivalent batched Torch operations in its manager
terms. Keeping these NumPy reference functions independent of Kit makes reward
and threshold behavior testable on ordinary GitHub runners.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

GRIPPER_COUPLING = np.asarray((1.0, 1.0, -1.0, 1.0, -1.0, -1.0), dtype=np.float32)
NUM_SWAP_PHASES = 8


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


def full_swap_success(
    faulty_position_error: ArrayLike,
    faulty_angle_error: ArrayLike,
    spare_lateral_error: ArrayLike,
    spare_angle_error: ArrayLike,
    spare_speed: ArrayLike,
    gripper_open: ArrayLike,
    retreat_distance: ArrayLike,
    stable_time: ArrayLike,
) -> NDArray[np.bool_]:
    """Evaluate the project's documented full-swap terminal thresholds."""

    return (
        (np.asarray(faulty_position_error) < 0.03)
        & (np.asarray(faulty_angle_error) < np.deg2rad(8.0))
        & (np.asarray(spare_lateral_error) < 0.0025)
        & (np.asarray(spare_angle_error) < np.deg2rad(3.0))
        & (np.asarray(spare_speed) < 0.02)
        & np.asarray(gripper_open, dtype=bool)
        & (np.asarray(retreat_distance) > 0.12)
        & (np.asarray(stable_time) >= 0.5)
    )


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


def advance_swap_phase(
    phase: ArrayLike,
    *,
    failed_reached: ArrayLike = False,
    gripper_closed: ArrayLike = False,
    failed_extracted: ArrayLike = False,
    failed_stowed: ArrayLike = False,
    spare_near: ArrayLike = False,
    spare_extracted: ArrayLike = False,
    spare_aligned: ArrayLike = False,
    spare_inserted: ArrayLike = False,
) -> NDArray[np.int64]:
    """Pure NumPy reference for the runtime's ordered eight-phase state machine."""

    result = np.asarray(phase, dtype=np.int64).copy()
    if np.any((result < 0) | (result >= NUM_SWAP_PHASES)):
        raise ValueError(f"phase must be in [0, {NUM_SWAP_PHASES - 1}]")

    shape = result.shape

    def condition(value: ArrayLike) -> NDArray[np.bool_]:
        return np.broadcast_to(np.asarray(value, dtype=bool), shape)

    reached = condition(failed_reached)
    closed = condition(gripper_closed)
    extracted = condition(failed_extracted)
    at_service_caddy = condition(failed_stowed)
    near_spare = condition(spare_near)
    spare_clear = condition(spare_extracted)
    aligned = condition(spare_aligned)
    inserted = condition(spare_inserted)

    # Keep this order identical to ``mdp.commands.update_swap_phase``. Sequential
    # evaluation intentionally permits a phase to advance again when the next
    # phase's completion predicate is already true in the same control step.
    result = np.where((result == 0) & reached, 1, result)
    result = np.where((result == 1) & reached & closed, 2, result)
    result = np.where((result == 2) & extracted, 3, result)
    result = np.where((result == 3) & at_service_caddy & ~closed, 4, result)
    result = np.where((result == 4) & near_spare & closed & spare_clear, 5, result)
    result = np.where((result == 5) & aligned, 6, result)
    return np.where((result == 6) & inserted, 7, result).astype(np.int64, copy=False)


def update_curriculum_stage(
    stage: int,
    completed_successes: ArrayLike,
    *,
    threshold: float = 0.70,
    window_size: int = 100,
    max_stage: int = 3,
) -> tuple[int, float, bool]:
    """Return the next curriculum stage from the latest completed episodes.

    Promotion requires a full rolling window and is inclusive at the requested
    success threshold. The rolling statistic resets to zero after promotion.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0 <= stage <= max_stage:
        raise ValueError("stage must be between zero and max_stage")
    outcomes = np.asarray(completed_successes, dtype=np.float64).reshape(-1)[-window_size:]
    if np.any((outcomes < 0.0) | (outcomes > 1.0)):
        raise ValueError("completed_successes must contain values in [0, 1]")
    rolling = float(outcomes.mean()) if outcomes.size else 0.0
    promoted = outcomes.size == window_size and rolling >= threshold and stage < max_stage
    return (stage + 1, 0.0, True) if promoted else (stage, rolling, False)
