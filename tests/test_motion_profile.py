"""Setpoint timing bounds; physical tracking is tested in recorded missions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from zero_g_blade_swap.motion_profile import (
    DEFAULT_MOTION_LIMITS,
    QUINTIC_PEAK_ACCELERATION,
    QUINTIC_PEAK_VELOCITY,
    MotionLimits,
    minimum_jerk_acceleration,
    minimum_jerk_blend,
    minimum_jerk_velocity,
    profile_duration,
)


def test_stationary_endpoints_and_monotonic_progress():
    progress = np.linspace(-0.1, 1.1, 10001)
    position = minimum_jerk_blend(progress)
    assert position[0] == 0.0
    assert position[-1] == 1.0
    assert np.all(np.diff(position) >= -1e-14)
    assert np.all((position >= 0.0) & (position <= 1.0))
    for endpoint in (-1.0, 0.0, 1.0, 2.0):
        assert minimum_jerk_velocity(endpoint) == 0.0
        assert minimum_jerk_acceleration(endpoint) == 0.0
    assert isinstance(minimum_jerk_blend(0.5), float)
    assert minimum_jerk_blend(0.5) == pytest.approx(0.5)


def test_reported_derivatives_agree_with_position_finite_differences():
    progress = np.linspace(0.02, 0.98, 101)
    step = 1e-4
    previous = minimum_jerk_blend(progress - step)
    current = minimum_jerk_blend(progress)
    following = minimum_jerk_blend(progress + step)
    np.testing.assert_allclose(
        (following - previous) / (2.0 * step), minimum_jerk_velocity(progress), atol=1e-7
    )
    np.testing.assert_allclose(
        (following - 2.0 * current + previous) / step**2,
        minimum_jerk_acceleration(progress),
        atol=5e-7,
    )


def test_exact_extrema_determine_bounds():
    assert minimum_jerk_velocity(0.5) == pytest.approx(QUINTIC_PEAK_VELOCITY)
    extrema = np.array([(3.0 - math.sqrt(3.0)) / 6.0, (3.0 + math.sqrt(3.0)) / 6.0])
    np.testing.assert_allclose(
        minimum_jerk_acceleration(extrema), [QUINTIC_PEAK_ACCELERATION, -QUINTIC_PEAK_ACCELERATION]
    )


@pytest.mark.parametrize(
    ("translation", "rotation"),
    [
        ((0.245, 0.0, 0.0), 0.0),  # Rail-length translation, velocity limited.
        ((0.0001, 0.0, 0.0), 0.0),  # Short move, acceleration limited.
        ((0.0, 0.0, 0.0), math.pi / 2.0),  # Pure rotation.
        ((0.0, 0.0, 0.0), 0.001),  # Short angular acceleration limited move.
        ((0.05, -0.03, 0.04), 0.214),  # Simultaneous source retreat and squaring.
        ((0.4, -0.3, 0.2), 0.4),  # Diagonal speed must be Euclidean, not per axis.
    ],
)
def test_sampled_cartesian_trajectory_obeys_all_bounds(translation, rotation):
    displacement = np.asarray(translation)
    duration = profile_duration(float(np.linalg.norm(displacement)), rotation)
    time = np.linspace(0.0, duration, 10001)
    blend = minimum_jerk_blend(time / duration)
    position = blend[:, None] * displacement
    angle = blend * rotation
    step = time[1] - time[0]
    linear_velocity = np.diff(position, axis=0) / step
    linear_acceleration = np.diff(linear_velocity, axis=0) / step
    angular_velocity = np.diff(angle) / step
    angular_acceleration = np.diff(angular_velocity) / step
    limits = DEFAULT_MOTION_LIMITS
    fractions = [
        np.linalg.norm(linear_velocity, axis=1).max() / limits.linear_velocity_mps,
        np.linalg.norm(linear_acceleration, axis=1).max() / limits.linear_acceleration_mps2,
        np.abs(angular_velocity).max() / limits.angular_velocity_radps,
        np.abs(angular_acceleration).max() / limits.angular_acceleration_radps2,
    ]
    assert max(fractions) <= 1.0 + 1e-6
    # The duration is minimal for this polynomial: at least one bound is tight.
    assert max(fractions) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(position[-1], displacement)
    assert angle[-1] == pytest.approx(rotation)


def test_zero_move_and_known_rail_duration():
    assert profile_duration(0.0, 0.0) == 0.0
    assert profile_duration(0.245) == pytest.approx(4.59375)
    assert profile_duration(0.0, 0.1) > 0.0


@pytest.mark.parametrize("field", MotionLimits.__dataclass_fields__)
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_invalid_limits_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        MotionLimits(**{field: value})


@pytest.mark.parametrize("value", [-1.0, math.inf, -math.inf, math.nan])
def test_invalid_travel_is_rejected(value):
    with pytest.raises(ValueError, match="translation_distance_m"):
        profile_duration(value, 0.0)
    with pytest.raises(ValueError, match="rotation_angle_rad"):
        profile_duration(0.0, value)


@pytest.mark.parametrize("function", [minimum_jerk_blend, minimum_jerk_velocity, minimum_jerk_acceleration])
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_progress_is_rejected(function, value):
    with pytest.raises(ValueError, match="progress"):
        function([0.0, value, 1.0])


def test_unrepresentable_duration_is_rejected():
    with pytest.raises(ValueError, match="nonfinite duration"):
        profile_duration(1e308)
