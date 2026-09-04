from __future__ import annotations

import numpy as np
import pytest

from zero_g_blade_swap.math_utils import (
    axial_stiction_force,
    exponential_distance_reward,
    exponential_moving_average,
    first_order_filter_alpha,
    gaussian_camera_noise,
    insertion_curriculum_probabilities,
    quaternion_angular_error,
    robotiq_2f85_coupled_positions,
    transform_points,
    update_curriculum_stage,
    update_reverse_station_frontier,
)


def test_distance_reward_is_monotonic() -> None:
    rewards = exponential_distance_reward([0.0, 0.05, 0.2])
    assert rewards[0] > rewards[1] > rewards[2]
    assert rewards[0] == pytest.approx(1.0)


def test_quaternion_error_handles_double_cover() -> None:
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    same_rotation = -identity
    quarter_turn = np.asarray([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]])
    assert quaternion_angular_error(identity, same_rotation)[0] == pytest.approx(0.0)
    assert quaternion_angular_error(identity, quarter_turn)[0] == pytest.approx(np.pi / 2.0)


def test_scalar_first_rigid_transform_rotates_then_translates() -> None:
    quarter_turn_z = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    transformed = transform_points([1.0, 2.0, 3.0], quarter_turn_z, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(transformed, [1.0, 3.0, 3.0], atol=1.0e-12)


def test_robotiq_coupling_matches_official_joint_signs() -> None:
    targets = robotiq_2f85_coupled_positions(np.asarray([0.4]))[0]
    np.testing.assert_allclose(targets, [0.4, 0.4, -0.4, 0.4, -0.4, -0.4])


def test_stiction_opposes_motion_and_grows_with_speed() -> None:
    forces = axial_stiction_force([-0.2, 0.0, 0.1, 0.2], 10.0, 5.0)
    assert forces[0] > 0.0
    assert forces[1] == 0.0
    assert forces[2] < 0.0
    assert abs(forces[3]) > abs(forces[2])


def test_camera_noise_statistics_and_clipping() -> None:
    source = np.full((64, 64, 3), 0.5, dtype=np.float32)
    noisy = gaussian_camera_noise(source, sigma=0.025, seed=7)
    assert noisy.min() >= 0.0
    assert noisy.max() <= 1.0
    assert float(np.mean(noisy - source)) == pytest.approx(0.0, abs=0.002)
    assert float(np.std(noisy - source)) == pytest.approx(0.025, rel=0.08)


def test_curriculum_promotes_at_70_percent_of_100_completed_episodes() -> None:
    stage, rolling, promoted = update_curriculum_stage(0, [True] * 70 + [False] * 30)
    assert (stage, rolling, promoted) == (1, 0.0, True)


def test_curriculum_requires_full_window_and_respects_threshold_and_cap() -> None:
    assert update_curriculum_stage(0, [True] * 99) == (0, 1.0, False)
    stage, rolling, promoted = update_curriculum_stage(1, [True] * 69 + [False] * 31)
    assert stage == 1 and rolling == pytest.approx(0.69) and not promoted
    assert update_curriculum_stage(3, [True] * 100) == (3, 1.0, False)


def test_curriculum_requires_practice_time_before_promotion() -> None:
    outcomes = [True] * 90 + [False] * 10
    assert update_curriculum_stage(0, outcomes, threshold=0.8, steps_elapsed=1599, minimum_steps=1600) == (
        0,
        pytest.approx(0.9),
        False,
    )
    assert update_curriculum_stage(0, outcomes, threshold=0.8, steps_elapsed=1600, minimum_steps=1600) == (1, 0.0, True)


def test_reverse_station_curriculum_moves_toward_the_rack_mouth() -> None:
    outcomes = [True] * 205 + [False] * 51
    assert update_reverse_station_frontier(6, outcomes, steps_elapsed=1_599) == (
        6,
        pytest.approx(205 / 256),
        False,
    )
    assert update_reverse_station_frontier(6, outcomes, steps_elapsed=1_600) == (5, 0.0, True)
    assert update_reverse_station_frontier(0, [True] * 256, steps_elapsed=1_600) == (0, 1.0, False)


def test_insertion_curriculum_retains_earlier_reset_stages() -> None:
    np.testing.assert_allclose(insertion_curriculum_probabilities(0), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(insertion_curriculum_probabilities(1), [0.25, 0.75, 0.0])
    np.testing.assert_allclose(insertion_curriculum_probabilities(2), [0.20, 0.20, 0.60])
    with pytest.raises(ValueError, match="locked"):
        insertion_curriculum_probabilities(1, ((1, 0, 0), (0, 1, 1), (1, 1, 1)))


def test_force_filter_alpha_matches_its_time_constant() -> None:
    control_step = 1.0 / 30.0
    alpha = first_order_filter_alpha(control_step, 0.10)
    assert alpha == pytest.approx(1.0 - np.exp(-1.0 / 3.0))
    # A step input must reach roughly 1 - 1/e of its final value after exactly
    # one time constant, which is what makes the constant physically readable.
    value = 0.0
    for _ in range(3):
        value = float(exponential_moving_average(value, 1.0, alpha))
    assert value == pytest.approx(1.0 - np.exp(-1.0), abs=0.01)
    # A shorter constant must track faster than a longer one.
    assert first_order_filter_alpha(control_step, 0.05) > alpha


def test_exponential_moving_average_is_batched_and_bounded() -> None:
    filtered = exponential_moving_average([[0.0, 4.0, -2.0]], [[10.0, 4.0, 2.0]], 0.25)
    np.testing.assert_allclose(filtered, [[2.5, 4.0, -1.0]])
    # Alpha of one is a pass-through, so the term degrades to the raw signal.
    np.testing.assert_allclose(exponential_moving_average([3.0], [-7.0], 1.0), [-7.0])


def test_invalid_physical_inputs_raise() -> None:
    with pytest.raises(ValueError):
        first_order_filter_alpha(0.0, 0.1)
    with pytest.raises(ValueError):
        first_order_filter_alpha(1.0 / 30.0, -0.1)
    with pytest.raises(ValueError):
        exponential_moving_average([0.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        exponential_moving_average([0.0], [1.0], 1.5)
    with pytest.raises(ValueError):
        exponential_distance_reward([-0.01])
    with pytest.raises(ValueError):
        robotiq_2f85_coupled_positions([1.0])
    with pytest.raises(ValueError):
        axial_stiction_force([0.1], -1.0, 2.0)
    with pytest.raises(ValueError):
        update_curriculum_stage(0, [True], window_size=0)
