"""Causal servo verification must retain force/RNG gates and detect actual cadence."""
import copy

import pytest
import torch

from scripts import review_servo_impact_v1 as review


def cadence_fixture(refinement=8):
    steps, stride = 480 * refinement, refinement // 4
    ticks = torch.arange(steps) % stride == 0
    torques = torch.arange(1920).float().repeat_interleave(stride)[:, None]
    data = {"native": {
        "torques": torques, "gripper_targets": torques.clone(),
        "servo_tick": ticks, "servo_updates_completed": ticks.double().cumsum(0),
    }}
    report = {
        "refinement": refinement, "native_physics_steps": steps,
        "servo_updates": 1920, "sensor_updates": 480,
        "controller_condition": {
            "external_servo_hz": 480, "external_sensor_hz": 120,
            "policy_hz": 15, "native_physics_hz": 120 * refinement, "initialization_physics_hz": 120,
        },
    }
    return report, data


@pytest.mark.parametrize("refinement", [4, 8])
def test_registered_servo_cadence_requires_observed_ticks_and_counts(refinement):
    report, data = cadence_fixture(refinement)
    assert all(review.servo_condition_checks(report, data).values())
    old_ticks = torch.arange(480 * refinement) % refinement == 0
    data["native"]["servo_tick"] = old_ticks
    data["native"]["servo_updates_completed"] = old_ticks.double().cumsum(0)
    checks = review.servo_condition_checks(report, data)
    assert checks["timing_counts"]  # A claimed count cannot hide missing updates.
    assert not checks["observed_native_servo_ticks"]
    assert not checks["observed_native_servo_update_counts"]


def test_fine_native_torque_or_gripper_update_between_servo_ticks_is_rejected():
    report, data = cadence_fixture()
    data["native"]["torques"][1, 0] += 1
    checks = review.servo_condition_checks(report, data)
    assert not checks["held_native_torque"]
    assert checks["held_native_gripper_targets"]
    data["native"]["gripper_targets"][3, 0] += 1
    assert not review.servo_condition_checks(report, data)["held_native_gripper_targets"]


def test_servo_check_does_not_admit_changed_sensor_rate_or_unregistered_native_rate():
    report, data = cadence_fixture()
    report["sensor_updates"] = 1920
    assert not review.servo_condition_checks(report, data)["timing_counts"]
    report["refinement"] = 2
    with pytest.raises(ValueError, match="480/960"):
        review.servo_condition_checks(report, data)


def test_pairing_preserves_noise_actions_and_conditions_without_requiring_same_motion():
    report = {"cases": [{"case_id": "fixed-case"}]}
    data = {key: torch.tensor([1.0]) for key in review.PAIRING_KEYS}
    data["applied_action"] = torch.ones((3, 1, 7))
    data["active"] = torch.ones((3, 1), dtype=torch.bool)
    other = copy.deepcopy(data)
    data["actor_before"], other["actor_before"] = torch.tensor([1.0]), torch.tensor([2.0])
    assert all(review.pairing_checks(report, data, report, other).values())
    other["sensor_draws"][0] += 1
    other["requested_action"][0] += 1
    checks = review.pairing_checks(report, data, report, other)
    assert not checks["sensor_draws"]
    assert not checks["requested_action"]
    assert checks["physical_initial"]


def test_descriptive_aggregates_are_sensor_intervals_with_original_metrics(monkeypatch):
    original = {"physics_hz": 480, "cases": [{"active": {"wrist_peak_n": 21.0},
        "contact_servo_intervals": [{"servo_interval_index": 7, "native_wrist_peak_n": 22.0,
                                    "contains_post_terminal_steps": True}]}]}
    monkeypatch.setattr(review.metrics, "describe_run", lambda report, data: copy.deepcopy(original))
    described = review.describe_run({}, {})
    assert described["cases"][0]["active"] == original["cases"][0]["active"]
    assert "contact_servo_intervals" not in described["cases"][0]
    interval = described["cases"][0]["contact_sensor_intervals"][0]
    assert interval["sensor_interval_index"] == 7
    assert interval["external_servo_updates_per_complete_interval"] == 4
    assert interval["contains_post_terminal_steps"]
    assert interval["native_wrist_peak_n"] == 22.0


def test_missing_run_cannot_be_verified(tmp_path):
    result = review.verify_single(tmp_path)
    assert result["status"] == "verification_error"
    assert result["failed_checks"] == ["servo_verifier_completed"]


def test_different_terminal_times_allow_postterminal_holds_but_not_active_action_changes():
    left = {"applied_action": torch.zeros((3, 1, 7)), "active": torch.tensor([[True], [True], [False]])}
    right = {"applied_action": torch.zeros((3, 1, 7)), "active": torch.tensor([[True], [True], [True]])}
    right["applied_action"][2] = 0.5
    scoped = review.common_active_actions(left, right)
    assert scoped["common_active_actions_equal"]
    assert scoped["compared_active_controls"] == 2
    assert scoped["excluded_controls"] == scoped["excluded_action_differences"] == 1
    right["applied_action"][1, 0, 0] = 0.1
    assert not review.common_active_actions(left, right)["common_active_actions_equal"]