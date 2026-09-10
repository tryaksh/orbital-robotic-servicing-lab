"""Independent actuator-reference replay catches order, rate and terminal errors."""
import copy

import torch

from assembly_recovery.approach_slew_v1 import clamp_impedance_target, slew_reference
from scripts.review_approach_slew_v1 import replay_reference, verify_single


def trace_fixture():
    initial_tool = torch.tensor([[0.6, 0.0, 0.1]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    threshold = torch.full((1, 3), 1e-5)
    frame = torch.tensor([[0.65, 0.02, 0.11]])
    poses = initial_tool[None].repeat(4, 1, 1)
    poses[1:] += torch.tensor([0.01, 0.0, 0.0])
    quats = quat[None].repeat(4, 1, 1)
    quats[1:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    active = torch.tensor([[True], [False], [False], [False]])
    native = {"tool_pos": poses, "tool_quat": quats, "active_after": active,
              "servo_tick": torch.tensor([True, False, True, False]),
              "servo_updates_completed": torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float64),
              "slew_updates_completed": torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float64),
              "incoming_gripper_command": torch.zeros(4), "gripper_targets": torch.zeros(4, 1, 2)}
    records = {key: [] for key in (
        "requested_cartesian_reference", "slew_reference", "servo_tool_pos_before",
        "clamped_impedance_target", "final_impedance_target", "reward_delta_pos",
        "incoming_impedance_quat", "final_impedance_quat")}
    previous = initial_tool.clone()
    latest = {}
    for step in range(4):
        if step % 2 == 0:
            tool = initial_tool if step == 0 else poses[step - 1]
            previous = slew_reference(previous, frame)
            clamped = clamp_impedance_target(previous, tool, threshold)
            latest = {
                "requested_cartesian_reference": frame, "slew_reference": previous,
                "servo_tool_pos_before": tool, "clamped_impedance_target": clamped,
                "final_impedance_target": clamped if step == 0 else poses[1],
                "reward_delta_pos": frame - tool,
                "incoming_impedance_quat": quat,
                "final_impedance_quat": quat if step == 0 else quats[1],
            }
        for key in records:
            records[key].append(latest[key].clone())
    native.update({key: torch.stack(value) for key, value in records.items()})
    data = {
        "native": native, "controller_initial": {"slew_reference": initial_tool.clone()},
        "initial": {"fingertip_midpoint_pos": initial_tool, "fingertip_midpoint_quat": quat,
                    "init_fixed_pos_obs_noise": torch.zeros(1, 3), "pos_threshold": threshold},
        "sensor_initial": {"fixed_pos_obs_frame": frame}, "applied_action": torch.zeros(1, 1, 7),
    }
    report = {"refinement": 8, "controller_position_bounds": [0.05, 0.05, 0.05]}
    return report, data


def test_reference_clamp_and_delayed_terminal_hold_replay():
    report, data = trace_fixture()
    checks, diagnostics = replay_reference(report, data)
    assert all(checks.values())
    assert diagnostics["reference_step_m"] <= 0.02 / 480 + diagnostics["reference_step_rounding_allowance_m"]
    assert diagnostics["reference_increment_m"] == 0.02 / 480


def test_per_axis_rate_bound_cannot_masquerade_as_euclidean_bound():
    report, data = trace_fixture()
    changed = copy.deepcopy(data)
    changed["native"]["slew_reference"][0] = data["controller_initial"]["slew_reference"] + 0.02 / 480
    checks, _ = replay_reference(report, changed)
    assert not checks["reference_step_bound"]
    assert not checks["euclidean_reference_recurrence"]


def test_original_error_clamp_and_reward_bookkeeping_are_separate_checks():
    report, data = trace_fixture()
    data["native"]["clamped_impedance_target"][0] = data["native"]["slew_reference"][0]
    checks, _ = replay_reference(report, data)
    assert not checks["original_position_error_clamp"]
    assert checks["original_reward_delta_pos"]
    report, data = trace_fixture()
    data["native"]["reward_delta_pos"][0] = data["native"]["slew_reference"][0] - data["native"]["servo_tool_pos_before"][0]
    checks, _ = replay_reference(report, data)
    assert not checks["original_reward_delta_pos"]


def test_absorbing_target_and_quaternion_must_use_captured_terminal_pose():
    report, data = trace_fixture()
    data["native"]["final_impedance_target"][2] = data["native"]["clamped_impedance_target"][2]
    checks, _ = replay_reference(report, data)
    assert not checks["terminal_hold_target"]
    report, data = trace_fixture()
    data["native"]["final_impedance_quat"][2] = data["native"]["incoming_impedance_quat"][2]
    checks, _ = replay_reference(report, data)
    assert not checks["quaternion_terminal_hold_unchanged"]


def test_reference_state_cannot_update_between_actual_servo_ticks():
    report, data = trace_fixture()
    data["native"]["slew_reference"][1] += 1e-3
    checks, _ = replay_reference(report, data)
    assert not checks["reference_records_held_between_servo_ticks"]


def test_gripper_change_and_fake_controller_initial_state_fail():
    report, data = trace_fixture()
    data["native"]["incoming_gripper_command"][0] = 0.01
    data["controller_initial"]["slew_reference"] += 0.01
    checks, _ = replay_reference(report, data)
    assert not checks["original_zero_gripper_command_and_targets"]
    assert not checks["reference_initial_is_between_job_tool_pose"]


def test_missing_artifacts_fail_closed(tmp_path):
    result = verify_single(tmp_path)
    assert result["status"] == "verification_error"
    assert result["failed_checks"] == ["approach_verifier_completed"]
