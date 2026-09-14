"""Known physical trajectories defend timing, relative frames and missing data."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_workflow_motion import analyze_capture, motion_series, phase_spans, stall_dwell


def _capture(count=61, fps=30.0):
    poses = np.zeros((count, 3, 7))
    poses[:, :, 3] = 1
    return {
        "poses": poses,
        "paths": np.array([f"/World/envs/env_0/{name}" for name in ("SpareBlade", "Robot/wrist_3_link", "Robot/base_link")]),
        "fps": np.array(fps),
    }


def _report(*events):
    return {"seed": 10, "timeline": [{"event": "start:capture", "step": 0}, *events]}


def test_constant_velocity_uses_simulation_time_despite_video_speed():
    capture = _capture()
    times = np.arange(61) / 30
    capture["poses"][:, :, 0] = (0.12 * times)[:, None]
    result = analyze_capture(capture, _report(), playback_speed=1.5)
    assert result["all_motion"]["module"]["speed_mps"]["max"] == pytest.approx(0.12)
    assert result["all_motion"]["module"]["acceleration_mps2"]["max"] == pytest.approx(0, abs=1e-12)
    assert result["timing"]["simulation_duration_s"] == 2
    assert result["timing"]["presentation_motion_duration_s"] == pytest.approx(4 / 3)
    assert result["base_translation"]["path_length_m"] == pytest.approx(0.24)
    assert result["actuator_target_continuity"]["available"] is False


def test_known_constant_acceleration_and_quaternion_sign_changes():
    times = np.arange(31) / 30
    positions = np.column_stack((0.5 * 0.4 * times**2, np.zeros((31, 2))))
    quaternions = np.zeros((31, 4))
    quaternions[:, 0] = np.cos(0.3 * times / 2)
    quaternions[:, 3] = np.sin(0.3 * times / 2)
    quaternions[::2] *= -1
    metrics = motion_series(positions, quaternions, 30)
    np.testing.assert_allclose(metrics["acceleration_mps2"][2:], 0.4, atol=1e-12)
    np.testing.assert_allclose(metrics["angular_speed_radps"][1:], 0.3, atol=1e-12)


def test_rigid_payload_rotating_with_wrist_has_no_relative_slip():
    capture = _capture()
    angle = np.linspace(0, np.pi / 2, 61)
    # A payload 20 cm from the wrist rotates with it, while the base stays put.
    capture["poses"][:, 0, 0] = 0.2 * np.cos(angle)
    capture["poses"][:, 0, 1] = 0.2 * np.sin(angle)
    capture["poses"][:, :2, 3] = np.cos(angle / 2)[:, None]
    capture["poses"][:, :2, 6] = np.sin(angle / 2)[:, None]
    result = analyze_capture(capture, _report())
    drift = result["phases"][0]["tool_to_module_transform_change_from_phase_start"]
    assert drift["max_position_m"] < 1e-14
    assert drift["max_orientation_rad"] < 1e-14
    assert result["all_motion"]["module"]["speed_mps"]["max"] > 0.15


def test_phase_boundaries_conserve_duration_and_include_boundary_deceleration():
    capture = _capture(count=7, fps=10)
    capture["poses"][:, 0, 0] = [0, 0.01, 0.02, 0.03, 0.03, 0.03, 0.03]
    report = _report({"event": "capture -> extract", "step": 3})
    phases = analyze_capture(capture, report)["phases"]
    assert [phase["duration_s"] for phase in phases] == [0.3, 0.3]
    assert phases[1]["first_state_after_phase_action"] == 4
    assert phases[1]["motion"]["module"]["speed_mps"]["max"] == 0
    assert phases[1]["motion"]["module"]["acceleration_mps2"]["max"] == pytest.approx(1)


def test_dwell_counts_elapsed_intervals_and_breaks_on_motion():
    positions = np.zeros((7, 3))
    positions[:, 0] = [0.010, 0.001, 0.001, 0.001, -0.003, -0.003, -0.003]
    result = stall_dwell(positions, fps=10, plane_x_m=0, radius_m=0.005, speed_limit_mps=0.01)
    assert result["near_plane_duration_s"] == 0.5
    assert result["low_speed_near_plane_duration_s"] == 0.4
    assert result["longest_low_speed_near_plane_interval_s"] == 0.2


def test_targets_are_measured_without_wrapping_or_inventing_them():
    capture = _capture(count=5)
    capture["arm_joint_targets"] = np.zeros((5, 6))
    capture["arm_joint_targets"][2:, 1] = 2 * np.pi
    result = analyze_capture(capture, _report())["actuator_target_continuity"]
    assert result["largest_single_joint_step_rad"] == 2 * np.pi
    assert result["largest_vector_step_ending_state"] == 2


def test_handoff_only_timeline_uses_selected_environment():
    trace = {
        "handoff_fields": np.array(["step", "env", "from_phase", "to_phase"]),
        "handoff": np.array([[5, 0, 0, 2], [3, 1, 0, 2], [8, 1, 2, 3]]),
    }
    phases = phase_spans({}, trace, 10, env_id=1)
    assert [(p["phase"], p["start_driver_step"], p["end_driver_step_exclusive"]) for p in phases] == [
        ("capture", 0, 3), ("extract", 3, 8), ("transit", 8, 10),
    ]


def test_duplicate_completion_trace_sample_does_not_create_time():
    trace = {
        "insert_fields": np.array(["step", "env", "target_x_m"]),
        "insert": np.array([[10, 0, 0.1], [15, 0, 0.12], [15, 0, 0.12], [20, 0, 0.13]]),
    }
    result = analyze_capture(_capture(), _report(), trace)["sparse_trace_metrics"]["insert"]
    assert result["same_step_adjacent_samples"] == 1
    assert result["largest_sample_gap_s"] == pytest.approx(5 / 30)
    assert result["largest_observed_target_change_m"] == pytest.approx(0.02)


@pytest.mark.parametrize("fps", [0, -30, float("nan"), float("inf")])
def test_invalid_timebase_is_rejected(fps):
    with pytest.raises(ValueError, match="fps"):
        analyze_capture(_capture(fps=fps), _report())


def test_nonfinite_capture_and_ambiguous_timeline_are_rejected():
    capture = _capture()
    capture["poses"][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze_capture(capture, _report())
    with pytest.raises(ValueError, match="increasing"):
        analyze_capture(_capture(), _report({"event": "capture -> extract", "step": 0}))


def test_cli_emits_strict_json_and_input_hashes(tmp_path):
    poses = tmp_path / "body_poses.npz"
    report = tmp_path / "workflow_report.json"
    output = tmp_path / "motion.json"
    np.savez_compressed(poses, **_capture())
    report.write_text(json.dumps(_report()), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_workflow_motion.py"
    subprocess.run([sys.executable, str(script), "--poses", str(poses), "--report", str(report), "--output", str(output)], check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema"] == "workflow_motion_analysis_v1"
    assert len(result["input_artifacts"]["body_poses"]["sha256"]) == 64
    assert result["analysis_source"]["analyzer"]["path"] == str(script)
