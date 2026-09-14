"""Fault injection on recorded physical evidence and portable mission artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_g_blade_swap.service.mission_cli import verify_job
from zero_g_blade_swap.service.models import BackendKind, InputProvenance, Job, JobProvenance, JobStatus
from zero_g_blade_swap.service.presets import command_contract, live_workflow_argv, sha256_file
from zero_g_blade_swap.service.store import JobStore, utc_now
from zero_g_blade_swap.service.verification import verify_mission

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "evidence/rgbd_strict_rack_retention_datum_pair_seed6070.json"


def recorded_report() -> dict:
    return json.loads(REPORT.read_text())


def test_known_continuous_camera_episode_satisfies_the_strict_contract():
    result = verify_mission(recorded_report())
    assert result.passed
    assert result.detections == result.detection_attempts == 1772
    assert result.rack_only_hold_s == 0.733333
    assert result.transit_position_drift_mm == pytest.approx(1.048)


@pytest.mark.parametrize(("path", "value", "failed"), [
    (("completed",), "true", "terminal_seating"),
    (("insertion_conditions", "axial_depth"), False, "terminal_seating"),
    (("all_conditions_including_released_gripper",), False, "both_robot_supports_released"),
    (("capture_interface", "observed_per_environment", 0, "released_after_seating"), False, "both_robot_supports_released"),
    (("capture_interface", "observed_per_environment", 0, "hand_opened_after_settling_verification"), False, "both_robot_supports_released"),
    (("destination_rack_retention", "world_constraint"), True, "rack_only_hold"),
    (("destination_rack_retention", "module_pose_write"), True, "rack_only_hold"),
    (("destination_rack_retention", "observed_per_environment", 0, "rack_only_interval_s"), 0.69, "rack_only_hold"),
    (("destination_rack_retention", "observed_per_environment", 0, "rack_only_interval_s"), float("nan"), "rack_only_hold"),
    (("destination_rack_retention", "observed_per_environment", 0, "max_rack_to_module_position_drift_m"), 0.003, "rack_only_hold"),
    (("robot_carried_transit", "carrier"), "payload_stage", "bounded_robot_carried_transit"),
    (("robot_carried_transit", "observed_per_environment", 0, "max_position_drift_m"), 0.003, "bounded_robot_carried_transit"),
    (("robot_carried_transit", "observed_per_environment", 0, "max_orientation_drift_rad"), float("inf"), "bounded_robot_carried_transit"),
    (("robot_carried_transit", "observed_per_environment", 0, "retained_throughout"), "true", "bounded_robot_carried_transit"),
    (("perception", "source"), "oracle", "live_rgbd"),
    (("perception", "detector_availability", "attempts"), 0, "live_rgbd"),
    (("perception", "detector_availability", "failures"), True, "live_rgbd"),
    (("learned_phases",), ["capture", "extract", "insert"], "executed_controllers"),
    (("planning", "source_occupied_destination_clear"), False, "visual_occupancy_plan"),
    (("num_envs",), 8, "single_camera_episode"),
])
def test_contradictory_measurement_cannot_inherit_completed_flag(path, value, failed):
    report = recorded_report()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = verify_mission(report)
    assert not result.passed
    assert failed in result.failed_checks


@pytest.mark.parametrize("report", [None, [], True, {}, {"completed": True}])
def test_incomplete_reports_fail_closed(report):
    assert not verify_mission(report).passed


def make_bundle(tmp_path: Path) -> Path:
    store = JobStore(tmp_path)
    report = recorded_report()
    job = Job(
        id="da68e3a8-f184-4bb3-830e-b58dfb3982ca", preset_id="isaac_full_chain_perception",
        preset_title="Test", seed=6070, status=JobStatus.SUCCEEDED, created_at=utc_now(),
        provenance=JobProvenance(
            service_version="test", preset_revision="test", backend=BackendKind.ISAAC,
            inputs=[InputProvenance(role=role, path=name, sha256=report["checkpoint_sha256"][name], size_bytes=1)
                    for name, role in (("capture", "capture_policy"), ("extract", "extract_policy"), ("insert", "insert_policy"))],
        ),
    )
    store.create(job)
    root = store.artifact_dir(job.id)
    (root / "workflow_report.json").write_text(json.dumps(report))
    (root / "mission_verification.json").write_text(verify_mission(report).model_dump_json())
    (root / "execution.log").write_text("recorded execution")
    (root / "handoff_trace.npz").write_bytes(b"trace fixture")
    (root / "video").mkdir()
    (root / "video" / "episode.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"0" * 32)
    store.refresh_artifacts(job.id)
    return store.job_dir(job.id) / "job.json"


def test_offline_bundle_passes_without_installed_checkpoints(tmp_path):
    result = verify_job(make_bundle(tmp_path))
    assert result["passed"]
    assert result["artifacts_checked"] == 5


@pytest.mark.parametrize("artifact", ["workflow_report.json", "video/episode.mp4", "handoff_trace.npz"])
def test_changed_artifact_fails_integrity(tmp_path, artifact):
    job_file = make_bundle(tmp_path)
    with (job_file.parent / "artifacts" / artifact).open("ab") as stream:
        stream.write(b"changed")
    result = verify_job(job_file)
    assert not result["passed"]
    assert not result["artifact_integrity"]
    assert any(artifact in error for error in result["integrity_errors"])


def test_updated_report_hash_does_not_hide_a_failed_rack_hold(tmp_path):
    job_file = make_bundle(tmp_path)
    path = job_file.parent / "artifacts/workflow_report.json"
    report = json.loads(path.read_text())
    report["destination_rack_retention"]["observed_per_environment"][0]["rack_only_interval_s"] = 0.2
    path.write_text(json.dumps(report))
    job = json.loads(job_file.read_text())
    artifact = next(row for row in job["artifacts"] if row["path"] == "workflow_report.json")
    artifact.update(sha256=sha256_file(path), size_bytes=path.stat().st_size)
    job_file.write_text(json.dumps(job))
    result = verify_job(job_file)
    assert result["artifact_integrity"]
    assert not result["passed"]
    assert "rack_only_hold" in result["mission"]["failed_checks"]


def test_manifest_path_escape_is_rejected(tmp_path):
    job_file = make_bundle(tmp_path)
    job = json.loads(job_file.read_text())
    job["artifacts"][0]["path"] = "../../outside.json"
    job_file.write_text(json.dumps(job))
    result = verify_job(job_file)
    assert not result["passed"]
    assert any("Invalid artifact path" in error for error in result["integrity_errors"])


def test_recipe_binding_includes_controller_and_geometry_but_is_portable(tmp_path):
    from zero_g_blade_swap.service.config import ServiceSettings
    settings = ServiceSettings(project_root=ROOT, runtime_dir=tmp_path, static_dir=tmp_path, isaac_python=tmp_path / "python")
    first = live_workflow_argv(settings, 6070, tmp_path / "a")
    second = live_workflow_argv(settings, 5070, tmp_path / "b")
    assert command_contract(first) == command_contract(second)
    assert "lead_in" in command_contract(first)
    assert "kinematics" in command_contract(first)
    assert "--rack_retention" in command_contract(first)
    assert "v7m130" in first[first.index("--grasp_checkpoint") + 1]
    assert "v19noised" in first[first.index("--extract_checkpoint") + 1]


def test_shipped_checkpoints_match_the_validated_policy_set():
    manifest = json.loads((ROOT / "policies/servicing_v2/MANIFEST.json").read_text(encoding="utf-8"))
    validated = json.loads((ROOT / "evidence/live_service_current_validation_seed6070.json").read_text(encoding="utf-8"))
    for row in manifest["checkpoints"]:
        path = ROOT / "policies/servicing_v2" / row["file"]
        assert path.stat().st_size == row["size_bytes"]
        assert sha256_file(path) == row["sha256"]
        role = "insert" if row["role"] == "insert_loaded_only" else row["role"]
        assert row["sha256"] == validated["checkpoint_sha256"][role].lower()
