from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from zero_g_blade_swap.service.config import ServiceSettings
from zero_g_blade_swap.service.models import BackendKind, Job, JobProvenance, JobStatus
from zero_g_blade_swap.service.presets import (
    ASSET_SOURCE,
    CAMERA_CONFIG_SOURCE,
    FIDUCIAL_EVIDENCE,
    FIDUCIAL_SOURCE,
    FULL_CHAIN_EVIDENCE,
    INSERT_W65_TWO_SLOT,
    LIVE_INPUT_REQUIREMENTS,
    LIVE_TASK_ID,
    PERCEPTION_SOURCE,
    WORKCELL_CONFIG_SOURCE,
    WORKFLOW_SCRIPT,
    ExecutionSpec,
    PresetRegistry,
    provenance_for,
    sha256_file,
)
from zero_g_blade_swap.service.runner import CompositeRunner, parse_process_line
from zero_g_blade_swap.service.store import JobStore, utc_now

ROOT = Path(__file__).resolve().parents[1]


def test_core_service_import_does_not_require_fastapi() -> None:
    """The simulator and CPU checks use service core without the HTTP extra."""

    blocker = """
import importlib.abc

class BlockFastAPI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fastapi" or fullname.startswith("fastapi."):
            raise ModuleNotFoundError("fastapi intentionally unavailable")
        return None

import sys
sys.meta_path.insert(0, BlockFastAPI())
from zero_g_blade_swap.service import ServiceSettings
from zero_g_blade_swap.service.config import ServiceSettings as DirectSettings
assert ServiceSettings is DirectSettings
"""
    completed = subprocess.run(
        [sys.executable, "-c", blocker],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _live_registry(
    tmp_path: Path,
    *,
    evidence_status: str | None = "passed",
) -> tuple[PresetRegistry, ServiceSettings]:
    project_root = (tmp_path / "project").resolve()
    if os.name == "nt":
        # Checkpoint paths are intentionally descriptive and can push pytest's
        # already-long temporary root past Win32's legacy MAX_PATH limit.
        project_root = Path("\\\\?\\" + str(project_root))
    for _label, _role, relative in LIVE_INPUT_REQUIREMENTS:
        if relative in {FIDUCIAL_EVIDENCE, FULL_CHAIN_EVIDENCE}:
            continue
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test input")

    def bindings(paths: tuple[Path, ...]) -> list[dict[str, str]]:
        return [{"path": path.as_posix(), "sha256": sha256_file(project_root / path)} for path in paths]

    if evidence_status is not None:
        evidence = project_root / FIDUCIAL_EVIDENCE
        evidence.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": evidence_status,
            "evidence_type": "rendered_rgbd_fiducial_pose_heldout_gate",
            "frames": 1024,
            "detected_frames": 970,
            "detection_rate": 0.947,
            "critical_bay_detection_rate": 1.0,
            "position_error_mm": {"p95": 1.7},
            "orientation_error_rad": {"p95": 0.012},
            "occupancy_exact_match": 1.0,
            "dataset_sha256": "a" * 64,
            "calibration": {"resolution_px": [384, 384]},
            "deployment_boundary": {
                "runtime_inputs": ["rgb", "registered_metric_depth", "camera_intrinsics"],
            },
            "runtime_source_bindings": bindings(
                (FIDUCIAL_SOURCE, ASSET_SOURCE, PERCEPTION_SOURCE, CAMERA_CONFIG_SOURCE)
            ),
        }
        evidence.write_text(json.dumps(payload), encoding="utf-8")
    workflow_evidence = project_root / FULL_CHAIN_EVIDENCE
    workflow_evidence.parent.mkdir(parents=True, exist_ok=True)
    workflow_evidence.write_text(
        json.dumps(
            {
                "task": LIVE_TASK_ID,
                "completed": True,
                "reached_phase": "done",
                "predicate_fired": True,
                "seated_conditions_still_held_after_settling": True,
                "visual_randomization": "on",
                "perception": {
                    "source": "rgb_fiducial_calibrated_pnp",
                    "terminal_bay_occupancy_scores": [0.0, 1.0],
                },
                "planning": {"source_occupied_destination_clear": True},
                "checkpoint_sha256": {
                    "capture": sha256_file(project_root / LIVE_INPUT_REQUIREMENTS[1][2]),
                    "extract": sha256_file(project_root / LIVE_INPUT_REQUIREMENTS[2][2]),
                    "insert": sha256_file(project_root / INSERT_W65_TWO_SLOT),
                },
                "runtime_source_bindings": bindings(
                    (
                        WORKFLOW_SCRIPT,
                        FIDUCIAL_SOURCE,
                        ASSET_SOURCE,
                        PERCEPTION_SOURCE,
                        CAMERA_CONFIG_SOURCE,
                        WORKCELL_CONFIG_SOURCE,
                    )
                ),
            }
        ),
        encoding="utf-8",
    )
    isaac_python = tmp_path / "isaac-python.bat"
    isaac_python.write_text("test launcher", encoding="utf-8")
    settings = ServiceSettings(
        project_root=project_root,
        runtime_dir=tmp_path / "runtime",
        static_dir=tmp_path / "static",
        isaac_python=isaac_python,
    )
    registry = PresetRegistry(settings)
    registry._gpu_probe = lambda: (True, "test GPU")  # type: ignore[method-assign]
    return registry, settings


def _job(status: JobStatus = JobStatus.QUEUED) -> Job:
    return Job(
        id=str(uuid4()),
        preset_id="replay_full_chain",
        preset_title="Replay",
        seed=1,
        status=status,
        created_at=utc_now(),
        provenance=JobProvenance(
            service_version="test",
            preset_revision="test-v1",
            backend=BackendKind.REPLAY,
        ),
    )


def test_store_persists_and_recovers_interrupted_jobs(tmp_path: Path) -> None:
    first = JobStore(tmp_path / "runtime")
    queued = _job(JobStatus.QUEUED)
    running = _job(JobStatus.RUNNING)
    done = _job(JobStatus.SUCCEEDED)
    for job in (queued, running, done):
        first.create(job)

    reopened = JobStore(tmp_path / "runtime")
    recovered = reopened.recover_interrupted()
    assert {job.id for job in recovered} == {queued.id, running.id}
    assert reopened.get(queued.id).status == JobStatus.FAILED
    assert reopened.get(running.id).current_stage == "interrupted"
    assert reopened.get(done.id).status == JobStatus.SUCCEEDED
    assert reopened.events(queued.id)[0].type == "error"


def test_store_resolves_only_regular_files_below_artifact_root(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "runtime")
    job = _job()
    store.create(job)
    artifact = store.artifact_dir(job.id) / "nested" / "report.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    store.refresh_artifacts(job.id)
    assert store.resolve_artifact(job.id, "nested/report.json") == artifact.resolve()
    with pytest.raises(FileNotFoundError):
        store.resolve_artifact(job.id, "../job.json")
    with pytest.raises(FileNotFoundError):
        store.resolve_artifact(job.id, "nested")


def test_torn_event_tail_does_not_hide_durable_events(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = JobStore(runtime)
    job = _job()
    store.create(job)
    first = store.append_event(job.id, event_type="lifecycle", message="durable")
    with (store.job_dir(job.id) / "events.jsonl").open("ab") as stream:
        stream.write(b'{"torn":"\xe2\x80')

    reopened = JobStore(runtime)
    second = reopened.append_event(job.id, event_type="lifecycle", message="after recovery")
    assert first.seq == 1
    assert second.seq == 2
    assert [event.message for event in reopened.events(job.id)] == ["durable", "after recovery"]


def test_phase_stdout_is_structured_without_inventing_missing_sensors() -> None:
    row = parse_process_line(
        "[PHASE] extract -> transit         step  812  t= 27.07s  blade_x=0.9123  grip=  4.20mm  torque= 1.25Nm"
    )
    assert row["event_type"] == "phase"
    assert row["stage"] == "transit"
    assert row["data"]["grip_error_mm"] == 4.2
    assert row["data"]["telemetry"] == {
        "grip_error_mm": 4.2,
        "peak_force_n": None,
        "peak_torque_nm": None,
        "safety_state": "unknown",
        "applicable_limits": {"force_n": None, "torque_nm": None, "grip_error_mm": None},
    }
    assert row["data"]["perception"]["confidence"] is None


def test_chain_progress_stdout_is_structured() -> None:
    row = parse_process_line("[CHAIN] step  1200  episodes=  64/192  {'capture': 1}")
    assert row["event_type"] == "progress"
    assert row["progress"] == pytest.approx(1 / 3)
    assert row["data"] == {"episodes_completed": 64, "episodes_requested": 192}


@pytest.mark.parametrize(
    ("passed", "expected", "level"),
    (("1/1", True, "info"), ("0/1", False, "warning")),
)
def test_visual_occupancy_preflight_stdout_is_a_structured_planning_gate(
    passed: str,
    expected: bool,
    level: str,
) -> None:
    row = parse_process_line(
        f"[PLAN] visual occupancy preflight passed={passed} source=bay0 destination=bay1 "
        "scores=[0.9234,0.0123] threshold=0.50"
    )
    assert row["event_type"] == "phase"
    assert row["stage"] == "planning"
    assert row["progress"] == 0.08
    assert row["level"] == level
    assert row["data"]["planning"] == {
        "request": {"source_bay": 0, "destination_bay": 1},
        "source_occupied_destination_clear": expected,
        "passed_environments": int(passed.split("/")[0]),
        "evaluated_environments": 1,
        "initial_bay_occupancy_scores": [0.9234, 0.0123],
        "decision_threshold": 0.5,
        "scores_are_calibrated_confidence": False,
        "used_to_gate_execution": True,
    }


def test_metric_names_containing_error_are_not_error_logs() -> None:
    assert parse_process_line('  "axial_error_m": 0.001,')["level"] == "info"
    assert parse_process_line("[Error] CUDA initialization failed")["level"] == "error"


def test_post_spawn_emit_failure_still_terminates_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdout:
        async def readline(self) -> bytes:
            return b""

    class FakeProcess:
        pid = 12345
        returncode = None
        stdout = FakeStdout()

    process = FakeProcess()

    async def fake_spawn(*_argv, **_options):
        return process

    terminated = False
    runner = CompositeRunner(replay_step_delay_s=0, cancel_grace_s=0)

    async def fake_terminate(target) -> None:
        nonlocal terminated
        assert target is process
        terminated = True
        target.returncode = -1

    async def broken_emit(**_values) -> None:
        raise OSError("event volume full")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(runner, "_terminate", fake_terminate)
    spec = ExecutionSpec(
        preset_id="test",
        preset_title="test",
        preset_revision="test-v1",
        backend=BackendKind.ISAAC,
        seed=1,
        artifact_dir=tmp_path / "artifacts",
        argv=("fixed-executable", "fixed-argument"),
        cwd=tmp_path,
        environment={},
        input_files=(),
    )
    with pytest.raises(OSError, match="event volume full"):
        asyncio.run(runner.run(spec, asyncio.Event(), broken_emit))
    assert terminated is True


def test_live_result_consumes_reported_perception_without_inventing_channels(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "workflow_report.json").write_text(
        json.dumps(
            {
                "completed": True,
                "final": {"grip_error_m": 0.004},
                "perception": {
                    "position_m": [0.5, -0.22, 0.7],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "confidence": None,
                    "source": "rgb_pose_head",
                    "pose_error_mm": 2.4,
                    "pose_error_is_privileged_simulation_diagnostic": True,
                },
                "planning": {
                    "request": {"source_bay": 0, "destination_bay": 1},
                    "source_occupied_destination_clear": True,
                    "initial_bay_occupancy_scores": [0.93, 0.08],
                    "decision_threshold": 0.5,
                    "scores_are_calibrated_confidence": False,
                    "used_to_gate_execution": True,
                },
            }
        ),
        encoding="utf-8",
    )
    spec = ExecutionSpec(
        preset_id="test",
        preset_title="test",
        preset_revision="test-v1",
        backend=BackendKind.ISAAC,
        seed=1,
        artifact_dir=artifact_dir,
        argv=None,
        cwd=tmp_path,
        environment={},
        input_files=(),
    )
    result = CompositeRunner._live_result(spec)
    assert result is not None
    assert result.perception.position_m == (0.5, -0.22, 0.7)
    assert result.perception.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert result.perception.pose_error_mm == 2.4
    assert result.perception.pose_error_is_privileged_simulation_diagnostic is True
    assert result.perception.confidence is None
    assert result.planning is not None
    assert result.planning.source == "rgb_pose_head_occupancy_gate"
    assert result.planning.source_bay == 0
    assert result.planning.destination_bay == 1
    assert result.planning.initial_occupancy_scores == (0.93, 0.08)
    assert result.planning.decision_threshold == 0.5
    assert result.planning.gate_passed is True
    assert result.planning.scores_calibrated is False
    assert result.telemetry.grip_error_mm == 4.0
    assert result.telemetry.peak_force_n is None
    assert result.qualification.passed is None
    assert result.qualification.success_rate is None
    assert result.qualification.trials == 1
    assert "not a statistical reliability qualification" in result.qualification.summary
    assert "passed its settled terminal gate" in result.summary


def test_live_preset_uses_fixed_argv_and_runtime_outputs(tmp_path: Path) -> None:
    registry, settings = _live_registry(tmp_path)
    artifacts = tmp_path / "runtime" / "jobs" / str(uuid4()) / "artifacts"
    spec = registry.build("isaac_full_chain_perception", 123, artifacts)
    assert spec.argv is not None
    assert spec.argv[0] == str(settings.isaac_python)
    assert spec.preset_title.startswith("Live RGB-D")
    assert "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0" in spec.argv
    assert spec.argv[spec.argv.index("--seed") + 1] == "123"
    # Long enough for one whole workflow. Measured end to end at about 3,800
    # control steps on this workcell, so the old 3,600 could only ever end by
    # running out of clock.
    assert int(spec.argv[spec.argv.index("--steps") + 1]) >= 4200
    # And on the rail, which is the configuration that squares the module. The
    # rail carries the robot; --base_rail_on_relocation carries the module and is
    # asserted absent below.
    assert "--robot_rail_on_relocation" in spec.argv
    assert spec.argv[spec.argv.index("--perception_backend") + 1] == "fiducial_pnp"
    # The robot carries the module: the form lock is commanded, and the
    # world-mounted payload stage that used to appear here is not.
    assert "--base_rail_on_relocation" not in spec.argv
    assert "--latch_on_release" in spec.argv
    assert spec.argv[spec.argv.index("--latch_joint_mode") + 1] == "fixed"
    assert float(spec.argv[spec.argv.index("--latch_rated_force_n") + 1]) > 0.0
    assert float(spec.argv[spec.argv.index("--latch_rated_torque_nm") + 1]) > 0.0
    assert spec.argv[spec.argv.index("--insert_checkpoint") + 1] == str(
        (settings.project_root / INSERT_W65_TWO_SLOT).resolve()
    )
    assert spec.argv[spec.argv.index("--report") + 1] == str(artifacts / "workflow_report.json")
    assert "--stable_lighting" not in spec.argv
    assert {role for role, _path in spec.input_files} == {
        "workflow_driver",
        "capture_policy",
        "extract_policy",
        "insert_policy",
        "perception_evidence",
        "workflow_evidence",
        "fiducial_estimator",
        "service_plate_asset",
        "perception_integration",
        "service_latch_geometry",
        "camera_config",
        "workcell_config",
    }
    provenance = provenance_for(spec)
    evidence_input = next(item for item in provenance.inputs if item.role == "perception_evidence")
    assert evidence_input.path == FIDUCIAL_EVIDENCE.as_posix()
    assert len(evidence_input.sha256) == 64
    assert evidence_input.size_bytes > 0
    assert all(";" not in argument and "&&" not in argument for argument in spec.argv)


def test_live_failed_terminal_gate_is_an_outcome_not_a_qualification(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "workflow_report.json").write_text(
        json.dumps(
            {
                "completed": False,
                "planning": {
                    "source_occupied_destination_clear": False,
                    "initial_bay_occupancy_scores": [0.2, 0.8],
                    "decision_threshold": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    spec = ExecutionSpec(
        preset_id="test",
        preset_title="test",
        preset_revision="test-v1",
        backend=BackendKind.ISAAC,
        seed=1,
        artifact_dir=artifact_dir,
        argv=None,
        cwd=tmp_path,
        environment={},
        input_files=(),
    )
    result = CompositeRunner._live_result(spec)
    assert result is not None
    assert result.completed is False
    assert result.planning is not None
    assert result.planning.gate_passed is False
    assert result.planning.source_bay is None
    assert result.planning.initial_occupancy_scores == (0.2, 0.8)
    assert result.qualification.passed is None
    assert result.qualification.success_rate is None
    assert "failed its terminal workflow predicate" in result.qualification.summary
    assert "no relocation success is claimed" in result.summary


def test_live_result_rejects_truthy_completion_and_invalid_measurements(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "workflow_report.json").write_text(
        json.dumps(
            {
                "completed": "false",
                "perception": {"confidence": 4.0, "pose_error_mm": -1.0},
                "planning": {
                    "source_occupied_destination_clear": True,
                    "initial_bay_occupancy_scores": [1.2, -0.1],
                    "decision_threshold": 0.5,
                },
                "telemetry": {"peak_force_n": -4.0},
            }
        ),
        encoding="utf-8",
    )
    spec = ExecutionSpec(
        preset_id="test",
        preset_title="test",
        preset_revision="test-v1",
        backend=BackendKind.ISAAC,
        seed=1,
        artifact_dir=artifact_dir,
        argv=None,
        cwd=tmp_path,
        environment={},
        input_files=(),
    )
    result = CompositeRunner._live_result(spec)
    assert result is not None
    assert result.completed is False
    assert result.perception.confidence is None
    assert result.perception.pose_error_mm is None
    assert result.planning is not None
    assert result.planning.initial_occupancy_scores is None
    assert result.planning.gate_passed is None
    assert result.telemetry.peak_force_n is None


def test_live_capability_requires_passing_rgbd_evidence(tmp_path: Path) -> None:
    missing_registry, _ = _live_registry(tmp_path / "missing", evidence_status=None)
    missing_live = missing_registry.capabilities().presets[1]
    assert missing_live.available is False
    assert any("Missing RGB-D fiducial evidence" in reason for reason in missing_live.unavailable_reasons)

    failed_registry, _ = _live_registry(tmp_path / "failed", evidence_status="failed")
    failed_live = failed_registry.capabilities().presets[1]
    assert failed_live.available is False
    assert any("has not passed" in reason for reason in failed_live.unavailable_reasons)


def test_live_capability_requires_rgbd_pose_and_occupancy_metrics(tmp_path: Path) -> None:
    registry, settings = _live_registry(tmp_path)
    evidence_path = settings.project_root / FIDUCIAL_EVIDENCE
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["occupancy_exact_match"] = 0.80
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    live = registry.capabilities().presets[1]
    assert live.available is False
    assert any("occupancy exact match" in reason for reason in live.unavailable_reasons)


def test_live_capability_rejects_stale_camera_binding(tmp_path: Path) -> None:
    registry, settings = _live_registry(tmp_path)
    camera_config = settings.project_root / CAMERA_CONFIG_SOURCE
    camera_config.write_bytes(camera_config.read_bytes() + b"\nchanged")

    live = registry.capabilities().presets[1]
    assert live.available is False
    assert any("stale" in reason and "scene_cfg.py" in reason for reason in live.unavailable_reasons)


def test_live_capability_requires_a_statistically_useful_holdout(tmp_path: Path) -> None:
    registry, settings = _live_registry(tmp_path)
    evidence_path = settings.project_root / FIDUCIAL_EVIDENCE
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["frames"] = 12
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    live = registry.capabilities().presets[1]
    assert live.available is False
    assert any("at least 1000 frames" in reason for reason in live.unavailable_reasons)


def test_live_capability_requires_successful_full_chain_evidence(tmp_path: Path) -> None:
    registry, settings = _live_registry(tmp_path)
    evidence_path = settings.project_root / FULL_CHAIN_EVIDENCE
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["completed"] = False
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    live = registry.capabilities().presets[1]
    assert live.available is False
    assert any("not a settled successful relocation" in reason for reason in live.unavailable_reasons)
