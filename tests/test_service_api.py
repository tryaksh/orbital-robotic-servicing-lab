from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient

from zero_g_blade_swap.service.app import create_app
from zero_g_blade_swap.service.config import ServiceSettings
from zero_g_blade_swap.service.models import (
    ApplicableLimits,
    JobResult,
    PerceptionResult,
    QualificationResult,
    TelemetryResult,
)
from zero_g_blade_swap.service.presets import ExecutionSpec, PresetRegistry
from zero_g_blade_swap.service.runner import ExecutionCancelled, ExecutionResult
from zero_g_blade_swap.service.store import JobStore

ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path, *, static: Path | None = None) -> ServiceSettings:
    return ServiceSettings(
        project_root=ROOT,
        runtime_dir=tmp_path / "runtime",
        static_dir=static or tmp_path / "no-static",
        isaac_python=tmp_path / "missing-isaac-python",
        replay_step_delay_s=0.0,
        cancel_grace_s=0.1,
    )


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_replay_job_exercises_complete_api(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["worker_state"] == "idle"

        created = client.post("/api/jobs", json={"preset_id": "replay_full_chain", "seed": 17})
        assert created.status_code == 202
        job_id = created.json()["id"]
        job = _wait_for_terminal(client, job_id)

        assert job["status"] == "succeeded"
        assert job["progress"] == 1.0
        assert job["result"]["is_live_simulation"] is False
        assert job["result"]["perception"] == {
            "position_m": None,
            "quaternion_wxyz": None,
            "confidence": None,
            "source": "representative_replay_fixture (no model inference)",
            "pose_error_mm": None,
            "pose_error_is_privileged_simulation_diagnostic": False,
        }
        assert job["result"]["planning"] == {
            "source": "representative_replay_fixture (no occupancy inference)",
            "source_bay": 0,
            "destination_bay": 1,
            "initial_occupancy_scores": None,
            "decision_threshold": None,
            "gate_passed": None,
            "scores_calibrated": False,
        }
        assert job["result"]["telemetry"]["peak_force_n"] is None
        assert job["result"]["qualification"]["passed"] is None
        assert job["provenance"]["backend"] == "replay"
        assert job["provenance"]["command_argv"] is None

        listing = client.get("/api/jobs", params={"status": "succeeded"})
        assert [row["id"] for row in listing.json()["jobs"]] == [job_id]

        first_page = client.get(f"/api/jobs/{job_id}/events", params={"limit": 2}).json()
        assert len(first_page["events"]) == 2
        second_page = client.get(
            f"/api/jobs/{job_id}/events",
            params={"after": first_page["next_after"], "limit": 100},
        ).json()
        assert second_page["events"]
        phase = next(event for event in second_page["events"] if event["type"] == "phase")
        assert set(phase["data"]) >= {"perception", "telemetry", "qualification", "is_live_simulation"}
        planning_event = next(event for event in second_page["events"] if event["stage"] == "planning")
        assert planning_event["data"]["planning"]["gate_passed"] is None
        assert planning_event["data"]["planning"]["initial_occupancy_scores"] is None
        assert planning_event["data"]["metrics"] == {"fixture": True, "gate_passed": None}

        artifacts = client.get(f"/api/jobs/{job_id}/artifacts").json()["artifacts"]
        assert {item["path"] for item in artifacts} == {"summary.json", "telemetry.jsonl"}
        summary = client.get(f"/api/jobs/{job_id}/artifacts/summary.json")
        assert summary.status_code == 200
        assert summary.json()["mode"] == "representative_replay"
        assert "reference_evidence" not in summary.json()
        assert "No Isaac simulation" in summary.json()["result"]["summary"]
        assert summary.headers["x-content-type-options"] == "nosniff"


def test_request_surface_rejects_commands_and_unknown_presets(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        arbitrary = client.post(
            "/api/jobs",
            json={"preset_id": "replay_full_chain", "seed": 1, "command": "calc.exe"},
        )
        assert arbitrary.status_code == 422
        assert client.post("/api/jobs", json={"preset_id": "../../shell", "seed": 1}).status_code == 422
        missing = client.post("/api/jobs", json={"preset_id": "not_a_preset", "seed": 1})
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "preset_not_found"


def test_capabilities_are_live_and_replay_remains_available(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    registry = PresetRegistry(settings)
    registry._gpu_probe = lambda: (False, None)  # type: ignore[method-assign]
    app = create_app(settings, registry=registry)
    with TestClient(app) as client:
        payload = client.get("/api/capabilities").json()
    assert payload["replay_available"] is True
    assert payload["gpu_available"] is False
    replay, live = payload["presets"]
    assert replay["available"] is True
    assert replay["title"].startswith("Representative")
    assert "not a simulation run" in replay["description"]
    assert live["available"] is False
    assert live["title"].startswith("Live RGB-D")
    assert "robot-carried transit" in live["description"]
    assert "form lock" in live["description"]
    assert "settling verification" in live["description"]
    assert any("Isaac Python" in reason for reason in live["unavailable_reasons"])
    assert any("NVIDIA GPU" in reason for reason in live["unavailable_reasons"])


class _BlockingRunner:
    async def run(self, spec: ExecutionSpec, cancel_event: asyncio.Event, emit) -> ExecutionResult:
        await emit(event_type="phase", message="blocked", stage="capture", progress=0.2)
        while not cancel_event.is_set():
            await asyncio.sleep(0.005)
        raise ExecutionCancelled("cancelled by test")


def test_running_job_can_be_cancelled(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), runner=_BlockingRunner())
    with TestClient(app) as client:
        response = client.post("/api/jobs", json={"preset_id": "replay_full_chain"})
        job_id = response.json()["id"]
        deadline = time.monotonic() + 2
        while client.get(f"/api/jobs/{job_id}").json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.005)
        cancelled = client.delete(f"/api/jobs/{job_id}")
        assert cancelled.status_code == 202
        job = _wait_for_terminal(client, job_id)
        assert job["status"] == "cancelled"
        assert job["cancel_requested"] is True
        assert client.delete(f"/api/jobs/{job_id}").status_code == 409


def test_queued_job_is_cancelled_without_reaching_runner(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), runner=_BlockingRunner())
    with TestClient(app) as client:
        active_id = client.post("/api/jobs", json={"preset_id": "replay_full_chain", "seed": 1}).json()["id"]
        deadline = time.monotonic() + 2
        while client.get(f"/api/jobs/{active_id}").json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.005)

        queued_id = client.post("/api/jobs", json={"preset_id": "replay_full_chain", "seed": 2}).json()["id"]
        cancelled = client.delete(f"/api/jobs/{queued_id}").json()
        assert cancelled["status"] == "cancelled"
        assert cancelled["started_at"] is None
        assert _wait_for_terminal(client, queued_id)["status"] == "cancelled"

        client.delete(f"/api/jobs/{active_id}")
        assert _wait_for_terminal(client, active_id)["status"] == "cancelled"


class _SerialRunner:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def run(self, spec: ExecutionSpec, cancel_event: asyncio.Event, emit) -> ExecutionResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.04)
            result = JobResult(
                completed=True,
                is_live_simulation=False,
                perception=PerceptionResult(source="test"),
                telemetry=TelemetryResult(applicable_limits=ApplicableLimits()),
                qualification=QualificationResult(summary="test"),
                summary="test",
            )
            return ExecutionResult(0, result)
        finally:
            self.active -= 1


class _UnsuccessfulRunner:
    async def run(self, spec: ExecutionSpec, cancel_event: asyncio.Event, emit) -> ExecutionResult:
        result = JobResult(
            completed=False,
            is_live_simulation=True,
            perception=PerceptionResult(source="rgb_pose_head", pose_error_mm=None),
            telemetry=TelemetryResult(applicable_limits=ApplicableLimits()),
            qualification=QualificationResult(
                passed=False,
                success_rate=0.0,
                trials=1,
                summary="Terminal predicate failed.",
            ),
            summary="Workflow failed its terminal predicate.",
        )
        return ExecutionResult(0, result)


def test_exit_zero_does_not_mask_unsuccessful_workflow(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), runner=_UnsuccessfulRunner())
    with TestClient(app) as client:
        job_id = client.post("/api/jobs", json={"preset_id": "replay_full_chain"}).json()["id"]
        job = _wait_for_terminal(client, job_id)
    assert job["status"] == "failed"
    assert job["result"]["completed"] is False
    assert job["result"]["qualification"]["passed"] is False
    assert "terminal success predicate" in job["error"]


class _FailRunningSaveStore(JobStore):
    def __init__(self, runtime_dir: Path) -> None:
        super().__init__(runtime_dir)
        self.failed_once = False

    def save(self, job):
        if job.status == "running" and not self.failed_once:
            self.failed_once = True
            raise OSError("simulated runtime volume failure")
        return super().save(job)


class _FailInitialEventStore(JobStore):
    def __init__(self, runtime_dir: Path) -> None:
        super().__init__(runtime_dir)
        self.failed_once = False

    def append_event(self, job_id: str, **values):
        if not self.failed_once:
            self.failed_once = True
            raise OSError("simulated initial event failure")
        return super().append_event(job_id, **values)


def test_partial_admission_is_terminalized_not_stranded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _FailInitialEventStore(settings.runtime_dir)
    app = create_app(settings, store=store)
    with TestClient(app) as client:
        response = client.post("/api/jobs", json={"preset_id": "replay_full_chain"})
        jobs = client.get("/api/jobs").json()["jobs"]
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "runtime_storage_unavailable"
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["current_stage"] == "admission_failed"


def test_worker_persistence_failure_degrades_health_and_rejects_work(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _FailRunningSaveStore(settings.runtime_dir)
    app = create_app(settings, store=store, runner=_SerialRunner())
    with TestClient(app) as client:
        job_id = client.post("/api/jobs", json={"preset_id": "replay_full_chain"}).json()["id"]
        job = _wait_for_terminal(client, job_id)
        health = client.get("/api/health").json()
        rejected = client.post("/api/jobs", json={"preset_id": "replay_full_chain"})
    assert job["status"] == "failed"
    assert job["current_stage"] == "worker_failed"
    assert health["status"] == "degraded"
    assert health["worker_state"] == "failed"
    assert "runtime volume failure" in health["error"]
    assert rejected.status_code == 503


def test_worker_serializes_jobs(tmp_path: Path) -> None:
    runner = _SerialRunner()
    app = create_app(_settings(tmp_path), runner=runner)
    with TestClient(app) as client:
        ids = [
            client.post("/api/jobs", json={"preset_id": "replay_full_chain", "seed": seed}).json()["id"]
            for seed in (1, 2, 3)
        ]
        jobs = [_wait_for_terminal(client, job_id) for job_id in ids]
    assert {job["status"] for job in jobs} == {"succeeded"}
    assert runner.max_active == 1


def test_manager_can_restart_cleanly_with_same_app(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        first_id = client.post("/api/jobs", json={"preset_id": "replay_full_chain"}).json()["id"]
        assert _wait_for_terminal(client, first_id)["status"] == "succeeded"
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        second_id = client.post("/api/jobs", json={"preset_id": "replay_full_chain"}).json()["id"]
        assert _wait_for_terminal(client, second_id)["status"] == "succeeded"


def test_artifact_traversal_is_rejected(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        job_id = client.post("/api/jobs", json={"preset_id": "replay_full_chain"}).json()["id"]
        _wait_for_terminal(client, job_id)
        response = client.get(f"/api/jobs/{job_id}/artifacts/%2e%2e/job.json")
        assert response.status_code == 404


def test_static_index_is_served_when_present(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>service dashboard</title>", encoding="utf-8")
    app = create_app(_settings(tmp_path, static=static))
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "service dashboard" in response.text
    assert response.headers["cache-control"] == "no-cache"


def test_dashboard_does_not_promote_planning_gate_to_qualification() -> None:
    dashboard = (ROOT / "src" / "zero_g_blade_swap" / "service" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    qualification_line = next(line for line in dashboard.splitlines() if "passed: hasTypedQualification" in line)
    assert 'latestBoolean(sources, ["qualification_passed", "acceptance_passed"])' in qualification_line
    assert '"gate_passed"' not in qualification_line
    assert "authoritative unknowns" in dashboard
    assert 'textContent = "Start run"' in dashboard
    assert "Schema fixture; no model inference" in dashboard
    pose_error_line = next(line for line in dashboard.splitlines() if line.strip().startswith("poseError:"))
    assert "grip_error" not in pose_error_line
