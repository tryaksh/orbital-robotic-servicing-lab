"""Public API and persisted job models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


EventType: TypeAlias = Literal["lifecycle", "phase", "progress", "log", "artifact", "error"]
EventLevel: TypeAlias = Literal["debug", "info", "warning", "error"]
Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
NonnegativeFloat: TypeAlias = Annotated[float, Field(ge=0.0)]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class BackendKind(StrEnum):
    REPLAY = "replay"
    ISAAC = "isaac"


class JobCreate(StrictModel):
    """The complete user-controlled job surface.

    There is intentionally no command, path, environment, or upload field.
    """

    preset_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    seed: int = Field(default=4070, ge=0, le=2_147_483_647)


class InputProvenance(StrictModel):
    role: str
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class JobProvenance(StrictModel):
    schema_version: Literal[1] = 1
    service_version: str
    source_revision: str | None = None
    source_dirty: bool | None = None
    preset_revision: str
    backend: BackendKind
    command_argv: list[str] | None = None
    inputs: list[InputProvenance] = Field(default_factory=list)


class Artifact(StrictModel):
    path: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str


class PerceptionResult(StrictModel):
    position_m: tuple[float, float, float] | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    confidence: Probability | None = None
    source: str
    pose_error_mm: NonnegativeFloat | None = None
    pose_error_is_privileged_simulation_diagnostic: bool = False


class ApplicableLimits(StrictModel):
    force_n: NonnegativeFloat | None = None
    torque_nm: NonnegativeFloat | None = None
    grip_error_mm: NonnegativeFloat | None = None


class TelemetryResult(StrictModel):
    grip_error_mm: NonnegativeFloat | None = None
    peak_force_n: NonnegativeFloat | None = None
    peak_torque_nm: NonnegativeFloat | None = None
    safety_state: Literal["nominal", "warning", "stopped", "unknown"] = "unknown"
    applicable_limits: ApplicableLimits = Field(default_factory=ApplicableLimits)


class QualificationResult(StrictModel):
    passed: bool | None = None
    success_rate: Probability | None = None
    trials: int = Field(default=0, ge=0)
    threshold: Probability | None = None
    summary: str


class PlanningResult(StrictModel):
    source: str
    source_bay: int | None = Field(default=None, ge=0)
    destination_bay: int | None = Field(default=None, ge=0)
    initial_occupancy_scores: tuple[Probability, Probability] | None = None
    decision_threshold: Probability | None = None
    gate_passed: bool | None = None
    scores_calibrated: Literal[False] = False


class JobResult(StrictModel):
    completed: bool
    is_live_simulation: bool
    perception: PerceptionResult
    planning: PlanningResult | None = None
    telemetry: TelemetryResult
    qualification: QualificationResult
    summary: str


class Job(StrictModel):
    id: str
    preset_id: str
    preset_title: str
    seed: int
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_stage: str | None = None
    error: str | None = None
    exit_code: int | None = None
    cancel_requested: bool = False
    artifacts: list[Artifact] = Field(default_factory=list)
    result: JobResult | None = None
    provenance: JobProvenance


class Event(StrictModel):
    seq: int = Field(ge=1)
    timestamp: datetime
    type: EventType
    level: EventLevel = "info"
    message: str
    stage: str | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    data: dict[str, Any] = Field(default_factory=dict)


class PresetCapability(StrictModel):
    id: str
    title: str
    description: str
    backend: BackendKind
    available: bool
    unavailable_reasons: list[str] = Field(default_factory=list)
    estimated_runtime_s: int = Field(ge=0)
    produces_video: bool
    perception: bool


class Capabilities(StrictModel):
    service_version: str
    worker_concurrency: Literal[1] = 1
    replay_available: Literal[True] = True
    isaac_python: str
    isaac_available: bool
    gpu_available: bool
    gpu_name: str | None = None
    presets: list[PresetCapability]


class Health(StrictModel):
    status: Literal["ok", "degraded", "stopping"]
    service: Literal["zero-g-blade-swap-compute"] = "zero-g-blade-swap-compute"
    version: str
    worker_state: Literal["idle", "busy", "failed", "stopping"]
    queue_depth: int = Field(ge=0)
    active_job_id: str | None = None
    error: str | None = None


class JobList(StrictModel):
    jobs: list[Job]


class EventList(StrictModel):
    events: list[Event]
    next_after: int = Field(ge=0)


class ArtifactList(StrictModel):
    artifacts: list[Artifact]


class ErrorDetail(StrictModel):
    code: str
    message: str
