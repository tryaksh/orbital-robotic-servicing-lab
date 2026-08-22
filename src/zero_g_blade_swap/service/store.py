"""Crash-tolerant JSON persistence scoped to the configured runtime directory."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from .models import Artifact, Event, EventLevel, EventType, Job, JobStatus


class JobNotFoundError(KeyError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


class JobStore:
    """One-directory-per-job store with atomic metadata replacement."""

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir.resolve()
        self.jobs_dir = self.runtime_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._event_sequences: dict[str, int] = {}
        self._load()

    @staticmethod
    def _validated_id(job_id: str) -> str:
        try:
            parsed = UUID(job_id)
        except (ValueError, AttributeError) as exc:
            raise JobNotFoundError(job_id) from exc
        if str(parsed) != job_id:
            raise JobNotFoundError(job_id)
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / self._validated_id(job_id)

    def artifact_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "artifacts"

    def _load(self) -> None:
        for metadata in self.jobs_dir.glob("*/job.json"):
            try:
                job = Job.model_validate_json(metadata.read_text(encoding="utf-8"))
                self._validated_id(job.id)
                if metadata.parent.name != job.id:
                    continue
            except (OSError, ValueError):
                continue
            self._jobs[job.id] = job
            events = metadata.parent / "events.jsonl"
            if events.is_file():
                try:
                    maximum = 0
                    with events.open(encoding="utf-8", errors="replace") as stream:
                        for line in stream:
                            try:
                                maximum = max(maximum, Event.model_validate_json(line).seq)
                            except ValueError:
                                continue
                    self._event_sequences[job.id] = maximum
                except (OSError, UnicodeError):
                    self._event_sequences[job.id] = 0
            else:
                self._event_sequences[job.id] = 0

    def recover_interrupted(self) -> list[Job]:
        recovered: list[Job] = []
        with self._lock:
            for job in tuple(self._jobs.values()):
                if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    continue
                updated = job.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "finished_at": utc_now(),
                        "error": "Service stopped before this job reached a terminal state",
                        "current_stage": "interrupted",
                    }
                )
                self._jobs[job.id] = updated
                self._write_job(updated)
                self.append_event(
                    job.id,
                    event_type="error",
                    level="error",
                    message=updated.error or "Job interrupted",
                    stage="interrupted",
                )
                recovered.append(updated)
        return recovered

    def create(self, job: Job) -> Job:
        with self._lock:
            directory = self.job_dir(job.id)
            directory.mkdir(parents=False, exist_ok=False)
            (directory / "artifacts").mkdir()
            self._jobs[job.id] = job
            self._event_sequences[job.id] = 0
            self._write_job(job)
        return job.model_copy(deep=True)

    def _write_job(self, job: Job) -> None:
        metadata = self.job_dir(job.id) / "job.json"
        temporary = metadata.with_suffix(".json.tmp")
        temporary.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, metadata)

    def save(self, job: Job) -> Job:
        with self._lock:
            if job.id not in self._jobs:
                raise JobNotFoundError(job.id)
            self._jobs[job.id] = job.model_copy(deep=True)
            self._write_job(job)
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> Job:
        with self._lock:
            try:
                return self._jobs[job_id].model_copy(deep=True)
            except KeyError as exc:
                raise JobNotFoundError(job_id) from exc

    def list(self, *, status: JobStatus | None = None, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)
            if status is not None:
                jobs = [job for job in jobs if job.status == status]
            return [job.model_copy(deep=True) for job in jobs[:limit]]

    def append_event(
        self,
        job_id: str,
        *,
        event_type: EventType,
        message: str,
        level: EventLevel = "info",
        stage: str | None = None,
        progress: float | None = None,
        data: dict[str, object] | None = None,
    ) -> Event:
        with self._lock:
            self.get(job_id)
            path = self.job_dir(job_id) / "events.jsonl"
            seq = self._event_sequences.get(job_id, 0) + 1
            event = Event(
                seq=seq,
                timestamp=utc_now(),
                type=event_type,
                level=level,
                message=message,
                stage=stage,
                progress=progress,
                data=data or {},
            )
            needs_separator = False
            if path.is_file() and path.stat().st_size:
                with path.open("rb") as stream:
                    stream.seek(-1, os.SEEK_END)
                    needs_separator = stream.read(1) not in {b"\n", b"\r"}
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                if needs_separator:
                    stream.write("\n")
                stream.write(event.model_dump_json() + "\n")
            self._event_sequences[job_id] = seq
            return event

    def events(self, job_id: str, *, after: int = 0, limit: int = 500) -> list[Event]:
        with self._lock:
            self.get(job_id)
            path = self.job_dir(job_id) / "events.jsonl"
            if not path.is_file():
                return []
            events: list[Event] = []
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        event = Event.model_validate_json(line)
                    except ValueError:
                        # A power loss may leave only the last JSONL record torn.
                        # Earlier durable events remain useful and replayable.
                        continue
                    if event.seq > after:
                        events.append(event)
                        if len(events) >= limit:
                            break
            return events

    def refresh_artifacts(self, job_id: str) -> list[Artifact]:
        root = self.artifact_dir(job_id).resolve()
        artifacts: list[Artifact] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            artifacts.append(
                Artifact(
                    path=relative,
                    media_type=media_type,
                    size_bytes=path.stat().st_size,
                    sha256=_digest(path),
                )
            )
        job = self.get(job_id)
        self.save(job.model_copy(update={"artifacts": artifacts}))
        return artifacts

    def resolve_artifact(self, job_id: str, artifact_path: str) -> Path:
        self.get(job_id)
        root = self.artifact_dir(job_id).resolve()
        candidate = (root / artifact_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FileNotFoundError(artifact_path) from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(artifact_path)
        return candidate
