"""Single-worker job orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Protocol
from uuid import uuid4

from zero_g_blade_swap import __version__

from .models import Health, Job, JobCreate, JobStatus
from .presets import ExecutionSpec, PresetRegistry, provenance_for
from .runner import CompositeRunner, ExecutionCancelled, ExecutionResult
from .store import JobNotFoundError, JobStore, utc_now


class Runner(Protocol):
    async def run(
        self,
        spec: ExecutionSpec,
        cancel_event: asyncio.Event,
        emit: Callable[..., Awaitable[None]],
    ) -> ExecutionResult: ...


class ManagerStoppingError(RuntimeError):
    pass


class JobConflictError(RuntimeError):
    pass


class JobAdmissionError(RuntimeError):
    pass


class JobManager:
    """Persist jobs and feed exactly one execution backend at a time."""

    def __init__(self, store: JobStore, registry: PresetRegistry, runner: Runner | None = None) -> None:
        self.store = store
        self.registry = registry
        self.runner = runner or CompositeRunner()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._specs: dict[str, ExecutionSpec] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._active_job_id: str | None = None
        self._stopping = False
        self._worker_error: str | None = None

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._queue = asyncio.Queue()
        self._stopping = False
        self.store.recover_interrupted()
        self._worker_error = None
        self._worker_task = asyncio.create_task(self._worker(), name="zero-g-compute-worker")

    async def shutdown(self) -> None:
        if self._worker_task is None:
            return
        self._stopping = True
        for job in self.store.list(limit=10_000):
            if job.status == JobStatus.QUEUED:
                with suppress(JobConflictError, JobNotFoundError):
                    await self.cancel(job.id)
        if self._active_job_id is not None:
            with suppress(JobConflictError, JobNotFoundError):
                await self.cancel(self._active_job_id)
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._worker_task, timeout=20.0)
        except TimeoutError:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        finally:
            self._worker_task = None
            self._specs.clear()
            self._cancel_events.clear()

    def health(self) -> Health:
        if self._stopping:
            worker_state = "stopping"
            service_status = "stopping"
        elif self._worker_error is not None or (self._worker_task is not None and self._worker_task.done()):
            worker_state = "failed"
            service_status = "degraded"
        elif self._active_job_id:
            worker_state = "busy"
            service_status = "ok"
        else:
            worker_state = "idle"
            service_status = "ok"
        return Health(
            status=service_status,
            version=__version__,
            worker_state=worker_state,
            queue_depth=self._queue.qsize(),
            active_job_id=self._active_job_id,
            error=self._worker_error,
        )

    async def submit(self, request: JobCreate) -> Job:
        if self._stopping:
            raise ManagerStoppingError("The service is shutting down")
        if self._worker_task is None:
            raise ManagerStoppingError("The service worker has not started")
        if self._worker_error is not None or self._worker_task.done():
            raise ManagerStoppingError(self._worker_error or "The service worker is not running")
        job_id = str(uuid4())
        artifact_dir = self.store.jobs_dir / job_id / "artifacts"
        spec = await asyncio.to_thread(self.registry.build, request.preset_id, request.seed, artifact_dir)
        provenance = await asyncio.to_thread(provenance_for, spec)
        if self._stopping:
            raise ManagerStoppingError("The service is shutting down")
        if self._worker_error is not None or self._worker_task.done():
            raise ManagerStoppingError(self._worker_error or "The service worker is not running")
        job = Job(
            id=job_id,
            preset_id=spec.preset_id,
            preset_title=spec.preset_title,
            seed=spec.seed,
            status=JobStatus.QUEUED,
            created_at=utc_now(),
            current_stage="queued",
            provenance=provenance,
        )
        try:
            self.store.create(job)
            self.store.append_event(
                job_id,
                event_type="lifecycle",
                message="Job accepted and queued for the serialized worker",
                stage="queued",
                progress=0.0,
            )
        except Exception as exc:
            # Admission is a transaction from the caller's perspective. If the
            # initial event cannot be made durable, never leave an unqueued job
            # looking runnable to polling clients.
            message = f"Job admission persistence failed: {type(exc).__name__}: {exc}"[:2000]
            with suppress(Exception):
                current = self.store.get(job_id)
                self.store.save(
                    current.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "finished_at": utc_now(),
                            "current_stage": "admission_failed",
                            "error": message,
                        }
                    )
                )
            raise JobAdmissionError(message) from exc
        self._specs[job_id] = spec
        self._cancel_events[job_id] = asyncio.Event()
        await self._queue.put(job_id)
        return self.store.get(job_id)

    async def cancel(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job.status.terminal:
            raise JobConflictError(f"Job is already {job.status.value}")
        cancel_event = self._cancel_events.get(job_id)
        if cancel_event is not None:
            cancel_event.set()
        if job.status == JobStatus.QUEUED:
            job = job.model_copy(
                update={
                    "status": JobStatus.CANCELLED,
                    "cancel_requested": True,
                    "finished_at": utc_now(),
                    "current_stage": "cancelled",
                }
            )
            self.store.save(job)
            self.store.append_event(
                job_id,
                event_type="lifecycle",
                level="warning",
                message="Queued job cancelled",
                stage="cancelled",
                progress=job.progress,
            )
        else:
            job = job.model_copy(update={"cancel_requested": True})
            self.store.save(job)
            self.store.append_event(
                job_id,
                event_type="lifecycle",
                level="warning",
                message="Cancellation requested; stopping the active backend",
                stage=job.current_stage,
                progress=job.progress,
            )
        return self.store.get(job_id)

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                if job_id is None:
                    return
                job = self.store.get(job_id)
                if job.status == JobStatus.CANCELLED:
                    self._specs.pop(job_id, None)
                    self._cancel_events.pop(job_id, None)
                    continue
                try:
                    await self._execute(job_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._worker_error = f"Worker persistence failure: {type(exc).__name__}: {exc}"[:2000]
                    self._active_job_id = None
                    self._record_worker_failure(job_id)
                    self._fail_queued_jobs()
                    return
            finally:
                self._queue.task_done()

    def _record_worker_failure(self, job_id: str) -> None:
        """Best-effort terminal state when the persistence boundary itself fails."""

        try:
            current = self.store.get(job_id)
            if not current.status.terminal:
                self.store.save(
                    current.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "finished_at": utc_now(),
                            "current_stage": "worker_failed",
                            "error": self._worker_error,
                        }
                    )
                )
                self.store.append_event(
                    job_id,
                    event_type="error",
                    level="error",
                    message=self._worker_error or "Worker failed",
                    stage="worker_failed",
                )
        except Exception:
            # Health still exposes the failure even if the runtime volume has
            # become completely unwritable.
            pass
        finally:
            self._specs.pop(job_id, None)
            self._cancel_events.pop(job_id, None)

    def _fail_queued_jobs(self) -> None:
        """Ensure clients never poll forever after the sole worker fails."""

        for queued in self.store.list(status=JobStatus.QUEUED, limit=10_000):
            message = f"Job could not start because the serialized worker failed: {self._worker_error}"
            try:
                self.store.save(
                    queued.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "finished_at": utc_now(),
                            "current_stage": "worker_unavailable",
                            "error": message,
                        }
                    )
                )
                self.store.append_event(
                    queued.id,
                    event_type="error",
                    level="error",
                    message=message,
                    stage="worker_unavailable",
                )
            except Exception:
                pass
            finally:
                self._specs.pop(queued.id, None)
                self._cancel_events.pop(queued.id, None)

    async def _execute(self, job_id: str) -> None:
        self._active_job_id = job_id
        job = self.store.get(job_id).model_copy(
            update={
                "status": JobStatus.RUNNING,
                "started_at": utc_now(),
                "current_stage": "starting",
                "progress": 0.01,
            }
        )
        self.store.save(job)
        self.store.append_event(
            job_id,
            event_type="lifecycle",
            message="Serialized worker started the job",
            stage="starting",
            progress=0.01,
        )
        spec = self._specs[job_id]
        cancel_event = self._cancel_events[job_id]

        async def emit(**values: Any) -> None:
            self.store.append_event(job_id, **values)
            current = self.store.get(job_id)
            updates: dict[str, Any] = {}
            if values.get("stage") is not None:
                updates["current_stage"] = values["stage"]
            if values.get("progress") is not None:
                updates["progress"] = max(current.progress, float(values["progress"]))
            if updates:
                self.store.save(current.model_copy(update=updates))

        final_status = JobStatus.FAILED
        error: str | None = None
        exit_code: int | None = None
        job_result = None
        try:
            execution_result = await self.runner.run(spec, cancel_event, emit)
            exit_code = execution_result.exit_code
            job_result = execution_result.result
            if cancel_event.is_set():
                final_status = JobStatus.CANCELLED
            elif exit_code == 0 and job_result is None:
                error = "Backend exited successfully but produced no valid terminal result"
            elif exit_code == 0 and not job_result.completed:
                error = "Workflow did not satisfy its terminal success predicate"
            elif exit_code == 0:
                final_status = JobStatus.SUCCEEDED
            else:
                error = f"Backend exited with code {exit_code}"
        except ExecutionCancelled as exc:
            final_status = JobStatus.CANCELLED
            error = str(exc)
        except asyncio.CancelledError:
            cancel_event.set()
            final_status = JobStatus.CANCELLED
            error = "Service shutdown interrupted the backend"
            raise
        except Exception as exc:  # the persisted error is the API boundary
            error = f"{type(exc).__name__}: {exc}"[:2000]
        finally:
            try:
                # Persist the terminal outcome before potentially expensive
                # video hashing. Shutdown can cancel indexing without losing the
                # result of the compute job itself.
                current = self.store.get(job_id)
                completed = final_status == JobStatus.SUCCEEDED
                current = current.model_copy(
                    update={
                        "status": final_status,
                        "finished_at": utc_now(),
                        "progress": 1.0 if completed else current.progress,
                        "current_stage": final_status.value,
                        "error": error,
                        "exit_code": exit_code,
                        "result": job_result,
                        "cancel_requested": current.cancel_requested or cancel_event.is_set(),
                    }
                )
                self.store.save(current)
                self.store.append_event(
                    job_id,
                    event_type="lifecycle" if completed or final_status == JobStatus.CANCELLED else "error",
                    level="info" if completed else ("warning" if final_status == JobStatus.CANCELLED else "error"),
                    message=(
                        "Job completed successfully"
                        if completed
                        else "Job cancelled"
                        if final_status == JobStatus.CANCELLED
                        else error or "Job failed"
                    ),
                    stage=final_status.value,
                    progress=current.progress,
                )
                try:
                    artifacts = await asyncio.to_thread(self.store.refresh_artifacts, job_id)
                    for artifact in artifacts:
                        self.store.append_event(
                            job_id,
                            event_type="artifact",
                            message=f"Artifact ready: {artifact.path}",
                            data={"artifact": artifact.model_dump(mode="json")},
                        )
                except (OSError, JobNotFoundError) as exc:
                    self.store.append_event(
                        job_id,
                        event_type="error",
                        level="error",
                        message=f"Artifact indexing failed: {exc}",
                        stage="artifact_index",
                    )
            finally:
                self._specs.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
                self._active_job_id = None
