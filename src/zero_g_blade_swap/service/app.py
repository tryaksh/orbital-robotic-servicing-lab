"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from zero_g_blade_swap import __version__

from .config import ServiceSettings
from .manager import JobAdmissionError, JobConflictError, JobManager, ManagerStoppingError, Runner
from .models import (
    ArtifactList,
    Capabilities,
    ErrorDetail,
    EventList,
    Health,
    Job,
    JobCreate,
    JobList,
    JobStatus,
)
from .presets import PresetRegistry, PresetUnavailableError
from .runner import CompositeRunner
from .store import JobNotFoundError, JobStore


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=ErrorDetail(code=code, message=message).model_dump())


def create_app(
    settings: ServiceSettings | None = None,
    *,
    store: JobStore | None = None,
    registry: PresetRegistry | None = None,
    runner: Runner | None = None,
    manager: JobManager | None = None,
) -> FastAPI:
    """Build an app; all execution dependencies can be replaced in tests."""

    settings = settings or ServiceSettings.from_env()
    if manager is not None:
        if store is not None or registry is not None or runner is not None:
            raise ValueError("Pass either manager or its store/registry/runner dependencies, not both")
        store = manager.store
        registry = manager.registry
    else:
        store = store or JobStore(settings.runtime_dir)
        registry = registry or PresetRegistry(settings)
        runner = runner or CompositeRunner(
            replay_step_delay_s=settings.replay_step_delay_s,
            cancel_grace_s=settings.cancel_grace_s,
        )
        manager = JobManager(store, registry, runner)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await manager.start()
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(
        title="Robotic Serviceability Qualification Compute Service",
        summary="Observable serialized Isaac/GPU design-evaluation jobs with an explicitly synthetic replay path",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.registry = registry
    app.state.manager = manager

    @app.exception_handler(JobNotFoundError)
    async def job_not_found(_request: Request, exc: JobNotFoundError) -> JSONResponse:
        detail = ErrorDetail(code="job_not_found", message=f"Job '{exc.args[0]}' was not found")
        return JSONResponse(status_code=404, content={"detail": detail.model_dump()})

    @app.get("/api/health", response_model=Health, tags=["service"])
    async def health() -> Health:
        return manager.health()

    @app.get("/api/capabilities", response_model=Capabilities, tags=["service"])
    async def capabilities() -> Capabilities:
        return await asyncio.to_thread(registry.capabilities)

    @app.post("/api/jobs", response_model=Job, status_code=status.HTTP_202_ACCEPTED, tags=["jobs"])
    async def create_job(request: JobCreate) -> Job:
        try:
            return await manager.submit(request)
        except KeyError as exc:
            raise _error(404, "preset_not_found", str(exc).strip("'")) from exc
        except PresetUnavailableError as exc:
            raise _error(409, "preset_unavailable", str(exc)) from exc
        except ManagerStoppingError as exc:
            raise _error(503, "service_unavailable", str(exc)) from exc
        except JobAdmissionError as exc:
            raise _error(503, "runtime_storage_unavailable", str(exc)) from exc

    @app.get("/api/jobs", response_model=JobList, tags=["jobs"])
    async def list_jobs(
        job_status: Annotated[JobStatus | None, Query(alias="status")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> JobList:
        return JobList(jobs=store.list(status=job_status, limit=limit))

    @app.get("/api/jobs/{job_id}", response_model=Job, tags=["jobs"])
    async def get_job(job_id: str) -> Job:
        return store.get(job_id)

    @app.delete("/api/jobs/{job_id}", response_model=Job, status_code=status.HTTP_202_ACCEPTED, tags=["jobs"])
    async def cancel_job(job_id: str) -> Job:
        try:
            return await manager.cancel(job_id)
        except JobConflictError as exc:
            raise _error(409, "job_terminal", str(exc)) from exc

    @app.get("/api/jobs/{job_id}/events", response_model=EventList, tags=["events"])
    async def get_events(
        job_id: str,
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 500,
    ) -> EventList:
        events = store.events(job_id, after=after, limit=limit)
        return EventList(events=events, next_after=events[-1].seq if events else after)

    @app.get("/api/jobs/{job_id}/artifacts", response_model=ArtifactList, tags=["artifacts"])
    async def list_artifacts(job_id: str) -> ArtifactList:
        job = store.get(job_id)
        # Terminal state is persisted before potentially expensive video
        # hashing. Make this authoritative listing close that short race for
        # polling clients while keeping the event loop responsive.
        artifacts = await asyncio.to_thread(store.refresh_artifacts, job_id) if job.status.terminal else job.artifacts
        return ArtifactList(artifacts=artifacts)

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}", tags=["artifacts"])
    async def get_artifact(job_id: str, artifact_path: str) -> FileResponse:
        try:
            path = store.resolve_artifact(job_id, artifact_path)
        except FileNotFoundError as exc:
            raise _error(404, "artifact_not_found", f"Artifact '{artifact_path}' was not found") from exc
        artifact = next(
            (item for item in store.get(job_id).artifacts if item.path == artifact_path),
            None,
        )
        return FileResponse(
            path,
            media_type=artifact.media_type if artifact else None,
            headers={"X-Content-Type-Options": "nosniff"},
        )

    index = settings.static_dir / "index.html"
    if settings.static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index_page() -> Response:
        if index.is_file():
            return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})
        return JSONResponse(
            {
                "service": "zero-g-blade-swap-compute",
                "version": __version__,
                "api_docs": "/docs",
            }
        )

    return app
