"""Run and independently verify a recorded robot mission, without a web server."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .config import ServiceSettings
from .manager import JobManager
from .models import BackendKind, Job, JobCreate, JobStatus
from .presets import PresetRegistry, PresetUnavailableError, sha256_file
from .store import JobStore
from .verification import verify_mission

LIVE_PRESET = "isaac_full_chain_perception"


def verify_job(job_file: Path) -> dict:
    """Check every recorded artifact digest, then independently audit the report.

    Hashes detect changes relative to job.json. This is not a digital signature;
    an untrusted party able to replace both manifest and files can forge a bundle.
    """
    job_file = job_file.resolve()
    job = Job.model_validate_json(job_file.read_text(encoding="utf-8"))
    root = (job_file.parent / "artifacts").resolve()
    errors = []
    seen = set()
    for artifact in job.artifacts:
        relative = Path(artifact.path)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(root) or artifact.path in seen):
            errors.append(f"Invalid artifact path: {artifact.path}")
            continue
        seen.add(artifact.path)
        if not path.is_file():
            errors.append(f"Missing artifact: {artifact.path}")
        elif path.stat().st_size != artifact.size_bytes or sha256_file(path) != artifact.sha256:
            errors.append(f"Changed artifact: {artifact.path}")
    for required in ("workflow_report.json", "mission_verification.json", "handoff_trace.npz", "execution.log"):
        if required not in seen:
            errors.append(f"Unbound required artifact: {required}")
    if not any(name.startswith("video/") and name.endswith(".mp4") for name in seen):
        errors.append("No hashed mission video")
    report = {}
    # Do not parse a report whose integrity check failed or whose path escaped.
    if not errors:
        report = json.loads((root / "workflow_report.json").read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            report = {}
        if report.get("seed") != job.seed:
            errors.append("Report seed differs from the submitted mission")
        recorded = report.get("checkpoint_sha256", {})
        for name, role in (("capture", "capture_policy"), ("extract", "extract_policy"), ("insert", "insert_policy")):
            inputs = [item for item in job.provenance.inputs if item.role == role]
            digest = recorded.get(name) if isinstance(recorded, dict) else None
            if len(inputs) != 1 or not isinstance(digest, str) or digest.lower() != inputs[0].sha256.lower():
                errors.append(f"Report does not match the submitted {name} checkpoint")
    verification = verify_mission(report, video_dir=root / "video")
    live = job.provenance.backend == BackendKind.ISAAC and job.status == JobStatus.SUCCEEDED
    return {
        "passed": not errors and live and verification.passed,
        "artifact_integrity": not errors,
        "integrity_errors": errors,
        "artifacts_checked": len(job.artifacts),
        "mission": verification.model_dump(mode="json"),
        "job_file": str(job_file),
        "scope": "One recorded simulation episode. Hash integrity is relative to job.json, not an authenticity signature.",
    }


async def run_mission(settings: ServiceSettings, seed: int) -> Path:
    """Use the same serialized worker as the API and wait through artifact hashing."""
    store = JobStore(settings.runtime_dir)
    manager = JobManager(store, PresetRegistry(settings))
    await manager.start()
    try:
        job = await manager.submit(JobCreate(preset_id=LIVE_PRESET, seed=seed))
        print(f"Mission {job.id}; seed {seed}", flush=True)
        cursor = 0
        while True:
            for event in store.events(job.id, after=cursor):
                cursor = event.seq
                if event.type in {"phase", "lifecycle", "planning", "error"}:
                    print(event.message, flush=True)
            current = store.get(job.id)
            if current.status.terminal and manager.health().active_job_id is None:
                return store.job_dir(job.id) / "job.json"
            await asyncio.sleep(0.25)
    finally:
        await manager.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Check the GPU, checkpoints, and evidence without starting Isaac")
    run = commands.add_parser("run", help="Record one live camera-driven mission with hashed artifacts")
    run.add_argument("--seed", type=int, default=6070)
    run.add_argument("--output", type=Path, help="New output directory; defaults to artifacts/missions/<unique-id>")
    verify = commands.add_parser("verify", help="Check a saved job's hashes and strict completion without Isaac")
    verify.add_argument("job_file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            result = verify_job(args.job_file)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 2
        settings = ServiceSettings.from_env()
        registry = PresetRegistry(settings)
        live = next(preset for preset in registry.capabilities().presets if preset.id == LIVE_PRESET)
        if args.command == "preflight":
            print(live.model_dump_json(indent=2))
            return 0 if live.available else 2
        if not live.available:
            raise PresetUnavailableError(LIVE_PRESET, live.unavailable_reasons)
        JobCreate(preset_id=LIVE_PRESET, seed=args.seed)
        output = args.output or settings.project_root / "artifacts" / "missions" / str(uuid4())
        # A separate runtime prevents recovery or cancellation of another process's jobs.
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        job_file = asyncio.run(run_mission(replace(settings, runtime_dir=output), args.seed))
        result = verify_job(job_file)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    except (OSError, ValueError, ValidationError, PresetUnavailableError) as exc:
        print(f"Mission unavailable: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Mission cancelled; partial artifacts retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
