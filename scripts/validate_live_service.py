"""Record and audit the proposed live service profile before promoting its evidence.

Requires a clean tracked checkout, local policy weights, and Isaac Sim. This is
an offline validation run; service admission continues to require passing evidence.
Every result, including failure, stays in a new output directory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from zero_g_blade_swap.provenance import git_source_revision
from zero_g_blade_swap.service.config import ServiceSettings
from zero_g_blade_swap.service.models import BackendKind
from zero_g_blade_swap.service.presets import (
    FIDUCIAL_EVIDENCE,
    FULL_CHAIN_EVIDENCE,
    LIVE_INPUT_REQUIREMENTS,
    ExecutionSpec,
    PresetRegistry,
    command_contract,
    live_workflow_argv,
    provenance_for,
    sha256_file,
)
from zero_g_blade_swap.service.runner import CompositeRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    parser.add_argument("--seed", type=int, default=6070)
    args = parser.parse_args()
    if not 0 <= args.seed <= 2_147_483_647:
        parser.error("seed must be between 0 and 2147483647")
    settings = ServiceSettings.from_env()
    revision = git_source_revision(settings.project_root)
    if revision.get("dirty") is not False:
        parser.error("commit tracked changes before validating a source-bound profile")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    preset = PresetRegistry(settings).get("isaac_full_chain_perception")
    argv = live_workflow_argv(settings, args.seed, output)
    spec = ExecutionSpec(
        preset_id=preset.id, preset_title=preset.title, preset_revision=preset.revision,
        backend=BackendKind.ISAAC, seed=args.seed, artifact_dir=output, argv=argv,
        cwd=settings.project_root, environment={"PYTHONUNBUFFERED": "1"},
        input_files=tuple((role, settings.project_root / relative)
                          for _, role, relative in LIVE_INPUT_REQUIREMENTS
                          if relative not in {FIDUCIAL_EVIDENCE, FULL_CHAIN_EVIDENCE}),
    )
    provenance = provenance_for(spec)
    (output / "launch_provenance.json").write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")

    async def execute():
        async def emit(**event):
            if event.get("event_type") in {"phase", "lifecycle", "planning", "error"}:
                print(event["message"], flush=True)
        return await CompositeRunner().run(spec, asyncio.Event(), emit)

    result = asyncio.run(execute())
    report_path = output / "workflow_report.json"
    if not report_path.is_file():
        print(f"No report produced; execution log retained at {output}")
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        service_preset_revision=preset.revision,
        service_command_contract=command_contract(argv),
        source_report_sha256=sha256_file(report_path),
        generated_by="scripts/validate_live_service.py",
        service_validation=(result.result.verification.model_dump(mode="json")
                            if result.result and result.result.verification else None),
        scope_and_limitations=[
            "One recorded simulation episode at one seed; not a success rate or hardware qualification.",
            "Stable lighting for video, one environment. Published pooled cohorts use eight environments per seed.",
            "Module pose is RGB-D-derived; module velocity is zero before capture and robot-kinematic after capture.",
            "Capture uses v7m130 and extraction v19noised. The insert checkpoint is loaded but guarded advance produces the insertion actions.",
            "The guard uses the previously derived lead-in catch; final seating and the 0.70 s rack-only recheck are unchanged.",
            "Robot-side and rack-side fixed joints are idealized load paths; visible pawls have no contact colliders.",
            "The robot base is world-fixed. Spacecraft reaction and carriage compliance are not simulated.",
        ],
    )
    (output / "service_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = result.exit_code == 0 and result.result is not None and result.result.completed
    print(json.dumps({"passed": passed, "report": str(output / "service_validation.json")}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
