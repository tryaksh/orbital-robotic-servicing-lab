"""Replay and subprocess execution backends."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .models import (
    ApplicableLimits,
    BackendKind,
    JobResult,
    PerceptionResult,
    PlanningResult,
    QualificationResult,
    TelemetryResult,
)
from .presets import ExecutionSpec

Emit = Callable[..., Awaitable[None]]


class ExecutionCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    exit_code: int
    result: JobResult | None = None


_PHASE_LINE = re.compile(
    r"^\[PHASE\]\s+(?P<event>.+?)\s+step\s+(?P<step>\d+)\s+"
    r"t=\s*(?P<time>[\d.]+)s\s+blade_x=\s*(?P<blade>-?[\d.]+)\s+"
    r"grip=\s*(?P<grip>-?[\d.]+)mm\s+torque=\s*(?P<torque>-?[\d.]+)Nm"
)
_CHAIN_LINE = re.compile(r"^\[CHAIN\].*episodes=\s*(?P<done>\d+)/(?P<total>\d+)")
_PLAN_LINE = re.compile(
    r"^\[PLAN\]\s+visual occupancy preflight\s+"
    r"passed=(?P<passed>\d+)/(?P<total>\d+)\s+"
    r"source=bay(?P<source>\d+)\s+destination=bay(?P<destination>\d+)\s+"
    r"scores=\[(?P<source_score>[-+\d.eE]+),(?P<destination_score>[-+\d.eE]+)\]\s+"
    r"threshold=(?P<threshold>[-+\d.eE]+)\s*$"
)
_STAGE_PROGRESS = {
    "planning": 0.08,
    "capture": 0.12,
    "seat": 0.28,
    "extract": 0.42,
    "transit": 0.64,
    "insert": 0.82,
    "done": 0.96,
}


def parse_process_line(line: str) -> dict[str, Any]:
    """Convert known workflow output into a stable event envelope."""

    plan = _PLAN_LINE.match(line)
    if plan:
        values = plan.groupdict()
        passed = int(values["passed"])
        total = int(values["total"])
        accepted = total > 0 and passed == total
        planning = {
            "request": {
                "source_bay": int(values["source"]),
                "destination_bay": int(values["destination"]),
            },
            "source_occupied_destination_clear": accepted,
            "passed_environments": passed,
            "evaluated_environments": total,
            "initial_bay_occupancy_scores": [
                float(values["source_score"]),
                float(values["destination_score"]),
            ],
            "decision_threshold": float(values["threshold"]),
            "scores_are_calibrated_confidence": False,
            "used_to_gate_execution": True,
        }
        outcome = "passed" if accepted else "failed"
        return {
            "event_type": "phase",
            "level": "info" if accepted else "warning",
            "message": f"Visual occupancy planning preflight {outcome} for {passed}/{total} environments",
            "stage": "planning",
            "progress": _STAGE_PROGRESS["planning"],
            "data": {"planning": planning},
        }

    phase = _PHASE_LINE.match(line)
    if phase:
        values = phase.groupdict()
        transition = values["event"].strip()
        stage = transition.rsplit("->", 1)[-1].strip().removeprefix("start:")
        return {
            "event_type": "phase",
            "message": transition,
            "stage": stage,
            "progress": _STAGE_PROGRESS.get(stage),
            "data": {
                "step": int(values["step"]),
                "simulation_time_s": float(values["time"]),
                "blade_centre_x_m": float(values["blade"]),
                "grip_error_mm": float(values["grip"]),
                "drive_torque_nm": float(values["torque"]),
                "perception": {
                    "position_m": None,
                    "quaternion_wxyz": None,
                    "confidence": None,
                    "source": "rgb_pose_head (estimate not printed by workflow driver)",
                    "pose_error_mm": None,
                    "pose_error_is_privileged_simulation_diagnostic": True,
                },
                "telemetry": {
                    "grip_error_mm": abs(float(values["grip"])),
                    "peak_force_n": None,
                    "peak_torque_nm": None,
                    "safety_state": "unknown",
                    "applicable_limits": {"force_n": None, "torque_nm": None, "grip_error_mm": None},
                },
            },
        }
    chain = _CHAIN_LINE.match(line)
    if chain:
        done, total = int(chain["done"]), int(chain["total"])
        return {
            "event_type": "progress",
            "message": f"Completed {done} of {total} evaluation episodes",
            "stage": "evaluation",
            "progress": min(0.98, done / total) if total else 0.0,
            "data": {"episodes_completed": done, "episodes_requested": total},
        }
    lowered = line.lower()
    if any(marker in lowered for marker in ("traceback (most recent", "[error]", " error:", "exception:")):
        level = "error"
    elif "[warning]" in lowered or " warning:" in lowered:
        level = "warning"
    else:
        level = "info"
    return {
        "event_type": "log",
        "level": level,
        "message": line[:8192],
    }


class CompositeRunner:
    """Dispatch fixed specs to either replay or the one serialized subprocess."""

    def __init__(self, *, replay_step_delay_s: float = 0.35, cancel_grace_s: float = 8.0) -> None:
        self.replay_step_delay_s = replay_step_delay_s
        self.cancel_grace_s = cancel_grace_s

    async def run(
        self,
        spec: ExecutionSpec,
        cancel_event: asyncio.Event,
        emit: Emit,
    ) -> ExecutionResult:
        if spec.backend == BackendKind.REPLAY:
            return await self._run_replay(spec, cancel_event, emit)
        return await self._run_subprocess(spec, cancel_event, emit)

    async def _run_replay(
        self,
        spec: ExecutionSpec,
        cancel_event: asyncio.Event,
        emit: Emit,
    ) -> ExecutionResult:
        empty_telemetry = {
            "grip_error_mm": None,
            "peak_force_n": None,
            "peak_torque_nm": None,
            "safety_state": "unknown",
            "applicable_limits": {"force_n": None, "torque_nm": None, "grip_error_mm": None},
        }
        replay_perception = {
            "position_m": None,
            "quaternion_wxyz": None,
            "confidence": None,
            "source": "representative_replay_fixture (no model inference)",
            "pose_error_mm": None,
            "pose_error_is_privileged_simulation_diagnostic": False,
        }
        replay_planning = {
            "source": "representative_replay_fixture (no occupancy inference)",
            "source_bay": 0,
            "destination_bay": 1,
            "initial_occupancy_scores": None,
            "decision_threshold": None,
            "gate_passed": None,
            "scores_calibrated": False,
        }
        stages = (
            ("perception", 0.12, "Representative perception event replayed", {"fixture": True}),
            (
                "planning",
                0.20,
                "Representative visual bay-planning gate event replayed; outcome remains unknown",
                {"fixture": True, "gate_passed": None},
            ),
            ("capture", 0.28, "Representative capture event replayed", {"fixture": True}),
            ("extract", 0.46, "Representative extraction event replayed", {"fixture": True}),
            ("transit", 0.67, "Representative inter-bay transit event replayed", {"fixture": True}),
            ("insert", 0.86, "Representative insertion event replayed", {"fixture": True}),
            ("verification", 0.96, "Representative terminal-verification event replayed", {"fixture": True}),
        )
        telemetry: list[dict[str, Any]] = []
        for index, (stage, progress, message, metrics) in enumerate(stages):
            if cancel_event.is_set():
                raise ExecutionCancelled("Replay cancelled")
            await asyncio.sleep(self.replay_step_delay_s)
            # Fixture rows are deterministic and carry no pseudo-measurements.
            row = {
                "step": index,
                "stage": stage,
                "progress": progress,
                "metrics": metrics,
                "perception": replay_perception,
                "planning": replay_planning,
                "telemetry": empty_telemetry,
                "qualification": {
                    "passed": None,
                    "success_rate": None,
                    "trials": 0,
                    "threshold": None,
                    "summary": "Replay is not a qualification trial.",
                },
            }
            telemetry.append(row)
            await emit(
                event_type="phase",
                message=message,
                stage=stage,
                progress=progress,
                data={
                    "metrics": metrics,
                    "perception": replay_perception,
                    "planning": replay_planning,
                    "telemetry": empty_telemetry,
                    "qualification": row["qualification"],
                    "is_live_simulation": False,
                },
            )

        spec.artifact_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = spec.artifact_dir / "telemetry.jsonl"
        summary_path = spec.artifact_dir / "summary.json"
        telemetry_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in telemetry),
            encoding="utf-8",
        )
        summary = {
            "mode": "representative_replay",
            "is_live_simulation": False,
            "completed": True,
            "seed": spec.seed,
            "workflow": "representative two-bay relocation event sequence",
            "perception": "schema fixture only; no image or model inference was executed",
            "represented_learned_phases": ["capture", "extract", "insert"],
            "represented_scripted_phases": ["seat", "transit"],
            "stages": telemetry,
            "result": {
                "completed": True,
                "is_live_simulation": False,
                "perception": replay_perception,
                "planning": replay_planning,
                "telemetry": empty_telemetry,
                "qualification": {
                    "passed": None,
                    "success_rate": None,
                    "trials": 0,
                    "threshold": None,
                    "summary": "Replay validates the compute-service path; it does not create qualification evidence.",
                },
                "summary": (
                    "Representative service-path replay completed. No Isaac simulation, model inference, "
                    "or physical workflow success was executed."
                ),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        result = JobResult.model_validate(summary["result"])
        return ExecutionResult(exit_code=0, result=result)

    async def _run_subprocess(
        self,
        spec: ExecutionSpec,
        cancel_event: asyncio.Event,
        emit: Emit,
    ) -> ExecutionResult:
        if spec.argv is None:
            raise RuntimeError("Isaac execution spec has no argv")
        spec.artifact_dir.mkdir(parents=True, exist_ok=True)
        log_path = spec.artifact_dir / "execution.log"
        environment = os.environ.copy()
        environment.update(spec.environment)
        process_options: dict[str, Any] = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True

        process = await asyncio.create_subprocess_exec(
            *spec.argv,
            cwd=spec.cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **process_options,
        )
        try:
            assert process.stdout is not None
            await emit(
                event_type="lifecycle",
                message="Isaac subprocess started",
                stage="startup",
                progress=0.02,
                data={"pid": process.pid},
            )
            with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as log:
                while True:
                    if cancel_event.is_set():
                        await self._terminate(process)
                        raise ExecutionCancelled("Isaac run cancelled")
                    try:
                        raw = await asyncio.wait_for(process.stdout.readline(), timeout=0.25)
                    except TimeoutError:
                        continue
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    log.write(line + "\n")
                    log.flush()
                    if line:
                        await emit(**parse_process_line(line))
            exit_code = await self._wait_for_exit(process, cancel_event)
            return ExecutionResult(exit_code=exit_code, result=self._live_result(spec))
        finally:
            if process.returncode is None:
                await self._terminate(process)

    async def _wait_for_exit(
        self,
        process: asyncio.subprocess.Process,
        cancel_event: asyncio.Event,
    ) -> int:
        wait_task = asyncio.create_task(process.wait())
        cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            done, _ = await asyncio.wait({wait_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task in done and cancel_event.is_set() and process.returncode is None:
                await self._terminate(process)
                raise ExecutionCancelled("Isaac run cancelled")
            return await wait_task
        finally:
            for task in (wait_task, cancel_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(wait_task, cancel_task, return_exceptions=True)

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            await self._kill_windows_tree(process)
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=self.cancel_grace_s)
            return
        except TimeoutError:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()

    @staticmethod
    async def _kill_windows_tree(process: asyncio.subprocess.Process) -> None:
        """Force-stop the batch wrapper and the Isaac/Kit process it spawned."""

        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        taskkill = os.path.join(system_root, "System32", "taskkill.exe")
        killer: asyncio.subprocess.Process | None = None
        tree_killed = False
        try:
            killer = await asyncio.create_subprocess_exec(
                taskkill,
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                tree_killed = await asyncio.wait_for(killer.wait(), timeout=5.0) == 0
            except TimeoutError:
                try:
                    if killer.returncode is None:
                        killer.kill()
                    await killer.wait()
                except OSError:
                    pass
        except OSError:
            tree_killed = False
        if not tree_killed and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                return
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                # taskkill has already been attempted against the full tree and
                # process.kill against the wrapper. Do not hang service shutdown.
                return

    @staticmethod
    def _live_result(spec: ExecutionSpec) -> JobResult | None:
        report_path = spec.artifact_dir / "workflow_report.json"
        if not report_path.is_file():
            return None
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        # Do not let truthy strings such as ``"false"`` promote a malformed
        # report into a successful workflow outcome.
        completed = report.get("completed") is True
        final = report.get("final") if isinstance(report.get("final"), dict) else {}
        perception = report.get("perception") if isinstance(report.get("perception"), dict) else {}
        planning = report.get("planning") if isinstance(report.get("planning"), dict) else None
        telemetry = report.get("telemetry") if isinstance(report.get("telemetry"), dict) else {}
        limits = telemetry.get("applicable_limits") if isinstance(telemetry.get("applicable_limits"), dict) else {}
        grip_m = final.get("grip_error_m")
        grip_mm = _nonnegative_number(telemetry.get("grip_error_mm"))
        if grip_mm is None and (grip_value := _finite_number(grip_m)) is not None:
            grip_mm = abs(grip_value * 1000.0)
        source = perception.get("source")
        planning_result = None
        if planning is not None:
            request = planning.get("request") if isinstance(planning.get("request"), dict) else {}
            gate_value = planning.get("source_occupied_destination_clear")
            threshold = _finite_number(planning.get("decision_threshold"))
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                threshold = None
            occupancy_scores = _probability_tuple(planning.get("initial_bay_occupancy_scores"), 2)
            gate_passed = gate_value if isinstance(gate_value, bool) else None
            if occupancy_scores is None or threshold is None:
                # A decision without valid scores and its declared threshold is
                # not auditable planning evidence.
                gate_passed = None
            planning_result = PlanningResult(
                source=(
                    "rgbd_fiducial_occupancy_gate"
                    if isinstance(source, str) and "fiducial" in source
                    else "rgb_pose_head_occupancy_gate"
                ),
                source_bay=_nonnegative_int(request.get("source_bay")),
                destination_bay=_nonnegative_int(request.get("destination_bay")),
                initial_occupancy_scores=occupancy_scores,
                decision_threshold=threshold,
                gate_passed=gate_passed,
                # The workflow report explicitly labels these sigmoid values
                # as decision scores, not calibrated confidence.
                scores_calibrated=False,
            )
        return JobResult(
            completed=completed,
            is_live_simulation=True,
            perception=PerceptionResult(
                position_m=_finite_tuple(perception.get("position_m"), 3),
                quaternion_wxyz=_finite_tuple(perception.get("quaternion_wxyz"), 4),
                confidence=_probability(perception.get("confidence")),
                source=(
                    source.strip()
                    if isinstance(source, str) and source.strip()
                    else "rgb_pose_head (terminal estimate not recorded)"
                ),
                pose_error_mm=_nonnegative_number(perception.get("pose_error_mm")),
                pose_error_is_privileged_simulation_diagnostic=(
                    perception.get("pose_error_is_privileged_simulation_diagnostic") is True
                    or _nonnegative_number(perception.get("pose_error_mm")) is not None
                ),
            ),
            planning=planning_result,
            telemetry=TelemetryResult(
                grip_error_mm=grip_mm,
                peak_force_n=_nonnegative_number(telemetry.get("peak_force_n")),
                peak_torque_nm=_nonnegative_number(telemetry.get("peak_torque_nm")),
                safety_state=(
                    telemetry["safety_state"]
                    if telemetry.get("safety_state") in {"nominal", "warning", "stopped", "unknown"}
                    else "unknown"
                ),
                applicable_limits=ApplicableLimits(
                    force_n=_nonnegative_number(limits.get("force_n")),
                    torque_nm=_nonnegative_number(limits.get("torque_nm")),
                    grip_error_mm=_nonnegative_number(limits.get("grip_error_mm")),
                ),
            ),
            qualification=QualificationResult(
                # A single demonstration is an observed outcome, not a
                # reliability qualification.  Only an aggregate run with a
                # declared threshold may turn this field true or false.
                passed=None,
                success_rate=None,
                trials=1,
                threshold=None,
                summary=(
                    "Live Isaac execution completed and settled one relocation successfully; one run is "
                    "still not a statistical reliability qualification."
                    if completed
                    else "Experimental live execution failed its terminal workflow predicate; no "
                    "qualification decision or relocation success was issued."
                ),
            ),
            summary=(
                "Live Isaac execution completed the simulated relocation and passed its settled terminal "
                "gate; this is one measured outcome, not a reliability rate."
                if completed
                else "Experimental live Isaac execution failed the terminal workflow gate; no relocation "
                "success is claimed."
            ),
        )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _nonnegative_number(value: object) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number >= 0.0 else None


def _probability(value: object) -> float | None:
    number = _finite_number(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _finite_tuple(value: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, list | tuple) or len(value) != length:
        return None
    parsed = tuple(_finite_number(item) for item in value)
    return tuple(item for item in parsed if item is not None) if all(item is not None for item in parsed) else None


def _probability_tuple(value: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, list | tuple) or len(value) != length:
        return None
    parsed = tuple(_probability(item) for item in value)
    return tuple(item for item in parsed if item is not None) if all(item is not None for item in parsed) else None
