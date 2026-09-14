"""Measure recorded workflow motion on the CPU, without replaying or changing it.

Input poses are the initial state followed by one post-physics state per control
step, in metres and scalar-first quaternions. All derivatives use recorded
simulation time, never the video's playback speed. Example::

    python scripts/analyze_workflow_motion.py --poses capture/body_poses.npz \
        --report capture/workflow_report.json --trace capture/handoff_trace.npz \
        --output capture/motion_analysis.json

The result is descriptive evidence about one recording, not a success predicate
or a replacement for the mission verifier. Missing actuator targets are reported
as unavailable; movement is not evidence of what the controller commanded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zero_g_blade_swap import grapple_geometry  # noqa: E402

PHASE_NAMES = ("capture", "seat", "extract", "transit", "insert", "done")


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _quaternions(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm < 1.0e-8):
        raise ValueError("Pose quaternions must be finite and nonzero")
    return values / norm


def _conjugate(quaternions: np.ndarray) -> np.ndarray:
    result = quaternions.copy()
    result[..., 1:] *= -1
    return result


def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[..., :1] * right[..., :1] - np.sum(left[..., 1:] * right[..., 1:], axis=-1, keepdims=True)
    vector = left[..., :1] * right[..., 1:] + right[..., :1] * left[..., 1:] + np.cross(left[..., 1:], right[..., 1:])
    return np.concatenate((scalar, vector), axis=-1)


def _rotate(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    uv = np.cross(quaternions[..., 1:], vectors)
    return vectors + 2 * (quaternions[..., :1] * uv + np.cross(quaternions[..., 1:], uv))


def _rotation_vectors(quaternions: np.ndarray) -> np.ndarray:
    # q and -q represent the same rotation. Use the shorter incremental arc.
    q = _quaternions(quaternions)
    q = np.where(q[..., :1] < 0, -q, q)
    length = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(length, q[..., :1])
    return q[..., 1:] * np.divide(angle, length, out=np.full_like(angle, 2.0), where=length > 1.0e-12)


def _statistics(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "samples": int(values.size),
        "max": float(values.max()) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
        "rms": float(np.sqrt(np.mean(values**2))) if values.size else None,
    }


def motion_series(positions: np.ndarray, quaternions: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Backward finite differences aligned with their ending captured state."""
    _positive(fps, "fps")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2 or not np.isfinite(positions).all():
        raise ValueError("Positions must be a finite (N >= 2, 3) array")
    q = _quaternions(np.asarray(quaternions, dtype=float))
    if q.shape != (len(positions), 4):
        raise ValueError("Quaternions must have shape (N, 4)")
    velocity = np.full_like(positions, np.nan)
    velocity[1:] = np.diff(positions, axis=0) * fps
    acceleration = np.full_like(positions, np.nan)
    acceleration[2:] = np.diff(velocity[1:], axis=0) * fps
    jerk = np.full_like(positions, np.nan)
    jerk[3:] = np.diff(acceleration[2:], axis=0) * fps
    angular = np.full_like(positions, np.nan)
    angular[1:] = _rotation_vectors(_multiply(q[1:], _conjugate(q[:-1]))) * fps
    return {
        "step_translation_m": np.linalg.norm(velocity, axis=1) / fps,
        "speed_mps": np.linalg.norm(velocity, axis=1),
        "acceleration_mps2": np.linalg.norm(acceleration, axis=1),
        "jerk_mps3": np.linalg.norm(jerk, axis=1),
        "angular_speed_radps": np.linalg.norm(angular, axis=1),
    }


def phase_spans(report: dict, trace: dict[str, np.ndarray], last_state: int, env_id: int) -> list[dict]:
    """Controller step s acts on state s and produces state s + 1."""
    transitions: list[tuple[int, str]] = []
    if env_id == 0:
        for event in report.get("timeline", []):
            label = event["event"]
            if label.startswith("start:"):
                transitions.append((int(event["step"]), label.split(":", 1)[1]))
            elif " -> " in label:
                transitions.append((int(event["step"]), label.split(" -> ", 1)[1]))
    if not transitions and "handoff" in trace:
        fields = list(trace["handoff_fields"])
        rows = trace["handoff"]
        rows = rows[rows[:, fields.index("env")] == env_id]
        if len(rows):
            transitions = [(0, PHASE_NAMES[int(rows[0, fields.index("from_phase")])])]
            transitions.extend((int(row[fields.index("step")]), PHASE_NAMES[int(row[fields.index("to_phase")])]) for row in rows)
    if not transitions or transitions[0][0] != 0:
        raise ValueError("A timeline beginning at step 0 or an environment handoff trace is required")
    steps = [row[0] for row in transitions]
    if steps != sorted(set(steps)) or any(step > last_state or step < 0 for step in steps):
        raise ValueError("Phase transition steps must be strictly increasing within the captured states")
    result = []
    for index, (start, phase) in enumerate(transitions):
        stop = transitions[index + 1][0] if index + 1 < len(transitions) else last_state
        result.append({"phase": phase, "start_driver_step": start, "end_driver_step_exclusive": stop})
    return result


def stall_dwell(
    positions: np.ndarray, fps: float, plane_x_m: float, radius_m: float, speed_limit_mps: float
) -> dict:
    """Count whole intervals within the plane band, including their endpoints."""
    _positive(radius_m, "stall radius")
    _positive(speed_limit_mps, "stall speed limit")
    _positive(fps, "fps")
    positions = np.asarray(positions, dtype=float)
    near = np.abs(positions[:, 0] - plane_x_m) <= radius_m
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps
    band_intervals = near[:-1] & near[1:]
    stalled = band_intervals & (speed <= speed_limit_mps)
    longest = current = 0
    for value in stalled:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {
        "plane_x_m": plane_x_m,
        "band_radius_m": radius_m,
        "speed_limit_mps": speed_limit_mps,
        "near_plane_duration_s": float(band_intervals.sum() / fps),
        "low_speed_near_plane_duration_s": float(stalled.sum() / fps),
        "longest_low_speed_near_plane_interval_s": longest / fps,
        "definition": "Whole extraction intervals with both endpoints in the band; low speed uses 3-D displacement.",
        "is_success_criterion": False,
    }


def _body(poses: np.ndarray, paths: list[str], env_id: int, relative: str) -> tuple[np.ndarray, str]:
    path = f"/World/envs/env_{env_id}/{relative}"
    if paths.count(path) != 1:
        raise ValueError(f"Expected exactly one captured body {path}")
    return poses[:, paths.index(path)], path


def _target_metrics(capture: dict[str, np.ndarray], count: int, fps: float) -> dict:
    if "arm_joint_targets" not in capture:
        return {"available": False, "reason": "The capture did not record actuator targets; pose jumps are reported separately."}
    values = np.asarray(capture["arm_joint_targets"], dtype=float)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("arm_joint_targets must be a finite (N states, joints) array")
    delta = np.diff(values, axis=0)
    magnitude = np.linalg.norm(delta, axis=1)
    largest = int(np.argmax(magnitude)) + 1
    result = {
        "available": True,
        "largest_vector_step_rad": float(magnitude.max()),
        "largest_single_joint_step_rad": float(np.abs(delta).max()),
        "largest_vector_step_ending_state": largest,
        "largest_vector_step_ending_simulation_time_s": largest / fps,
        "per_joint_max_step_rad": np.abs(delta).max(axis=0).tolist(),
        "joint_angles_unwrapped": False,
        "scope": "Adjacent sampled actuator targets; a commanded step is not an instantaneous physical displacement.",
    }
    if "arm_joint_names" in capture:
        names = capture["arm_joint_names"].tolist()
        if len(names) != values.shape[1]:
            raise ValueError("arm_joint_names does not match arm_joint_targets")
        result["joint_names"] = names
    return result


def _trace_metrics(trace: dict[str, np.ndarray], env_id: int, fps: float) -> dict:
    result = {}
    for name, column in (("transit", "tool_to_module_drift_m"), ("insert", "target_x_m")):
        if name not in trace:
            continue
        fields = list(trace[f"{name}_fields"])
        rows = trace[name]
        if column not in fields or rows.ndim != 2:
            continue
        rows = rows[rows[:, fields.index("env")] == env_id]
        if not len(rows):
            continue
        values = rows[:, fields.index(column)]
        steps = rows[:, fields.index("step")]
        if not np.isfinite(values).all() or np.any(np.diff(steps) < 0):
            raise ValueError(f"{name} trace values must be finite with nondecreasing steps")
        # A phase-completion sample and the regular periodic sample can both
        # appear at the same step. Preserve them without inventing elapsed time.
        result[name] = {
            "samples": len(rows), "column": column, "max_recorded_value": float(values.max()),
            "same_step_adjacent_samples": int(np.count_nonzero(np.diff(steps) == 0)),
        }
        if name == "insert" and len(rows) > 1:
            result[name].update({
                "largest_observed_target_change_m": float(np.abs(np.diff(values)).max()),
                "largest_sample_gap_s": float(np.diff(steps).max() / fps),
                "scope": "Changes between sparse trace samples; intervening per-control-step target changes are not recorded.",
            })
    return result


def analyze_capture(
    capture: dict[str, np.ndarray], report: dict, trace: dict[str, np.ndarray] | None = None,
    *, env_id: int = 0, playback_speed: float = 1.5, plane_x_m: float | None = None,
    stall_radius_m: float = 0.005, stall_speed_mps: float = 0.005,
) -> dict:
    """Analyze one environment's complete, unedited, uniformly sampled recording."""
    trace = {} if trace is None else trace
    fps = _positive(float(np.asarray(capture["fps"]).item()), "fps")
    _positive(playback_speed, "playback speed")
    plane = grapple_geometry.EXTRACTED_BLADE_CENTRE_X if plane_x_m is None else plane_x_m
    if not math.isfinite(plane):
        raise ValueError("Extraction plane must be finite")
    poses = np.asarray(capture["poses"], dtype=float)
    paths = [str(path) for path in capture["paths"]]
    if poses.ndim != 3 or poses.shape[0] < 2 or poses.shape[1:] != (len(paths), 7):
        raise ValueError("poses must have shape (N >= 2, number of paths, 7)")
    if not np.isfinite(poses).all():
        raise ValueError("Captured poses must be finite")
    module, module_path = _body(poses, paths, env_id, "SpareBlade")
    wrist, wrist_path = _body(poses, paths, env_id, "Robot/wrist_3_link")
    base, base_path = _body(poses, paths, env_id, "Robot/base_link")
    tool_q = _quaternions(wrist[:, 3:])
    module_q = _quaternions(module[:, 3:])
    offset = np.asarray(grapple_geometry.GRAPPLE_TOOL_OFFSET_POS)
    tool_position = wrist[:, :3] + _rotate(tool_q, offset)
    bodies = {"module": (module[:, :3], module_q), "tool": (tool_position, tool_q), "base": (base[:, :3], base[:, 3:])}
    series = {name: motion_series(position, q, fps) for name, (position, q) in bodies.items()}
    relative_position = _rotate(_conjugate(tool_q), module[:, :3] - tool_position)
    relative_rotation = _multiply(_conjugate(tool_q), module_q)
    phases = phase_spans(report, trace, len(poses) - 1, env_id)
    for phase in phases:
        start, end = phase["start_driver_step"], phase["end_driver_step_exclusive"]
        phase["duration_s"] = (end - start) / fps
        phase["presentation_duration_s_at_requested_speed"] = phase["duration_s"] / playback_speed
        phase["first_state_after_phase_action"] = start + 1 if end > start else None
        # Include cross-phase derivative changes: velocities belong to the
        # interval they describe, not an isolated segment padded with zeros.
        phase["motion"] = {
            name: {key: _statistics(values[start + 1:end + 1]) for key, values in body.items()}
            for name, body in series.items()
        }
        relative_delta = relative_position[start:end + 1] - relative_position[start]
        rotation_delta = _multiply(relative_rotation[start:end + 1], _conjugate(relative_rotation[start]))
        phase["tool_to_module_transform_change_from_phase_start"] = {
            "max_position_m": float(np.linalg.norm(relative_delta, axis=1).max()),
            "max_orientation_rad": float(np.linalg.norm(_rotation_vectors(rotation_delta), axis=1).max()),
            "reference_state": start,
            "scope": "Phase-local reference; capture and compliant mating intentionally change this transform.",
        }
        if phase["phase"] == "extract":
            phase["stall_dwell"] = stall_dwell(module[start:end + 1, :3], fps, plane, stall_radius_m, stall_speed_mps)
    peak_base = int(np.nanargmax(series["base"]["step_translation_m"]))
    return {
        "schema": "workflow_motion_analysis_v1",
        "scope_and_limitations": [
            "One captured simulation episode; descriptive motion metrics do not certify reliability or physical success.",
            "Unsmooth finite differences at the recorded control frequency; no filtering, interpolation or video retiming.",
            "Substep contact impulses and actuator forces cannot be reconstructed from sampled body poses.",
            "Largest pose step is measured displacement, not proof of teleportation or a collision.",
            "Per-phase transform changes use phase-local anchors and need not equal the driver's lock-engagement drift.",
            "Reported source provenance is retained for attribution, not independently verified by this analyzer.",
        ],
        "environment": env_id,
        "seed": report.get("seed"),
        "completed_in_source_report": report.get("completed"),
        "source_revision_reported_by_simulation": report.get("source_revision"),
        "runtime_source_bindings_reported_by_simulation": report.get("runtime_source_bindings"),
        "timing": {
            "captured_states": len(poses), "simulated_control_intervals": len(poses) - 1,
            "control_fps_from_capture": fps, "simulation_duration_s": (len(poses) - 1) / fps,
            "requested_presentation_speed": playback_speed,
            "presentation_motion_duration_s": (len(poses) - 1) / fps / playback_speed,
            "presentation_scope": "Continuous-time mapping only; encoder frame rounding and decorative ending holds are excluded.",
            "state_convention": "State 0 is initial; driver action s produces captured state s+1. Timeline steps are pre-action.",
            "derivative_convention": "Backward finite differences assigned to ending state; phase-boundary changes retained.",
        },
        "body_paths": {"module": module_path, "tool_parent_wrist": wrist_path, "base": base_path},
        "tool_offset_in_wrist_frame_m": offset.tolist(),
        "all_motion": {name: {key: _statistics(values) for key, values in body.items()} for name, body in series.items()},
        "base_translation": {
            "net_xyz_m": (base[-1, :3] - base[0, :3]).tolist(),
            "path_length_m": float(np.nansum(series["base"]["step_translation_m"])),
            "max_distance_from_initial_m": float(np.linalg.norm(base[:, :3] - base[0, :3], axis=1).max()),
            "largest_per_step_translation_m": float(series["base"]["step_translation_m"][peak_base]),
            "largest_step_ending_state": peak_base,
            "largest_step_ending_simulation_time_s": peak_base / fps,
        },
        "actuator_target_continuity": _target_metrics(capture, len(poses), fps),
        "sparse_trace_metrics": _trace_metrics(trace, env_id, fps),
        "phases": phases,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _hash(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--playback-speed", type=float, default=1.5, help="Presentation mapping only; derivatives always use capture fps.")
    parser.add_argument("--extraction-plane-x-m", type=float)
    parser.add_argument("--stall-radius-m", type=float, default=0.005)
    parser.add_argument("--stall-speed-mps", type=float, default=0.005)
    args = parser.parse_args()
    result = analyze_capture(
        _load_npz(args.poses), json.loads(args.report.read_text(encoding="utf-8")),
        _load_npz(args.trace) if args.trace else None, env_id=args.env_id,
        playback_speed=args.playback_speed, plane_x_m=args.extraction_plane_x_m,
        stall_radius_m=args.stall_radius_m, stall_speed_mps=args.stall_speed_mps,
    )
    inputs = {"body_poses": args.poses, "workflow_report": args.report}
    if args.trace:
        inputs["handoff_trace"] = args.trace
    result["input_artifacts"] = {name: _hash(path) for name, path in inputs.items()}
    result["analysis_source"] = {
        "analyzer": _hash(Path(__file__)), "geometry": _hash(Path(grapple_geometry.__file__)),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() in {path.resolve() for path in inputs.values()}:
            raise ValueError("Output must not overwrite an input artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Motion analysis written to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
