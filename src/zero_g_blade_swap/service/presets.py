"""Fixed, auditable compute presets and live capability checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from zero_g_blade_swap import __version__

from .config import ServiceSettings
from .models import BackendKind, Capabilities, InputProvenance, JobProvenance, PresetCapability

GRASP = Path(
    "logs/rl_games/zero_g_blade_insertion_contact/grapple_grasp_l0_seed70_v6w65/nn/"
    "last_zero_g_blade_insertion_contact_ep_2400_rew__37.24023_.pth"
)
EXTRACT = Path(
    "logs/rl_games/zero_g_blade_insertion_contact/grapple_extract_l0_seed70_v16w65/nn/"
    "last_zero_g_blade_insertion_contact_ep_9700_rew__176.34572_.pth"
)
INSERT_W65_TWO_SLOT = Path(
    "logs/rl_games/zero_g_blade_insertion_contact/grapple_insert_l0_seed70_v12w65/nn/"
    "last_zero_g_blade_insertion_contact_ep_7100_rew_-20.706831.pth"
)
FIDUCIAL_EVIDENCE = Path("evidence/fiducial_rgbd_service_plate.json")
FULL_CHAIN_EVIDENCE = Path("evidence/full_chain_rgbd_service_seed4070.json")
FIDUCIAL_SOURCE = Path("src/zero_g_blade_swap/fiducial.py")
ASSET_SOURCE = Path("src/zero_g_blade_swap/tasks/blade_swap/assets.py")
PERCEPTION_SOURCE = Path("src/zero_g_blade_swap/tasks/blade_swap/mdp/perception.py")
CAMERA_CONFIG_SOURCE = Path("src/zero_g_blade_swap/tasks/blade_swap/scene_cfg.py")
WORKCELL_CONFIG_SOURCE = Path("src/zero_g_blade_swap/tasks/blade_swap/grapple_pin_env_cfg.py")
# Legacy pose-head validator inputs remain defined for reading old evidence;
# they are not part of the live preset or its provenance.
POSE_HEAD_W65_OVERVIEW = Path("checkpoints/module_pose_head_two_slot_w65_overview.pth")
POSE_HEAD_W65_OVERVIEW_EVIDENCE = Path("evidence/module_pose_head_two_slot_w65_overview.json")
CAMERA_SCALE_W65_OVERVIEW_EVIDENCE = Path("evidence/camera_scale_grapple_w65.json")
WORKFLOW_SCRIPT = Path("scripts/run_workflow_demo.py")
LIVE_TASK_ID = "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0"
COLLECTION_TASK_ID = "Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Collect-v0"
OVERVIEW_IMAGE_SHAPE_HWC = [384, 384, 3]
OVERVIEW_POSE_DISTRIBUTION = "workflow_envelope"
MIN_HELD_OUT_FRAMES = 1_000
MIN_TRAINING_FRAMES = 4_000
MAX_POSE_P95_MM = 20.0
MAX_ORIENTATION_P95_RAD = 0.05
MIN_OVERALL_DETECTION_RATE = 0.90
MIN_CRITICAL_DETECTION_RATE = 0.99
MIN_OCCUPANCY_EXACT_MATCH = 0.95
COLLECTION_POSE_HOLD = "kinematic_reasserted_each_control_step"
MAX_CAPTURE_POSITION_DRIFT_M = 1.0e-4
MAX_CAPTURE_ORIENTATION_DRIFT_RAD = 1.0e-4

# One declaration drives both capability checks and provenance. A newly added
# command-line input therefore cannot accidentally be omitted from readiness.
LIVE_INPUT_REQUIREMENTS = (
    ("workflow driver", "workflow_driver", WORKFLOW_SCRIPT),
    ("w65 capture checkpoint", "capture_policy", GRASP),
    ("w65 extract checkpoint", "extract_policy", EXTRACT),
    ("w65 two-slot insert checkpoint", "insert_policy", INSERT_W65_TWO_SLOT),
    ("RGB-D fiducial evidence", "perception_evidence", FIDUCIAL_EVIDENCE),
    ("successful full-chain evidence", "workflow_evidence", FULL_CHAIN_EVIDENCE),
    ("fiducial estimator", "fiducial_estimator", FIDUCIAL_SOURCE),
    ("fiducial service-plate asset", "service_plate_asset", ASSET_SOURCE),
    ("perception integration", "perception_integration", PERCEPTION_SOURCE),
    ("RGB-D camera configuration", "camera_config", CAMERA_CONFIG_SOURCE),
    ("service-workcell configuration", "workcell_config", WORKCELL_CONFIG_SOURCE),
)


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    preset_id: str
    preset_title: str
    preset_revision: str
    backend: BackendKind
    seed: int
    artifact_dir: Path
    argv: tuple[str, ...] | None
    cwd: Path
    environment: dict[str, str]
    input_files: tuple[tuple[str, Path], ...]


@dataclass(frozen=True, slots=True)
class Preset:
    id: str
    title: str
    description: str
    revision: str
    backend: BackendKind
    estimated_runtime_s: int
    produces_video: bool
    perception: bool


class PresetUnavailableError(RuntimeError):
    def __init__(self, preset_id: str, reasons: list[str]) -> None:
        self.preset_id = preset_id
        self.reasons = reasons
        super().__init__(f"Preset '{preset_id}' is unavailable: {'; '.join(reasons)}")


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return len(normalized) == 64 and all(character in "0123456789abcdef" for character in normalized)


class PresetRegistry:
    """Registry whose command lines are code-owned rather than request-owned."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._presets = {
            "replay_full_chain": Preset(
                id="replay_full_chain",
                title="Representative service-path replay",
                description=(
                    "A synthetic, deterministic event replay covering perception through verification. "
                    "It exercises the compute-service and dashboard only; it is not a simulation run or success evidence."
                ),
                revision="representative-replay-v2",
                backend=BackendKind.REPLAY,
                estimated_runtime_s=4,
                produces_video=False,
                perception=True,
            ),
            "isaac_full_chain_perception": Preset(
                id="isaac_full_chain_perception",
                title="Live RGB-D compute-module service run",
                description=(
                    "Runs the measured two-bay Isaac workflow: calibrated RGB-D fiducial perception, "
                    "visual occupancy planning, learned capture and extraction, physical payload-shuttle "
                    "transfer, guarded insertion, settling verification, telemetry, video, and artifacts."
                ),
                revision="isaac-rgbd-serviceability-v1",
                backend=BackendKind.ISAAC,
                estimated_runtime_s=480,
                produces_video=True,
                perception=True,
            ),
        }

    def get(self, preset_id: str) -> Preset:
        try:
            return self._presets[preset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown preset '{preset_id}'") from exc

    @property
    def presets(self) -> tuple[Preset, ...]:
        return tuple(self._presets.values())

    def _project_path(self, relative: Path) -> Path:
        return (self.settings.project_root / relative).resolve()

    def _gpu_probe(self) -> tuple[bool, str | None]:
        candidates = [shutil.which("nvidia-smi")]
        if os.name == "nt":
            candidates.extend(
                [
                    r"C:\Windows\System32\nvidia-smi.exe",
                    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                ]
            )
        executable = next((str(path) for path in candidates if path and Path(path).is_file()), None)
        if executable is None:
            return False, None
        try:
            completed = subprocess.run(  # noqa: S603 - fixed executable and fixed arguments
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return False, None
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return completed.returncode == 0 and bool(names), ", ".join(names) or None

    def _unavailable_reasons(self, preset: Preset, gpu_available: bool) -> list[str]:
        if preset.backend == BackendKind.REPLAY:
            return []
        reasons: list[str] = []
        if not self.settings.isaac_python.is_file():
            reasons.append(f"Isaac Python launcher not found: {self.settings.isaac_python}")
        for label, _role, relative in LIVE_INPUT_REQUIREMENTS:
            if not self._project_path(relative).is_file():
                reasons.append(f"Missing {label}: {relative.as_posix()}")
        evidence_path = self._project_path(FIDUCIAL_EVIDENCE)
        if evidence_path.is_file():
            reasons.extend(self._fiducial_evidence_reasons(evidence_path))
        workflow_evidence_path = self._project_path(FULL_CHAIN_EVIDENCE)
        if workflow_evidence_path.is_file():
            reasons.extend(self._workflow_evidence_reasons(workflow_evidence_path))
        if not gpu_available:
            reasons.append("No responsive NVIDIA GPU was detected")
        return reasons

    def _runtime_binding_reasons(
        self,
        evidence: dict[str, object],
        expected: tuple[Path, ...],
        label: str,
    ) -> list[str]:
        rows = evidence.get("runtime_source_bindings")
        if not isinstance(rows, list):
            return [f"{label} does not bind its runtime source files"]
        bindings = {
            row.get("path"): row.get("sha256")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("path"), str)
        }
        reasons: list[str] = []
        for path in expected:
            recorded = bindings.get(path.as_posix())
            if not _valid_sha256(recorded):
                reasons.append(f"{label} does not bind {path.as_posix()}")
                continue
            current = self._project_path(path)
            if current.is_file() and sha256_file(current) != str(recorded).strip().lower():
                reasons.append(f"{label} is stale for {path.as_posix()}")
        return reasons

    def _fiducial_evidence_reasons(self, evidence_path: Path) -> list[str]:
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"Invalid RGB-D perception evidence: {type(exc).__name__}"]
        if not isinstance(evidence, dict):
            return ["Invalid RGB-D perception evidence: root must be an object"]
        reasons: list[str] = []
        if evidence.get("status") != "passed":
            reasons.append("RGB-D perception evidence has not passed")
        if evidence.get("evidence_type") != "rendered_rgbd_fiducial_pose_heldout_gate":
            reasons.append("RGB-D perception evidence has the wrong evidence type")
        frames = _nonnegative_int(evidence.get("frames"))
        if frames is None or frames < MIN_HELD_OUT_FRAMES:
            reasons.append(f"RGB-D perception evidence requires at least {MIN_HELD_OUT_FRAMES} frames")
        overall = _finite_float(evidence.get("detection_rate"))
        critical = _finite_float(evidence.get("critical_bay_detection_rate"))
        position = evidence.get("position_error_mm")
        orientation = evidence.get("orientation_error_rad")
        pose_p95 = _finite_float(position.get("p95")) if isinstance(position, dict) else None
        orientation_p95 = _finite_float(orientation.get("p95")) if isinstance(orientation, dict) else None
        occupancy = _finite_float(evidence.get("occupancy_exact_match"))
        if overall is None or overall < MIN_OVERALL_DETECTION_RATE:
            reasons.append(f"RGB-D full-envelope detection rate must be at least {MIN_OVERALL_DETECTION_RATE:.2f}")
        if critical is None or critical < MIN_CRITICAL_DETECTION_RATE:
            reasons.append(f"RGB-D critical-bay detection rate must be at least {MIN_CRITICAL_DETECTION_RATE:.2f}")
        if pose_p95 is None or not 0.0 <= pose_p95 < MAX_POSE_P95_MM:
            reasons.append(f"RGB-D position p95 must be below {MAX_POSE_P95_MM:g} mm")
        if orientation_p95 is None or not 0.0 <= orientation_p95 < MAX_ORIENTATION_P95_RAD:
            reasons.append(f"RGB-D orientation p95 must be below {MAX_ORIENTATION_P95_RAD:g} rad")
        if occupancy is None or not MIN_OCCUPANCY_EXACT_MATCH <= occupancy <= 1.0:
            reasons.append(f"RGB-D bay occupancy exact match must be at least {MIN_OCCUPANCY_EXACT_MATCH:.2f}")
        if not _valid_sha256(evidence.get("dataset_sha256")):
            reasons.append("RGB-D perception evidence does not identify its rendered corpus by SHA256")
        calibration = evidence.get("calibration")
        if not isinstance(calibration, dict) or calibration.get("resolution_px") != [384, 384]:
            reasons.append("RGB-D perception evidence is not for the deployed 384x384 camera")
        boundary = evidence.get("deployment_boundary")
        runtime_inputs = boundary.get("runtime_inputs") if isinstance(boundary, dict) else None
        if not isinstance(runtime_inputs, list) or "registered_metric_depth" not in runtime_inputs:
            reasons.append("RGB-D perception evidence does not require registered metric depth")
        reasons.extend(
            self._runtime_binding_reasons(
                evidence,
                (FIDUCIAL_SOURCE, ASSET_SOURCE, PERCEPTION_SOURCE, CAMERA_CONFIG_SOURCE),
                "RGB-D perception evidence",
            )
        )
        return reasons

    def _workflow_evidence_reasons(self, evidence_path: Path) -> list[str]:
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"Invalid full-chain evidence: {type(exc).__name__}"]
        if not isinstance(evidence, dict):
            return ["Invalid full-chain evidence: root must be an object"]
        reasons: list[str] = []
        if (
            evidence.get("task") != LIVE_TASK_ID
            or evidence.get("completed") is not True
            or evidence.get("reached_phase") != "done"
            or evidence.get("predicate_fired") is not True
            or evidence.get("seated_conditions_still_held_after_settling") is not True
        ):
            reasons.append("full-chain RGB-D evidence is not a settled successful relocation")
        if evidence.get("visual_randomization") != "on":
            reasons.append("full-chain evidence did not retain visual randomization")
        perception = evidence.get("perception")
        if not isinstance(perception, dict) or perception.get("source") != "rgb_fiducial_calibrated_pnp":
            reasons.append("full-chain evidence did not execute calibrated RGB-D fiducial perception")
        planning = evidence.get("planning")
        if not isinstance(planning, dict) or planning.get("source_occupied_destination_clear") is not True:
            reasons.append("full-chain evidence did not pass the visual occupancy planning gate")
        terminal_occupancy = perception.get("terminal_bay_occupancy_scores") if isinstance(perception, dict) else None
        if (
            not isinstance(terminal_occupancy, list)
            or len(terminal_occupancy) != 2
            or _finite_float(terminal_occupancy[0]) != 0.0
            or _finite_float(terminal_occupancy[1]) != 1.0
        ):
            reasons.append("full-chain evidence does not finish with destination-bay occupancy")
        digests = evidence.get("checkpoint_sha256")
        policies = {"capture": GRASP, "extract": EXTRACT, "insert": INSERT_W65_TWO_SLOT}
        if not isinstance(digests, dict):
            reasons.append("full-chain evidence does not bind policy checkpoints")
        else:
            for name, path in policies.items():
                recorded = digests.get(name)
                current = self._project_path(path)
                if not _valid_sha256(recorded) or (
                    current.is_file() and sha256_file(current) != str(recorded).strip().lower()
                ):
                    reasons.append(f"full-chain evidence is stale for the {name} policy")
        reasons.extend(
            self._runtime_binding_reasons(
                evidence,
                (
                    WORKFLOW_SCRIPT,
                    FIDUCIAL_SOURCE,
                    ASSET_SOURCE,
                    PERCEPTION_SOURCE,
                    CAMERA_CONFIG_SOURCE,
                    WORKCELL_CONFIG_SOURCE,
                ),
                "full-chain evidence",
            )
        )
        return reasons

    def _pose_head_evidence_reasons(self, evidence_path: Path) -> list[str]:
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"Invalid w65 overview pose-head evidence: {type(exc).__name__}"]
        if not isinstance(evidence, dict):
            return ["Invalid w65 overview pose-head evidence: root must be an object"]
        if evidence.get("status") != "passed":
            return [
                "W65 overview pose-head evidence has not passed its declared gate "
                f"(status={evidence.get('status', 'missing')!r})"
            ]
        reasons: list[str] = []
        if evidence.get("evidence_type") != "simulation_perception_regression":
            reasons.append("W65 overview pose-head evidence has the wrong evidence_type")

        gates = evidence.get("gates") if isinstance(evidence.get("gates"), dict) else {}
        pose_gate = gates.get("pose_p95") if isinstance(gates.get("pose_p95"), dict) else {}
        occupancy_gate = (
            gates.get("occupancy_exact_match") if isinstance(gates.get("occupancy_exact_match"), dict) else {}
        )
        held_out = evidence.get("held_out") if isinstance(evidence.get("held_out"), dict) else {}
        protocol = evidence.get("protocol") if isinstance(evidence.get("protocol"), dict) else {}
        if gates.get("promotion_passed") is not True or pose_gate.get("passed") is not True:
            reasons.append("W65 overview pose-head evidence does not pass its pose promotion gate")
        pose_p95 = _finite_float(held_out.get("position_error_mm_p95"))
        if pose_p95 is None or pose_p95 < 0.0 or pose_p95 >= MAX_POSE_P95_MM:
            reasons.append(f"W65 overview held-out pose p95 must be below {MAX_POSE_P95_MM:g} mm")
        occupancy_exact = _finite_float(held_out.get("occupancy_exact_match"))
        majority_pattern = _finite_float(held_out.get("occupancy_majority_pattern_rate"))
        if (
            occupancy_gate.get("applies") is not True
            or occupancy_gate.get("passed") is not True
            or held_out.get("occupancy_bays") != 2
            or occupancy_exact is None
            or not 0.0 <= occupancy_exact <= 1.0
            or occupancy_exact < MIN_OCCUPANCY_EXACT_MATCH
            or majority_pattern is None
            or not 0.0 <= majority_pattern <= 1.0
            or occupancy_exact <= majority_pattern
        ):
            reasons.append(
                "W65 overview occupancy evidence must cover two bays, reach at least "
                f"{MIN_OCCUPANCY_EXACT_MATCH:.2f} exact-match, and beat its majority-pattern baseline"
            )
        if (
            _nonnegative_int(protocol.get("training_frames")) is None
            or int(protocol.get("training_frames", -1)) < MIN_TRAINING_FRAMES
        ):
            reasons.append(f"W65 overview evidence requires at least {MIN_TRAINING_FRAMES} training frames")
        if (
            _nonnegative_int(protocol.get("held_out_frames")) is None
            or int(protocol.get("held_out_frames", -1)) < MIN_HELD_OUT_FRAMES
        ):
            reasons.append(f"W65 overview evidence requires at least {MIN_HELD_OUT_FRAMES} held-out frames")

        provenance = evidence.get("dataset_provenance") if isinstance(evidence.get("dataset_provenance"), dict) else {}
        metadata = provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {}
        image_signal = provenance.get("image_signal") if isinstance(provenance.get("image_signal"), dict) else {}
        dataset_sha256 = provenance.get("sha256")
        if not _valid_sha256(dataset_sha256):
            reasons.append("W65 overview evidence does not identify its training dataset by SHA256")
        if provenance.get("image_shape") != [protocol.get("frames"), *OVERVIEW_IMAGE_SHAPE_HWC]:
            reasons.append("W65 overview evidence dataset image shape is not the declared 256x256 RGB corpus")
        if image_signal.get("passed") is not True:
            reasons.append("W65 overview evidence dataset did not pass its rendered-image signal gate")
        settle_steps = _nonnegative_int(metadata.get("collection_settle_steps_minimum"))
        if settle_steps is None or settle_steps < 2:
            reasons.append("W65 overview dataset did not wait for the 15 Hz camera cadence")
        position_drift = _finite_float(metadata.get("capture_position_drift_max_m"))
        orientation_drift = _finite_float(metadata.get("capture_orientation_drift_max_rad"))
        position_limit = _finite_float(metadata.get("capture_position_drift_limit_m"))
        orientation_limit = _finite_float(metadata.get("capture_orientation_drift_limit_rad"))
        if (
            metadata.get("collection_pose_hold") != COLLECTION_POSE_HOLD
            or metadata.get("frame_label_sync_gate_passed") is not True
            or position_drift is None
            or not 0.0 <= position_drift <= MAX_CAPTURE_POSITION_DRIFT_M
            or orientation_drift is None
            or not 0.0 <= orientation_drift <= MAX_CAPTURE_ORIENTATION_DRIFT_RAD
            or position_limit is None
            or position_limit != MAX_CAPTURE_POSITION_DRIFT_M
            or orientation_limit is None
            or orientation_limit != MAX_CAPTURE_ORIENTATION_DRIFT_RAD
        ):
            reasons.append(
                "W65 overview dataset does not prove synchronized frame/label capture under the "
                "required kinematic pose hold"
            )
        camera_offset = _finite_float(metadata.get("camera_offset_mm"))
        camera_tilt = _finite_float(metadata.get("camera_tilt_mrad"))
        if camera_offset != 0.0 or camera_tilt != 0.0:
            reasons.append("W65 overview deployment head must be trained on the nominal camera mount")

        contract = evidence.get("deployment_contract") if isinstance(evidence.get("deployment_contract"), dict) else {}
        if contract.get("deployment_task_id") != LIVE_TASK_ID:
            reasons.append("W65 overview evidence is not bound to the live two-bay workflow task")
        if contract.get("collection_task_id") != COLLECTION_TASK_ID:
            reasons.append("W65 overview evidence is not bound to the two-bay collection task")
        if contract.get("image_shape_hwc") != OVERVIEW_IMAGE_SHAPE_HWC:
            reasons.append("W65 overview evidence does not declare a 256x256 RGB deployment input")
        if contract.get("pose_distribution") != OVERVIEW_POSE_DISTRIBUTION:
            reasons.append("W65 overview evidence was not trained on the workflow envelope")
        reasons.extend(
            self._bound_file_reasons(
                contract.get("camera_config_source"),
                CAMERA_CONFIG_SOURCE,
                "camera configuration",
            )
        )
        reasons.extend(
            self._bound_file_reasons(
                contract.get("camera_scale_evidence"),
                CAMERA_SCALE_W65_OVERVIEW_EVIDENCE,
                "camera-scale evidence",
            )
        )

        recorded_checkpoint = evidence.get("checkpoint")
        if not isinstance(recorded_checkpoint, str) or not recorded_checkpoint.strip():
            reasons.append("W65 overview pose-head evidence does not identify its checkpoint")
            return reasons
        try:
            recorded_path = Path(recorded_checkpoint)
        except (TypeError, ValueError):
            reasons.append("W65 overview pose-head evidence contains an invalid checkpoint path")
            return reasons
        if not recorded_path.is_absolute():
            recorded_path = self._project_path(recorded_path)
        if recorded_path.resolve() != self._project_path(POSE_HEAD_W65_OVERVIEW):
            reasons.append(f"W65 overview pose-head evidence names a different checkpoint: {recorded_checkpoint}")
            return reasons
        recorded_sha256 = evidence.get("checkpoint_sha256")
        if not isinstance(recorded_sha256, str) or not recorded_sha256.strip():
            reasons.append("W65 overview pose-head evidence does not record checkpoint_sha256")
            return reasons
        if not _valid_sha256(recorded_sha256):
            reasons.append("W65 overview pose-head evidence contains an invalid checkpoint_sha256")
            return reasons
        normalized_sha256 = recorded_sha256.strip().lower()
        checkpoint_path = self._project_path(POSE_HEAD_W65_OVERVIEW)
        if checkpoint_path.is_file() and sha256_file(checkpoint_path) != normalized_sha256:
            reasons.append("W65 overview pose-head evidence checkpoint_sha256 does not match the current weights")
        return reasons

    def _bound_file_reasons(self, record: object, expected: Path, label: str) -> list[str]:
        if not isinstance(record, dict):
            return [f"W65 overview evidence does not bind its {label}"]
        recorded_path = record.get("path")
        normalized_path = recorded_path.replace("\\", "/") if isinstance(recorded_path, str) else None
        if normalized_path != expected.as_posix():
            return [f"W65 overview evidence names a different {label}"]
        recorded_sha256 = record.get("sha256")
        if not _valid_sha256(recorded_sha256):
            return [f"W65 overview evidence has no valid {label} SHA256"]
        current = self._project_path(expected)
        if current.is_file() and sha256_file(current) != recorded_sha256.strip().lower():
            return [f"W65 overview evidence {label} SHA256 is stale"]
        return []

    @staticmethod
    def _camera_scale_evidence_reasons(evidence_path: Path) -> list[str]:
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"Invalid w65 overview camera-scale evidence: {type(exc).__name__}"]
        if not isinstance(evidence, dict):
            return ["Invalid w65 overview camera-scale evidence: root must be an object"]
        protocol = evidence.get("protocol") if isinstance(evidence.get("protocol"), dict) else {}
        framing = evidence.get("framing") if isinstance(evidence.get("framing"), dict) else {}
        in_frame = framing.get("in_frame") if isinstance(framing.get("in_frame"), dict) else {}
        gate = evidence.get("gate") if isinstance(evidence.get("gate"), dict) else {}
        reasons = []
        if evidence.get("status") != "passed" or gate.get("passed") is not True:
            reasons.append("W65 overview camera-scale evidence has not passed")
        if protocol.get("task") != COLLECTION_TASK_ID or protocol.get("resolution_px") != 256:
            reasons.append("W65 overview camera-scale evidence describes a different sensor task")
        required_points = {"first_slot_mouth", "second_slot_mouth", "transfer_clear_midpoint", "module_centre"}
        if not required_points.issubset({name for name, visible in in_frame.items() if visible is True}):
            reasons.append("W65 overview camera-scale evidence does not frame the full transfer envelope")
        return reasons

    def capabilities(self) -> Capabilities:
        gpu_available, gpu_name = self._gpu_probe()
        preset_rows = []
        for preset in self.presets:
            reasons = self._unavailable_reasons(preset, gpu_available)
            preset_rows.append(
                PresetCapability(
                    id=preset.id,
                    title=preset.title,
                    description=preset.description,
                    backend=preset.backend,
                    available=not reasons,
                    unavailable_reasons=reasons,
                    estimated_runtime_s=preset.estimated_runtime_s,
                    produces_video=preset.produces_video,
                    perception=preset.perception,
                )
            )
        isaac_row = next(row for row in preset_rows if row.backend == BackendKind.ISAAC)
        return Capabilities(
            service_version=__version__,
            isaac_python=str(self.settings.isaac_python),
            isaac_available=isaac_row.available,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            presets=preset_rows,
        )

    def build(self, preset_id: str, seed: int, artifact_dir: Path) -> ExecutionSpec:
        preset = self.get(preset_id)
        reasons: list[str] = []
        if preset.backend == BackendKind.ISAAC:
            gpu_available, _ = self._gpu_probe()
            reasons = self._unavailable_reasons(preset, gpu_available)
        if reasons:
            raise PresetUnavailableError(preset_id, reasons)

        input_files: tuple[tuple[str, Path], ...] = ()
        argv: tuple[str, ...] | None = None
        if preset.backend == BackendKind.ISAAC:
            input_files = tuple(
                (role, self._project_path(relative)) for _label, role, relative in LIVE_INPUT_REQUIREMENTS
            )
            report = artifact_dir / "workflow_report.json"
            video = artifact_dir / "video"
            argv = (
                str(self.settings.isaac_python),
                str(self._project_path(WORKFLOW_SCRIPT)),
                "--headless",
                "--workflow",
                "relocate",
                "--task",
                LIVE_TASK_ID,
                "--curriculum_stage",
                "0",
                "--grasp_checkpoint",
                str(self._project_path(GRASP)),
                "--extract_checkpoint",
                str(self._project_path(EXTRACT)),
                "--insert_checkpoint",
                str(self._project_path(INSERT_W65_TWO_SLOT)),
                "--perception_backend",
                "fiducial_pnp",
                "--base_rail_on_relocation",
                "--num_envs",
                "1",
                "--seed",
                str(seed),
                "--steps",
                "3600",
                "--settle_steps",
                "30",
                "--inspection_view",
                "workcell",
                "--video",
                "--video_dir",
                str(video),
                "--handoff_trace",
                str(artifact_dir / "handoff_trace.npz"),
                "--report",
                str(report),
            )
        return ExecutionSpec(
            preset_id=preset.id,
            preset_title=preset.title,
            preset_revision=preset.revision,
            backend=preset.backend,
            seed=seed,
            artifact_dir=artifact_dir,
            argv=argv,
            cwd=self.settings.project_root,
            environment={"PYTHONUNBUFFERED": "1"},
            input_files=input_files,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(project_root: Path) -> str | None:
    """Read the current revision without invoking Git or writing outside runtime."""

    head = project_root / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if not value.startswith("ref: "):
            return value or None
        reference = project_root / ".git" / value.removeprefix("ref: ")
        if reference.is_file():
            return reference.read_text(encoding="utf-8").strip() or None
        packed = project_root / ".git" / "packed-refs"
        if packed.is_file():
            suffix = value.removeprefix("ref: ")
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {suffix}"):
                    return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


def _git_dirty(project_root: Path) -> bool | None:
    git = shutil.which("git")
    if git is None or not (project_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and fixed arguments
            [git, "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def provenance_for(spec: ExecutionSpec) -> JobProvenance:
    inputs = [
        InputProvenance(
            role=role,
            path=path.relative_to(spec.cwd).as_posix(),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for role, path in spec.input_files
    ]
    return JobProvenance(
        service_version=__version__,
        source_revision=_git_revision(spec.cwd),
        source_dirty=_git_dirty(spec.cwd),
        preset_revision=spec.preset_revision,
        backend=spec.backend,
        command_argv=list(spec.argv) if spec.argv else None,
        inputs=inputs,
    )
