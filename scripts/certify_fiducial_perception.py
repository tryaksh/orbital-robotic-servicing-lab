"""Gate calibrated RGB fiducial pose and bay occupancy on a rendered corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from zero_g_blade_swap.fiducial import estimate_fiducial_pose
from zero_g_blade_swap.grapple_geometry import EXTRACTED_BLADE_CENTRE_X
from zero_g_blade_swap.servicing_camera import (
    CAMERA_FOCAL_LENGTH_MM,
    CAMERA_HORIZONTAL_APERTURE_MM,
    CAMERA_QUATERNION_WXYZ_ROS,
)
from zero_g_blade_swap.servicing_camera import (
    CAMERA_POSITION_M as CAMERA_POSITION_TUPLE_M,
)

CAMERA_POSITION_M = np.asarray(CAMERA_POSITION_TUPLE_M, dtype=np.float64)
CAMERA_QUATERNION_WXYZ = np.asarray(CAMERA_QUATERNION_WXYZ_ROS, dtype=np.float64)
FOCAL_LENGTH_MM = CAMERA_FOCAL_LENGTH_MM
HORIZONTAL_APERTURE_MM = CAMERA_HORIZONTAL_APERTURE_MM
SLOT_CENTRES_Y_M = np.asarray((0.0, -0.22), dtype=np.float64)
BAY_OCCUPANCY_HALF_WIDTH_M = 0.0725
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCES = (
    Path("src/zero_g_blade_swap/servicing_camera.py"),
    Path("src/zero_g_blade_swap/fiducial.py"),
    Path("src/zero_g_blade_swap/tasks/blade_swap/assets.py"),
    Path("src/zero_g_blade_swap/tasks/blade_swap/mdp/perception.py"),
    Path("src/zero_g_blade_swap/tasks/blade_swap/scene_cfg.py"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--position_p95_limit_mm", type=float, default=20.0)
    parser.add_argument("--orientation_p95_limit_rad", type=float, default=0.05)
    parser.add_argument(
        "--detection_rate_min",
        type=float,
        default=0.99,
        help="Minimum detection rate while a module is in either rack bay.",
    )
    parser.add_argument(
        "--overall_detection_rate_min",
        type=float,
        default=0.90,
        help="Minimum detection rate across rack and arm-occluded transfer poses.",
    )
    parser.add_argument("--occupancy_exact_min", type=float, default=0.95)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _intrinsic_matrix(width: int = 384, height: int = 384) -> np.ndarray:
    focal_pixels = FOCAL_LENGTH_MM * width / HORIZONTAL_APERTURE_MM
    return np.asarray(
        (
            (focal_pixels, 0.0, 0.5 * (width - 1)),
            (0.0, focal_pixels, 0.5 * (height - 1)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _occupancy(position_local: np.ndarray) -> np.ndarray:
    return (
        (position_local[0] > EXTRACTED_BLADE_CENTRE_X)
        & (np.abs(position_local[1] - SLOT_CENTRES_Y_M) <= BAY_OCCUPANCY_HALF_WIDTH_M)
    ).astype(np.float32)


def main() -> int:
    args = _parser().parse_args()
    with np.load(args.dataset, allow_pickle=False) as corpus:
        images = corpus["images"]
        labels = corpus["labels"]
        occupancy_labels = corpus.get("occupancy", None)
        depth_images = corpus.get("depth_image_plane_m", None)

    image_height, image_width = images.shape[1:3]
    intrinsic = _intrinsic_matrix(image_width, image_height)
    rotation_world_from_camera = _quaternion_matrix(CAMERA_QUATERNION_WXYZ)
    position_errors_mm: list[float] = []
    orientation_errors_rad: list[float] = []
    reprojection_errors_px: list[float] = []
    confidences: list[float] = []
    occupancy_correct = 0
    critical_frames = 0
    critical_detections = 0
    critical_occupancy_correct = 0
    failures: list[dict[str, object]] = []

    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        is_critical_bay_frame = bool(occupancy_labels is not None and float(occupancy_labels[index].sum()) == 1.0)
        critical_frames += int(is_critical_bay_frame)
        try:
            depth = None if depth_images is None else depth_images[index]
            estimate = estimate_fiducial_pose(image, intrinsic, depth)
        except (RuntimeError, ValueError) as error:
            failures.append({"index": index, "error": str(error)})
            continue
        critical_detections += int(is_critical_bay_frame)

        predicted_position_world = CAMERA_POSITION_M + rotation_world_from_camera @ estimate.position_camera_m
        true_rotation_world, _ = cv2.Rodrigues(label[3:].astype(np.float64))
        predicted_rotation_world = rotation_world_from_camera @ estimate.rotation_camera_from_object
        relative_rotation = predicted_rotation_world @ true_rotation_world.T
        relative_rotation_vector, _ = cv2.Rodrigues(relative_rotation)
        position_errors_mm.append(float(1_000.0 * np.linalg.norm(predicted_position_world - label[:3])))
        orientation_errors_rad.append(float(np.linalg.norm(relative_rotation_vector)))
        reprojection_errors_px.append(estimate.reprojection_error_px)
        confidences.append(estimate.confidence)
        if occupancy_labels is not None:
            correct = int(np.array_equal(_occupancy(predicted_position_world), occupancy_labels[index]))
            occupancy_correct += correct
            critical_occupancy_correct += correct * int(is_critical_bay_frame)

    detected = len(position_errors_mm)
    total = int(len(images))
    detection_rate = detected / total if total else 0.0
    critical_detection_rate = critical_detections / critical_frames if critical_frames else 0.0

    def statistics(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "p50": None, "p95": None, "max": None}
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
        }

    position = statistics(position_errors_mm)
    orientation = statistics(orientation_errors_rad)
    occupancy_exact_detected = occupancy_correct / detected if occupancy_labels is not None and detected else None
    occupancy_exact = (
        critical_occupancy_correct / critical_detections
        if occupancy_labels is not None and critical_detections
        else None
    )
    passed = bool(
        detection_rate >= args.overall_detection_rate_min
        and critical_detection_rate >= args.detection_rate_min
        and position["p95"] is not None
        and position["p95"] <= args.position_p95_limit_mm
        and orientation["p95"] is not None
        and orientation["p95"] <= args.orientation_p95_limit_rad
        and occupancy_exact is not None
        and occupancy_exact >= args.occupancy_exact_min
    )
    report = {
        "status": "passed" if passed else "failed",
        "title": "Calibrated RGB-D fiducial perception qualification",
        "evidence_type": "rendered_rgbd_fiducial_pose_heldout_gate",
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "frames": total,
        "detected_frames": detected,
        "detection_rate": detection_rate,
        "critical_bay_frames": critical_frames,
        "critical_bay_detections": critical_detections,
        "critical_bay_detection_rate": critical_detection_rate,
        "position_error_mm": position,
        "orientation_error_rad": orientation,
        "reprojection_error_px": statistics(reprojection_errors_px),
        "detector_quality_score": statistics(confidences),
        "occupancy_exact_match": occupancy_exact,
        "occupancy_exact_match_detected_full_envelope": occupancy_exact_detected,
        "gates": {
            "critical_bay_detection_rate_min": args.detection_rate_min,
            "overall_detection_rate_min": args.overall_detection_rate_min,
            "position_p95_limit_mm": args.position_p95_limit_mm,
            "orientation_p95_limit_rad": args.orientation_p95_limit_rad,
            "occupancy_exact_min": args.occupancy_exact_min,
        },
        "calibration": {
            "camera_position_m": CAMERA_POSITION_M.tolist(),
            "camera_quaternion_wxyz_ros": CAMERA_QUATERNION_WXYZ.tolist(),
            "intrinsic_matrix": intrinsic.tolist(),
            "resolution_px": [image_width, image_height],
            "source": "scene_cfg.make_tiled_camera_cfg",
        },
        "deployment_boundary": {
            "runtime_inputs": [
                "rgb",
                "registered_metric_depth",
                "camera_intrinsics",
                "camera_extrinsics",
                "known_fiducial_geometry",
            ],
            "simulator_pose_used_by_estimator": False,
            "simulator_pose_used_only_for_error_metrics": True,
            "dropout_behavior": (
                "hold-last before verified robot capture; tool forward-kinematics propagation only "
                "after capture; hold-last and guarded-motion pause after handoff"
            ),
            "critical_bay_definition": (
                "source occupancy preflight and destination insertion alignment; the collector holds "
                "the robot still, so continuous robot occlusion is exercised only by the strict RGB-D chain"
            ),
        },
        "runtime_source_bindings": [
            {"path": path.as_posix(), "sha256": _sha256(PROJECT_ROOT / path)} for path in RUNTIME_SOURCES
        ],
        "failure_examples": failures[:20],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
