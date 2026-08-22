from __future__ import annotations

import cv2
import numpy as np
import pytest

from zero_g_blade_swap.fiducial import (
    FIDUCIAL_DICTIONARY,
    FIDUCIAL_MARKER_ID,
    FIDUCIAL_TAG_BASIS_MODULE,
    FIDUCIAL_TAG_CENTER_M,
    FIDUCIAL_TAG_SIZE_M,
    estimate_fiducial_pose,
)


def _synthetic_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    intrinsic = np.asarray(((524.0, 0.0, 127.5), (0.0, 524.0, 127.5), (0.0, 0.0, 1.0)))
    rotation_module_from_tag = np.asarray(FIDUCIAL_TAG_BASIS_MODULE)
    rotation_camera_from_tag = np.diag((1.0, -1.0, -1.0))
    rotation_camera_from_module = rotation_camera_from_tag @ rotation_module_from_tag.T
    rotation_vector, _ = cv2.Rodrigues(rotation_camera_from_module)
    tag_centre_module = np.asarray(FIDUCIAL_TAG_CENTER_M)
    tag_position_camera = np.asarray((0.0, 0.0, 0.75))
    translation = tag_position_camera - rotation_camera_from_module @ tag_centre_module
    half = 0.5 * FIDUCIAL_TAG_SIZE_M
    object_points = np.asarray(((-half, half, 0.0), (half, half, 0.0), (half, -half, 0.0), (-half, -half, 0.0)))
    tag_rotation_vector, _ = cv2.Rodrigues(rotation_camera_from_tag)
    image_points, _ = cv2.projectPoints(
        object_points, tag_rotation_vector, tag_position_camera, intrinsic, np.zeros((4, 1))
    )
    image_points = image_points.reshape(4, 2).astype(np.float32)
    marker_pixels = 120
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(FIDUCIAL_DICTIONARY),
        FIDUCIAL_MARKER_ID,
        marker_pixels,
    )
    low = np.floor(image_points.min(axis=0) - 8).astype(int)
    high = np.ceil(image_points.max(axis=0) + 8).astype(int)
    cv2.rectangle(image, tuple(low), tuple(high), (255, 255, 255), thickness=-1)
    source = np.asarray(
        ((0, 0), (marker_pixels - 1, 0), (marker_pixels - 1, marker_pixels - 1), (0, marker_pixels - 1)),
        dtype=np.float32,
    )
    warp = cv2.warpPerspective(marker, cv2.getPerspectiveTransform(source, image_points), (256, 256))
    mask = cv2.warpPerspective(np.full_like(marker, 255), cv2.getPerspectiveTransform(source, image_points), (256, 256))
    image[mask > 0] = np.repeat(warp[..., None], 3, axis=-1)[mask > 0]
    depth = np.full((256, 256), tag_position_camera[2], dtype=np.float32)
    return image, depth, intrinsic, rotation_vector.reshape(3), translation


def test_calibrated_pnp_recovers_synthetic_rgb_pose() -> None:
    image, depth, intrinsic, rotation_vector, translation = _synthetic_frame()
    estimate = estimate_fiducial_pose(image, intrinsic, depth)
    expected_rotation, _ = cv2.Rodrigues(rotation_vector)
    assert np.linalg.norm(estimate.position_camera_m - translation) < 0.015
    assert np.linalg.norm(estimate.rotation_camera_from_object - expected_rotation) < 0.12
    assert estimate.reprojection_error_px < 1.0
    assert 0.0 <= estimate.confidence <= 1.0


def test_missing_marker_fails_closed() -> None:
    image, _, intrinsic, _, _ = _synthetic_frame()
    image[:] = 0
    with pytest.raises(RuntimeError, match="ArUco"):
        estimate_fiducial_pose(image, intrinsic)


def test_invalid_intrinsics_are_rejected() -> None:
    image, _, _, _, _ = _synthetic_frame()
    with pytest.raises(ValueError, match="intrinsic"):
        estimate_fiducial_pose(image, np.eye(2))
