'''Project the flush datum through the shipped camera over the workflow envelope.

CPU-only.  The check uses the same calibration constants as the scene and
certifier, projects both marker and quiet-zone corners at the collector's pose
extremes, and verifies from source that those three consumers remain bound to
the checked definitions.
'''

from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
from pathlib import Path

import numpy as np

from zero_g_blade_swap.grapple_geometry import TRANSIT_CLEAR_BLADE_CENTRE_X
from zero_g_blade_swap.servicing_camera import (
    CAMERA_CLIPPING_RANGE_M,
    CAMERA_FOCAL_LENGTH_MM,
    CAMERA_HEIGHT_PX,
    CAMERA_HORIZONTAL_APERTURE_MM,
    CAMERA_POSITION_M,
    CAMERA_QUATERNION_WXYZ_ROS,
    CAMERA_WIDTH_PX,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'src/zero_g_blade_swap/tasks/blade_swap/scene_cfg.py'
COLLECTOR = ROOT / 'scripts/collect_grapple_vision.py'
CERTIFIER = ROOT / 'scripts/certify_fiducial_perception.py'
FIDUCIAL = ROOT / 'src/zero_g_blade_swap/fiducial.py'


def _fiducial_literal(name: str) -> object:
    for node in ast.parse(FIDUCIAL.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _quaternion_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def check() -> dict[str, object]:
    fiducial_tag_size_m = float(_fiducial_literal('FIDUCIAL_TAG_SIZE_M'))
    fiducial_quiet_zone_size_m = float(
        _fiducial_literal('FIDUCIAL_QUIET_ZONE_SIZE_M')
    )
    fiducial_tag_center_m = tuple(
        float(value) for value in _fiducial_literal('FIDUCIAL_TAG_CENTER_M')
    )
    camera_position = np.asarray(CAMERA_POSITION_M, dtype=np.float64)
    rotation_world_from_camera = _quaternion_matrix(CAMERA_QUATERNION_WXYZ_ROS)
    focal_px = CAMERA_WIDTH_PX * CAMERA_FOCAL_LENGTH_MM / CAMERA_HORIZONTAL_APERTURE_MM
    principal = np.asarray(((CAMERA_WIDTH_PX - 1) / 2, (CAMERA_HEIGHT_PX - 1) / 2))

    x_bounds = (min(0.55, TRANSIT_CLEAR_BLADE_CENTRE_X - 0.08), 0.73)
    y_bounds = (-0.22 - 0.025, 0.025)
    z_bounds = (0.72 - 0.050, 0.72 + 0.050)
    rpy_bounds = ((-0.25, 0.25), (-0.25, 0.25), (-0.40, 0.40))
    quiet_half = 0.5 * fiducial_quiet_zone_size_m
    marker_half = 0.5 * fiducial_tag_size_m
    tag_z = fiducial_tag_center_m[2]

    frame_margins_px: list[float] = []
    depths_m: list[float] = []
    incidence_rad: list[float] = []
    marker_edges_px: list[float] = []
    for centre, roll, pitch, yaw in itertools.product(
        itertools.product(x_bounds, y_bounds, z_bounds), *rpy_bounds
    ):
        module_rotation = _rpy_matrix(roll, pitch, yaw)
        centre = np.asarray(centre, dtype=np.float64)

        def project(half_extent: float) -> np.ndarray:
            world = np.asarray(
                [
                    centre + module_rotation @ np.asarray((x, y, tag_z))
                    for x, y in (
                        (-half_extent, half_extent),
                        (half_extent, half_extent),
                        (half_extent, -half_extent),
                        (-half_extent, -half_extent),
                    )
                ]
            )
            camera = (rotation_world_from_camera.T @ (world - camera_position).T).T
            depths_m.extend(camera[:, 2].tolist())
            return focal_px * camera[:, :2] / camera[:, 2:3] + principal

        quiet_pixels = project(quiet_half)
        frame_margins_px.extend(
            (
                float(quiet_pixels[:, 0].min()),
                float(CAMERA_WIDTH_PX - 1 - quiet_pixels[:, 0].max()),
                float(quiet_pixels[:, 1].min()),
                float(CAMERA_HEIGHT_PX - 1 - quiet_pixels[:, 1].max()),
            )
        )
        marker_pixels = project(marker_half)
        marker_edges_px.extend(
            float(np.linalg.norm(marker_pixels[(index + 1) % 4] - marker_pixels[index]))
            for index in range(4)
        )
        tag_centre = centre + module_rotation @ np.asarray(fiducial_tag_center_m)
        view_to_camera = camera_position - tag_centre
        cosine = float(
            np.dot(module_rotation[:, 2], view_to_camera) / np.linalg.norm(view_to_camera)
        )
        incidence_rad.append(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    scene = SCENE.read_text(encoding='utf-8')
    collector = COLLECTOR.read_text(encoding='utf-8')
    certifier = CERTIFIER.read_text(encoding='utf-8')
    bindings = {
        'scene_uses_shared_position': 'pos=CAMERA_POSITION_M' in scene,
        'scene_uses_shared_orientation': 'rot=CAMERA_QUATERNION_WXYZ_ROS' in scene,
        'certifier_uses_shared_calibration': 'from zero_g_blade_swap.servicing_camera import (' in certifier,
        'collector_retains_workflow_envelope': (
            '0.55 + 0.18 * torch.rand' in collector
            and 'TRANSIT_CLEAR_BLADE_CENTRE_X - 0.08' in collector
            and 'roll[transfer] = 0.25' in collector
            and 'yaw[transfer] = 0.40' in collector
        ),
    }
    minimum_margin_px = min(frame_margins_px)
    minimum_depth_m = min(depths_m)
    maximum_depth_m = max(depths_m)
    minimum_marker_edge_px = min(marker_edges_px)
    maximum_incidence_rad = max(incidence_rad)
    passed = (
        minimum_margin_px > 0.0
        and minimum_depth_m > CAMERA_CLIPPING_RANGE_M[0]
        and maximum_depth_m < CAMERA_CLIPPING_RANGE_M[1]
        and maximum_incidence_rad < 0.5 * math.pi
        and all(bindings.values())
    )
    return {
        'status': 'passed' if passed else 'failed',
        'title': 'Fixed servicing-camera coverage of the flush fiducial workflow envelope',
        'evidence_type': 'geometric_derivation_no_simulator',
        'single_physical_change': 'camera placement and aim',
        'camera_position_m': list(CAMERA_POSITION_M),
        'camera_quaternion_wxyz_ros': list(CAMERA_QUATERNION_WXYZ_ROS),
        'flush_datum': {
            'tag_size_m': fiducial_tag_size_m,
            'quiet_zone_size_m': fiducial_quiet_zone_size_m,
            'centre_in_module_m': list(fiducial_tag_center_m),
        },
        'workflow_envelope': {
            'centre_x_m': list(x_bounds),
            'centre_y_m': list(y_bounds),
            'centre_z_m': list(z_bounds),
            'roll_rad': list(rpy_bounds[0]),
            'pitch_rad': list(rpy_bounds[1]),
            'yaw_rad': list(rpy_bounds[2]),
        },
        'projection': {
            'minimum_quiet_zone_frame_margin_px': minimum_margin_px,
            'minimum_marker_edge_px': minimum_marker_edge_px,
            'minimum_marker_cell_px': minimum_marker_edge_px / 6.0,
            'minimum_depth_m': minimum_depth_m,
            'maximum_depth_m': maximum_depth_m,
            'maximum_incidence_rad': maximum_incidence_rad,
            'maximum_incidence_deg': math.degrees(maximum_incidence_rad),
        },
        'source_bindings': bindings,
        'scope_and_limitations': [
            'Projection proves coverage and front-face incidence, not rendered detection.',
            'Robot and rack occlusion are exercised by the held-out rendered corpus and strict RGB-D chain.',
            'Lens, aperture, resolution, flush datum, and estimator gates are unchanged.',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = check()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
