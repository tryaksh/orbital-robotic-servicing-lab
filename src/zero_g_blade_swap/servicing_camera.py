'''Simulator-free calibration for the fixed servicing RGB-D camera.

The physically flush datum lies on the module top face.  The former oblique
overview saw that plane at roughly 68 degrees from its normal and detected only
43.27% of critical-bay frames.  This mount changes one physical variable: the
same camera is placed directly above the two-bay workflow envelope and aimed
straight down.  Resolution, lens, aperture, tag geometry, and estimator gates
remain unchanged.
'''

from __future__ import annotations

CAMERA_POSITION_M = (0.40, -0.11, 2.15)
CAMERA_TARGET_M = (0.40, -0.11, 0.72)

# ROS optical frame: +z forward (world -z), +x image-right (world -y), and +y
# image-down (world -x). This is a 180 degree rotation around (1, -1, 0)/sqrt(2).
CAMERA_QUATERNION_WXYZ_ROS = (
    0.0,
    0.7071067811865476,
    -0.7071067811865476,
    0.0,
)

CAMERA_HEIGHT_PX = 384
CAMERA_WIDTH_PX = 384
CAMERA_FOCAL_LENGTH_MM = 45.0
CAMERA_HORIZONTAL_APERTURE_MM = 30.0
CAMERA_FOCUS_DISTANCE_M = 1.4
CAMERA_CLIPPING_RANGE_M = (0.05, 4.0)
CAMERA_UPDATE_PERIOD_S = 1.0 / 15.0


__all__ = [
    'CAMERA_CLIPPING_RANGE_M',
    'CAMERA_FOCAL_LENGTH_MM',
    'CAMERA_FOCUS_DISTANCE_M',
    'CAMERA_HEIGHT_PX',
    'CAMERA_HORIZONTAL_APERTURE_MM',
    'CAMERA_POSITION_M',
    'CAMERA_QUATERNION_WXYZ_ROS',
    'CAMERA_TARGET_M',
    'CAMERA_UPDATE_PERIOD_S',
    'CAMERA_WIDTH_PX',
]
