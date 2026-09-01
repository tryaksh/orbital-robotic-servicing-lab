'''Simulator-free calibration for the fixed servicing RGB-D camera.

The physically flush datum lies on the module top face.  The first overhead
mount passed a static module-pose corpus but placed the camera above the rear
grapple approach, so the robot could stand in its line of sight during the
continuous workflow.  This mount shifts the same camera just beyond the
measured forward extent of that rear-mounted gripper and aims it at the centre
of the two-bay workflow envelope.  Resolution, lens, aperture, tag geometry,
and estimator gates remain unchanged.
'''

from __future__ import annotations

CAMERA_POSITION_M = (0.52, -0.11, 2.15)
CAMERA_TARGET_M = (0.40, -0.11, 0.72)

# ROS optical frame: +z points from CAMERA_POSITION_M to CAMERA_TARGET_M, +x
# image-right points along world -y, and +y image-down completes the basis.  The
# quaternion is the closed-form world-y tilt of the former straight-down mount.
CAMERA_QUATERNION_WXYZ_ROS = (
    0.029590823511594526,
    0.7064873552753126,
    -0.7064873552753126,
    -0.029590823511594526,
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
