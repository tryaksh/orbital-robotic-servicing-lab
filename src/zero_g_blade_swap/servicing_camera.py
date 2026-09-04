'''Simulator-free calibration for the fixed servicing RGB-D camera.

The physically flush datum lies on the module top face.  The first overhead
mount passed a static module-pose corpus but placed the camera above the rear
grapple approach, so the robot could stand in its line of sight during the
continuous workflow.  This mount shifts the same camera just beyond the
measured forward extent of that rear-mounted gripper and aims it at the centre
of the two-bay workflow envelope.  A subsequent one-change arm increases sensor
sampling after the 384-pixel detector failed late in the continuous approach.
Lens, aperture, tag geometry, and estimator gates remain unchanged.
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

# A second fixed view looks forward from the service aisle.  The primary view
# is beyond the rear-mounted gripper and sees capture cleanly; this view is on
# the other side of the rack mouth, so the destination's upper guide cannot
# cover the flush datum during guarded insertion.  The estimator retains the
# primary as its first choice and uses this calibrated view only when needed.
INSERT_CAMERA_POSITION_M = (0.15, -0.11, 2.15)
INSERT_CAMERA_TARGET_M = CAMERA_TARGET_M
INSERT_CAMERA_QUATERNION_WXYZ_ROS = (
    0.06111527065889844,
    -0.7044607325410619,
    0.7044607325410618,
    -0.06111527065889844,
)

# The 384-pixel arm resolved the marker at about six pixels per rendered cell
# and lost it repeatedly in a physically valid late-approach view.  At 640
# pixels the worst projected cell retains eight pixels: four interior samples
# after allowing a two-pixel rendered edge transition on each side.
CAMERA_HEIGHT_PX = 640
CAMERA_WIDTH_PX = 640
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
    'INSERT_CAMERA_POSITION_M',
    'INSERT_CAMERA_QUATERNION_WXYZ_ROS',
    'INSERT_CAMERA_TARGET_M',
]
