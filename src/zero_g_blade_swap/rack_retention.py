'''Simulator-free geometry and ratings for destination rack retention.

Two rack-owned pawls close behind a module only after the unchanged seating
predicate passes.  Their visible geometry discloses the idealized break-rated
Rack-to-module joint that carries load in simulation.
'''

from __future__ import annotations

from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    BLADE_THICKNESS_M,
    BLADE_WIDTH_M,
    GRAPPLE_PIN_HALF_WIDTH_Y,
    SLOT_ENTRY_RAMP_CATCH_M,
)
from zero_g_blade_swap.service_latch import (
    LIP_THICKNESS_M,
    RATED_FORCE_N as ROBOT_LATCH_RATED_FORCE_N,
    RATED_TORQUE_NM as ROBOT_LATCH_RATED_TORQUE_NM,
    REQUIRED_AXIAL_CAPACITY_N,
    WEB_INNER_HALF_GAP_M,
)

PAWL_FACE_CLEARANCE_M = WEB_INNER_HALF_GAP_M - GRAPPLE_PIN_HALF_WIDTH_Y
PAWL_AXIAL_THICKNESS_M = LIP_THICKNESS_M
PAWL_OVERLAP_M = LIP_THICKNESS_M
PAWL_LATERAL_THICKNESS_M = BLADE_THICKNESS_M
PAWL_HEIGHT_M = BLADE_THICKNESS_M

PAWL_CLOSED_INNER_HALF_GAP_M = 0.5 * BLADE_WIDTH_M - PAWL_OVERLAP_M
PAWL_OPEN_INNER_HALF_GAP_M = 0.5 * BLADE_WIDTH_M + SLOT_ENTRY_RAMP_CATCH_M
PAWL_CLOSE_STROKE_M = PAWL_OPEN_INNER_HALF_GAP_M - PAWL_CLOSED_INNER_HALF_GAP_M
PAWL_CLOSED_OUTER_HALF_GAP_M = PAWL_CLOSED_INNER_HALF_GAP_M + PAWL_LATERAL_THICKNESS_M

RATED_FORCE_N = ROBOT_LATCH_RATED_FORCE_N
RATED_TORQUE_NM = ROBOT_LATCH_RATED_TORQUE_NM


def pawl_tip_boxes(
    seated_module_position: tuple[float, float, float],
) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    '''Return the two closed pawl-tip boxes in the environment frame.'''

    centre_x, centre_y, centre_z = seated_module_position
    rear_face_x = centre_x - 0.5 * BLADE_LENGTH_M
    pawl_x = rear_face_x - PAWL_FACE_CLEARANCE_M - 0.5 * PAWL_AXIAL_THICKNESS_M
    half_width = 0.5 * (
        PAWL_CLOSED_INNER_HALF_GAP_M + PAWL_CLOSED_OUTER_HALF_GAP_M
    )
    size = (PAWL_AXIAL_THICKNESS_M, PAWL_LATERAL_THICKNESS_M, PAWL_HEIGHT_M)
    return (
        ('PawlLeft', (pawl_x, centre_y + half_width, centre_z), size),
        ('PawlRight', (pawl_x, centre_y - half_width, centre_z), size),
    )


def pawl_translation(*, engaged: bool, sign: float) -> tuple[float, float, float]:
    '''Return the carriage stroke from the authored closed pose.'''

    return (0.0, 0.0 if engaged else sign * PAWL_CLOSE_STROKE_M, 0.0)
