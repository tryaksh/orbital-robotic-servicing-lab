"""Measured gripper geometry and the grapple-pin dimensions derived from it.

This module holds numbers, not simulator objects, and imports nothing from Isaac
Lab. That is deliberate: the pin's dimensional contract is the part of this
design most likely to be broken by a later edit, and keeping it importable
without a simulator means the test suite can defend it on every commit rather
than only on a machine with Isaac Sim installed.

Everything under "Measured" comes from ``evidence/gripper_collision_envelope.json``,
produced by ``scripts/measure_gripper_envelope.py``, which reads the 2F-85's
collision meshes in the ``wrist_3_link`` frame. **Do not replace these with
values read off body origins.** Every 2F-85 body in this asset is collapsed to
within 18 mm of the flange, and reading them as pad locations produced a claim
this project had to retract.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Measured: the Robotiq 2F-85 as this asset actually models it.

#: Full travel of the actuated drive joint, from its own USD limits.
FINGER_JOINT_RANGE_RAD = (0.0, 0.8203)
#: Clear opening between the pad faces at ``finger_joint`` 0. Zero is fully
#: *open*; the opening falls monotonically as the command rises. The hardware's
#: published stroke is 85 mm, which this matches.
MAX_CLEAR_OPENING_M = 0.087077
#: Rate at which the clear opening closes with the drive command.
CLOSING_RATE_M_PER_RAD = 0.10623
#: Where the finger pads are, along the tool's approach axis, from the flange.
PAD_SPAN_FROM_FLANGE_M = (0.105, 0.162)
#: The palm. Nothing can sit closer to the flange than this on the tool axis.
PALM_FACE_FROM_FLANGE_M = 0.090
#: Half-width of a pad across the third axis.
PAD_HALF_WIDTH_M = 0.0135
#: Robotiq's published grip force range for this device.
RATED_GRIP_FORCE_N = (20.0, 235.0)


def clear_opening_m(finger_joint_rad: float) -> float:
    """Return the clear pad opening for a drive command, from the measured fit."""

    return MAX_CLEAR_OPENING_M - CLOSING_RATE_M_PER_RAD * finger_joint_rad


def drive_torque_for_grip_force_nm(grip_force_n: float) -> float:
    """Convert a pad normal force to the drive torque that produces it.

    Virtual work against the measured closing rate: the two pads close by
    ``CLOSING_RATE_M_PER_RAD`` per radian, so a pad force ``N`` needs a torque
    of ``N x CLOSING_RATE_M_PER_RAD``.
    """

    return grip_force_n * CLOSING_RATE_M_PER_RAD


# ---------------------------------------------------------------------------
# Scene geometry the pin has to live inside.

BLADE_LENGTH_M = 0.45
#: Leading edge of the rails. The gripper must stay outside this.
SLOT_MOUTH_X = 0.45
#: Blade centre at which the blade's rear face clears the mouth, so the module
#: is genuinely out of the rack. Decided with the owner on 2026-08-09.
EXTRACTED_BLADE_CENTRE_X = SLOT_MOUTH_X - 0.5 * BLADE_LENGTH_M
#: Top of the slot floor plate and bottom of the guided channel's upper lips.
#: The only pin section that ever passes the mouth has to fit between them.
SLOT_FLOOR_TOP_Z = 0.7025
SLOT_LIP_BOTTOM_Z = 0.7385

# ---------------------------------------------------------------------------
# The head-on grapple pin, in blade-local coordinates measured from the blade
# centre. Sections run from the free end toward the blade and must be
# contiguous; ``tests/test_grapple_geometry.py`` enforces that.

GRAPPLE_PIN_WEDGE_X = (-0.371, -0.311)
GRAPPLE_PIN_COLLAR_X = (-0.311, -0.305)
GRAPPLE_PIN_SHAFT_X = (-0.305, -0.225)
GRAPPLE_PIN_HALF_WIDTH_Y = 0.015
#: Free end then blade end. Thicker at the free end is what makes pulling wedge
#: the pin into the pads instead of pulling it out from between them.
GRAPPLE_PIN_WEDGE_HALF_HEIGHT = (0.035, 0.008)
#: Taller than the pads can ever open, so it is an absolute depth stop rather
#: than something a wide-open gripper slides past.
GRAPPLE_PIN_COLLAR_HALF_HEIGHT = 0.045
#: Thin enough to pass the slot between the floor plate and the upper lips.
GRAPPLE_PIN_SHAFT_HALF_HEIGHT = 0.015

#: Tool frame: the centre of the measured pad span, so the frame the policy
#: steers is the frame the pads grip with.
GRAPPLE_TOOL_OFFSET_POS = (0.0, 0.0, 0.5 * (PAD_SPAN_FROM_FLANGE_M[0] + PAD_SPAN_FROM_FLANGE_M[1]))
#: The matching point on the blade, so a zero tool-to-grip error means the pads
#: straddle the wedge with their leading faces on the collar.
GRAPPLE_PIN_GRIP_OFFSET = (
    GRAPPLE_PIN_COLLAR_X[0] - 0.5 * (PAD_SPAN_FROM_FLANGE_M[1] - PAD_SPAN_FROM_FLANGE_M[0]),
    0.0,
    0.0,
)
#: Head-on: the tool's +z approach axis along world +x, closing axis vertical.
GRAPPLE_HEAD_ON_TOOL_ROT = (0.0, 0.7071068, 0.0, 0.7071068)


def wedge_half_height_at(blade_local_x: float) -> float:
    """Return the wedge's half-height at a point along the pin."""

    distal_x, proximal_x = GRAPPLE_PIN_WEDGE_X
    distal_half, proximal_half = GRAPPLE_PIN_WEDGE_HALF_HEIGHT
    fraction = (blade_local_x - distal_x) / (proximal_x - distal_x)
    return distal_half + (proximal_half - distal_half) * min(max(fraction, 0.0), 1.0)


def wedge_taper_deg() -> float:
    """Return the wedge's half-angle from its axis, in degrees."""

    import math

    distal_x, proximal_x = GRAPPLE_PIN_WEDGE_X
    distal_half, proximal_half = GRAPPLE_PIN_WEDGE_HALF_HEIGHT
    return math.degrees(math.atan2(distal_half - proximal_half, proximal_x - distal_x))


def approach_clearance_per_side_m() -> float:
    """Room either side of the wedge's free end when the pads are fully open."""

    return 0.5 * (MAX_CLEAR_OPENING_M - 2.0 * GRAPPLE_PIN_WEDGE_HALF_HEIGHT[0])


__all__ = [
    "BLADE_LENGTH_M",
    "CLOSING_RATE_M_PER_RAD",
    "EXTRACTED_BLADE_CENTRE_X",
    "FINGER_JOINT_RANGE_RAD",
    "GRAPPLE_HEAD_ON_TOOL_ROT",
    "GRAPPLE_PIN_COLLAR_HALF_HEIGHT",
    "GRAPPLE_PIN_COLLAR_X",
    "GRAPPLE_PIN_GRIP_OFFSET",
    "GRAPPLE_PIN_HALF_WIDTH_Y",
    "GRAPPLE_PIN_SHAFT_HALF_HEIGHT",
    "GRAPPLE_PIN_SHAFT_X",
    "GRAPPLE_PIN_WEDGE_HALF_HEIGHT",
    "GRAPPLE_PIN_WEDGE_X",
    "GRAPPLE_TOOL_OFFSET_POS",
    "MAX_CLEAR_OPENING_M",
    "PAD_HALF_WIDTH_M",
    "PAD_SPAN_FROM_FLANGE_M",
    "PALM_FACE_FROM_FLANGE_M",
    "RATED_GRIP_FORCE_N",
    "SLOT_FLOOR_TOP_Z",
    "SLOT_LIP_BOTTOM_Z",
    "SLOT_MOUTH_X",
    "approach_clearance_per_side_m",
    "clear_opening_m",
    "drive_torque_for_grip_force_nm",
    "wedge_half_height_at",
    "wedge_taper_deg",
]
