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

#: Depth from the flange along the approach axis beyond which the only gripper
#: body present is an inner finger. Taken over the whole 0 to 0.8203 rad closure
#: range: no knuckle, outer finger, or palm bounding box reaches past this, while
#: the inner fingers reach 0.1621 m. A module feature narrower than the knuckles
#: is only safe inside this 37.6 mm band.
FINGERS_ONLY_DEPTH_FROM_FLANGE_M = 0.1245
#: Widest half-extent on the third axis reached by a body that is *not* an inner
#: finger, over every closure: the inner knuckle, at 17.5 mm against the fingers'
#: 13.5 mm. This is what an anti-yaw feature has to stay clear of.
NON_FINGER_HALF_WIDTH_M = 0.0175


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


# ---------------------------------------------------------------------------
# The anti-yaw yoke: the second-generation interface feature.
#
# A single-point tapered pin clamped by flat pads cannot resist rotation about
# the closing axis. The pads' contact normals lie along that axis, and a normal
# force cannot oppose a moment about its own direction, so only friction does and
# friction loses. Measured three ways on this workcell: extraction holds grip
# position at 12.2 mm for a whole pull and fails on grip attitude at 0.299 rad
# against a 0.20 rad limit; 93.0% of the insert skill's failures are outside that
# same tolerance at the step they end on; and the chained removal workflow ends
# inside it in 3.8% of episodes.
#
# The fix has to be lateral bearing surfaces, and the measured envelope says
# exactly where they fit. The fingers are 27 mm across and the pin is already
# 30 mm across, so the wedge's own side faces raised into walls need no new
# width, and confining them to the fingers-only band keeps every other part of
# the gripper out of them.
#
# The walls carry a lead-in flare at their mouth for the same reason the rack
# does, and this project has measured how much that matters: delete the rack's
# flares and two fully trained insertion policies both score 0%. A 1.5 mm slot
# the capture had to hit blind would trade a yaw problem for a capture problem.

#: Along the pin, mouth first then collar face.
GRAPPLE_YOKE_X = (-0.345, -0.311)
#: The parallel section, which is what actually constrains yaw.
GRAPPLE_YOKE_PARALLEL_X = (-0.335, -0.311)
#: Inner faces, flush with the pin's existing side faces, so the pin gains no
#: width. 1.5 mm of clearance per side against a 13.5 mm finger half-width.
GRAPPLE_YOKE_HALF_GAP_M = 0.015
#: Half-gap at the flared mouth.
GRAPPLE_YOKE_MOUTH_HALF_GAP_M = 0.01864
GRAPPLE_YOKE_WALL_THICKNESS_M = 0.003
#: The collar's, so the yoke reads as a channel extending from the depth stop
#: rather than as a separate part, and never exceeds its envelope.
GRAPPLE_YOKE_HALF_HEIGHT = 0.045


def yoke_lead_in_catch_m() -> float:
    """Lateral misalignment the yoke's mouth accepts, per side."""

    return GRAPPLE_YOKE_MOUTH_HALF_GAP_M - PAD_HALF_WIDTH_M


def yoke_flare_deg() -> float:
    """Half-angle of the yoke's lead-in from its axis, in degrees."""

    import math

    mouth_x, _ = GRAPPLE_YOKE_X
    parallel_x, _ = GRAPPLE_YOKE_PARALLEL_X
    return math.degrees(
        math.atan2(GRAPPLE_YOKE_MOUTH_HALF_GAP_M - GRAPPLE_YOKE_HALF_GAP_M, parallel_x - mouth_x)
    )


def yoke_mouth_depth_from_flange_m() -> float:
    """How deep the yoke's mouth sits when the pads are seated on the collar."""

    mouth_x, collar_x = GRAPPLE_YOKE_X
    return PAD_SPAN_FROM_FLANGE_M[1] - (collar_x - mouth_x)


def yoke_free_yaw_rad() -> float:
    """Rotation about the closing axis available before the walls take load.

    A shaft of half-width ``a`` in a slot of half-width ``a + c`` engaged over a
    length ``L`` rotates about ``2c / L`` before a corner reaches a wall. This is
    a geometric prediction, not a measurement: what the walls do under load is
    what ``scripts/grasp_diagnostics.py --load_axis yaw`` reports.
    """

    clearance = GRAPPLE_YOKE_HALF_GAP_M - PAD_HALF_WIDTH_M
    length = GRAPPLE_YOKE_PARALLEL_X[1] - GRAPPLE_YOKE_PARALLEL_X[0]
    return 2.0 * clearance / length


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
    "FINGERS_ONLY_DEPTH_FROM_FLANGE_M",
    "FINGER_JOINT_RANGE_RAD",
    "GRAPPLE_HEAD_ON_TOOL_ROT",
    "GRAPPLE_YOKE_HALF_GAP_M",
    "GRAPPLE_YOKE_HALF_HEIGHT",
    "GRAPPLE_YOKE_MOUTH_HALF_GAP_M",
    "GRAPPLE_YOKE_PARALLEL_X",
    "GRAPPLE_YOKE_WALL_THICKNESS_M",
    "GRAPPLE_YOKE_X",
    "NON_FINGER_HALF_WIDTH_M",
    "yoke_flare_deg",
    "yoke_free_yaw_rad",
    "yoke_lead_in_catch_m",
    "yoke_mouth_depth_from_flange_m",
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
