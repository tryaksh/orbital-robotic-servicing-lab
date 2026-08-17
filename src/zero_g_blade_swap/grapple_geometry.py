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
# Moving a module sideways from one bay to the next.
#
# Extraction stops the instant the module's rear face clears the mouth, which is
# the right definition of "removed" and the wrong place to turn. The lead-in
# flares stand *proud* of the mouth: each is an 80 mm plate rotated 12 degrees
# about z, centred at x = 0.41275, so it reaches to about x = 0.372. A module
# whose front face is still past that will drag its nose across the neighbouring
# bay's flare on the way across.
#
# So the module has to retreat past the flares before it can travel sideways,
# and the distance is derived rather than chosen.
FLARE_CENTER_X = 0.41275
FLARE_PLATE_LENGTH_M = 0.080
FLARE_PLATE_THICKNESS_M = 0.018
#: Half-extent of a flare along x, including the 12-degree rotation.
FLARE_HALF_EXTENT_X = 0.5 * (
    FLARE_PLATE_LENGTH_M * 0.9781476  # cos(12 deg)
    + FLARE_PLATE_THICKNESS_M * 0.2079117  # sin(12 deg)
)
#: Leading face of the flares: the plane a module must be fully behind before it
#: can be moved to another bay.
FLARE_LEADING_X = FLARE_CENTER_X - FLARE_HALF_EXTENT_X
#: Module centre at which its front face sits on that plane.
TRANSIT_CLEAR_BLADE_CENTRE_X = FLARE_LEADING_X - 0.5 * BLADE_LENGTH_M
#: How much further back than the extraction target the module has to come
#: before a lateral move is free of the rack. Positive, and about 80 mm.
TRANSIT_RETREAT_M = EXTRACTED_BLADE_CENTRE_X - TRANSIT_CLEAR_BLADE_CENTRE_X

# ---------------------------------------------------------------------------
# The head-on grapple pin, in blade-local coordinates measured from the blade
# centre. Sections run from the free end toward the blade and must be
# contiguous; ``tests/test_grapple_geometry.py`` enforces that.

# ---------------------------------------------------------------------------
# 2026-08-17: the tapered wedge is replaced by a KEYED FLAT SECTION between two
# axial stops. This is a design change, not a tuning change, and it is the one
# this project should have made three sessions ago.
#
# **Why the wedge was wrong.** Flat pads closing on a smooth taper hold a module
# by friction alone, and the geometry gives friction almost nothing to work with:
# the clamping normals lie along the closing axis, and a normal force cannot
# oppose a moment about its own direction. Measured four independent ways here,
# and it culminated in a module swinging *end-for-end* about the grip during the
# relocation transit -- the tool-to-module offset changed sign, -0.335 m to
# +0.305 m, while grip error read a mild 24 mm because the pads were still on the
# pin. Two features were bolted on to fix it, an anti-yaw yoke and a modelled
# latch, and both were net negatives; you cannot patch keying onto a grip that
# has none.
#
# **Why this shape.** It is what flight hardware does. ISS ORUs are handled by
# Dextre's OTCM gripping a *micro-square* -- a square boss, so rotation is
# blocked by form -- and then bolted. SSRMS grapple fixtures use snares plus
# three alignment ramps. SIROM uses three latches at 120 degrees; HOTDOCK uses
# external form-fit geometry. Not one of them makes friction load-bearing.
#
# **What each section now does**, free end first:
#
#   nose flange  20 mm, 60 mm tall   front axial stop: the pads cannot pull off
#   key          80 mm, 30 mm tall   FLAT faces: plane contact blocks every
#                                    rotation by form rather than by friction
#   collar        6 mm, 90 mm tall   rear axial stop and absolute depth stop
#   shaft        80 mm, 30 mm tall   the only section that passes the slot mouth
#
# The pads are 57 mm long and seat against the collar, so they lie wholly on the
# key with 23 mm of key left toward the nose. That 23 mm is the axial travel a
# pull gets before the nose flange stops it, which is the positive capture the
# taper used to approximate by wedging -- and because there is no taper left,
# closing no longer thrusts the payload along the pin, which is the separate
# defect that made raising grip force *reduce* holding capacity.
#
# The grip offset, the slot, and every calibrated arm pose are deliberately
# unchanged, so the existing checkpoints remain a valid starting point.
#: Axial play between the seated pads and the nose flange, and the single number
#: this interface's axial capacity now rests on.
#:
#: **Measured, not chosen.** The first keyed pin put the flange 23 mm from the
#: seated pads, and the axial gate FAILED at 51.6 N against the 66.36 N
#: requirement -- worse than the taper's 69 N. That was the taper's one real
#: virtue showing up as a loss: it was self-energising, so pulling dragged thicker
#: material into the pads and raised the normal force, and a flat key has no such
#: effect. Friction alone carries about 52 N here and then the module simply slides
#: until something stops it.
#:
#: The answer is the one flight hardware uses: do not carry axial load on friction
#: at all, carry it on a stop that engages inside the tolerance. The collar already
#: datums the insert direction; this clearance is what bounds the pull direction.
#: It has to exceed how accurately a capture places the pads axially -- measured at
#: 0.15 mm on the settled grasp -- and stay under the 2 mm the pull gate calls a
#: slipped grip.
KEY_SEAT_CLEARANCE_M = 0.0015
#: Derived so the pads, seated against the collar, sit on the flat with exactly
#: ``KEY_SEAT_CLEARANCE_M`` of travel before the nose flange bears.
GRAPPLE_PIN_KEY_X = (
    -0.311 - (PAD_SPAN_FROM_FLANGE_M[1] - PAD_SPAN_FROM_FLANGE_M[0]) - KEY_SEAT_CLEARANCE_M,
    -0.311,
)
#: 20 mm of flange, immediately ahead of the key.
GRAPPLE_PIN_NOSE_X = (GRAPPLE_PIN_KEY_X[0] - 0.020, GRAPPLE_PIN_KEY_X[0])
GRAPPLE_PIN_COLLAR_X = (-0.311, -0.305)
GRAPPLE_PIN_SHAFT_X = (-0.305, -0.225)
GRAPPLE_PIN_HALF_WIDTH_Y = 0.015
#: Constant, because the whole point is that the pads meet a flat. A plane
#: contact resists moments about every axis in the plane; a line contact on a
#: taper resists none of them.
GRAPPLE_PIN_KEY_HALF_HEIGHT = 0.015
#: Front axial stop. Shorter than the pads can open, so an approaching gripper
#: still passes over it, and taller than the key so a closed pad cannot ride off
#: the end.
GRAPPLE_PIN_NOSE_HALF_HEIGHT = 0.030
#: Taller than the pads can ever open, so it is an absolute depth stop rather
#: than something a wide-open gripper slides past.
GRAPPLE_PIN_COLLAR_HALF_HEIGHT = 0.045
#: Thin enough to pass the slot between the floor plate and the upper lips.
GRAPPLE_PIN_SHAFT_HALF_HEIGHT = 0.015

#: Tool frame: the centre of the measured pad span, so the frame the policy
#: steers is the frame the pads grip with.
GRAPPLE_TOOL_OFFSET_POS = (0.0, 0.0, 0.5 * (PAD_SPAN_FROM_FLANGE_M[0] + PAD_SPAN_FROM_FLANGE_M[1]))
#: The matching point on the blade, so a zero tool-to-grip error means the pads
#: straddle the key with their leading faces on the collar.
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


def key_seat_axial_travel_m() -> float:
    """Axial travel a pull gets before the nose flange stops the pads.

    The positive axial capture the taper used to approximate by wedging, and now
    a hard number instead of a friction coefficient: the pads seat against the
    collar, so what is left is the key's length less the pad span.
    """

    key_length = GRAPPLE_PIN_KEY_X[1] - GRAPPLE_PIN_KEY_X[0]
    pad_span = PAD_SPAN_FROM_FLANGE_M[1] - PAD_SPAN_FROM_FLANGE_M[0]
    return key_length - pad_span


def approach_clearance_per_side_m() -> float:
    """Room either side of the nose flange when the pads are fully open.

    The nose flange is the widest thing an approaching gripper has to pass over,
    so it is what sets the approach clearance.
    """

    return 0.5 * (MAX_CLEAR_OPENING_M - 2.0 * GRAPPLE_PIN_NOSE_HALF_HEIGHT)


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
    "KEY_SEAT_CLEARANCE_M",
    "GRAPPLE_PIN_KEY_HALF_HEIGHT",
    "GRAPPLE_PIN_KEY_X",
    "GRAPPLE_PIN_NOSE_HALF_HEIGHT",
    "GRAPPLE_PIN_NOSE_X",
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
    "key_seat_axial_travel_m",
]
