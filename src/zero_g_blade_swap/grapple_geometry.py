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

#: **Unchanged at 450 mm, and 250 mm was tried and reverted with the number that
#: reverted it.**
#:
#: The clearance a rigid module needs is *L* *theta*/2 and its leading corner
#: dips by the same, so shortening it is the direct fix for a chain that jams on
#: its lead-in. It is also the one change that moves every axial target at once:
#: ``EXTRACTED_BLADE_CENTRE_X`` and ``TRANSIT_CLEAR_BLADE_CENTRE_X`` are both
#: derived from it, so a 200 mm cut moves the extraction target 100 mm.
#:
#: Measured at 250 mm on the same seed and checkpoints: capture and seating
#: still work -- the pin is unchanged, so it is the same capture -- and
#: extraction does not. It ran 750 control steps against 233, ended past its
#: moved target, and left the module 544 mm off line and 1.21 rad round. The
#: extract policy is trained to pull to a place, and the place moved.
#:
#: So the clearance is taken out of the cross-section instead, where it costs no
#: axial target and no policy: see ``BLADE_SIZE``.
BLADE_LENGTH_M = 0.45
#: The module cross-section belongs beside its length. Keeping these three
#: dimensions in the simulator-free geometry module lets rack interfaces prove
#: their clearances without importing Isaac Lab, while ``assets.py`` remains
#: the only place that turns the dimensions into collision geometry.
BLADE_WIDTH_M = 0.130
BLADE_THICKNESS_M = 0.020
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
#: The destination bay's vertical lead-in, which exists because a relocation
#: enters the rack mouth from outside and no skill in this project ever had to.
#: Same plate and same angle as the lateral flare, so the two cannot drift
#: apart; the width is narrower than the module because the latch carriage runs
#: beside it. See ``docs/service_interface_spec.md`` section 6.1.
SLOT_ENTRY_RAMP_LENGTH_M = 0.080
SLOT_ENTRY_RAMP_WIDTH_M = 0.060
SLOT_ENTRY_RAMP_THICKNESS_M = 0.018
SLOT_ENTRY_RAMP_DEG = 12.0
#: Vertical catch it provides, per side, from its own geometry.
SLOT_ENTRY_RAMP_CATCH_M = SLOT_ENTRY_RAMP_LENGTH_M * 0.2079117

#: How much further back than the extraction target the module has to come
#: before a lateral move is free of the rack. Positive, and about 80 mm.
TRANSIT_RETREAT_M = EXTRACTED_BLADE_CENTRE_X - TRANSIT_CLEAR_BLADE_CENTRE_X

# ---------------------------------------------------------------------------
# The head-on grapple pin, in blade-local coordinates measured from the blade
# centre. Sections run from the free end toward the blade and must be
# contiguous; ``tests/test_grapple_geometry.py`` enforces that.

GRAPPLE_PIN_WEDGE_X = (-0.371, -0.311)
GRAPPLE_PIN_COLLAR_X = (-0.311, -0.305)
#: Runs to the blade's rear face, which moved with ``BLADE_LENGTH_M``. The
#: gripped sections -- wedge and collar -- are untouched, in blade-local and in
#: world coordinates, which is what keeps capture the same problem it was.
GRAPPLE_PIN_SHAFT_X = (-0.305, -0.5 * BLADE_LENGTH_M)
GRAPPLE_PIN_HALF_WIDTH_Y = 0.015
#: Free end then blade end. Thicker at the free end is what makes pulling wedge
#: the pin into the pads instead of pulling it out from between them.
GRAPPLE_PIN_WEDGE_HALF_HEIGHT = (0.035, 0.008)
#: Taller than the pads can ever open, so it is an absolute depth stop rather
#: than something a wide-open gripper slides past.
GRAPPLE_PIN_COLLAR_HALF_HEIGHT = 0.045
#: Thin enough to pass the slot between the floor plate and the upper lips --
#: and, since the module was thinned to 20 mm, thin enough to pass it *at the
#: attitude the arm delivers*.
#:
#: At 30 mm tall the shaft had 3 mm per side in the 36 mm channel while the
#: blade in front of it had 8. That inverts which part is binding: the last
#: 74 mm of the seating stroke is the shaft entering the mouth, and at the
#: 24.8 mrad the module was holding, its far end swings 3.6 mm. Measured, the
#: chain stopped exactly there -- module at 0.6759 against a seated 0.75, with
#: the blade itself well inside its own clearance.
#:
#: Twenty millimetres is also what a boss on a 20 mm blade is.
GRAPPLE_PIN_SHAFT_HALF_HEIGHT = 0.010

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
# What "the grip is still there" means, resolved onto the pin's own axes.
#
# ``GRAPPLE_PIN_GRIP_OFFSET`` above is a *drawing* dimension: where the pads
# would sit on an unloaded pin with their leading faces on the collar. It is not
# where they sit once the pull takes load, and the difference is not small.
#
# A tapered pin holds by feeding thicker material between the pads as the module
# lags the tool, which is the whole reason this interface exists -- a parallel
# jaw on a passive feature holds about 6 N along the pull axis against the
# 66.4 N the job needs, and the wedge holds 77 N from geometry alone. Feeding is
# the load path working. Measured over 433 successful extractions of the current
# module, the pads come to rest **12.0 mm** along the pin from the drawing pose,
# in a band 11.2 to 12.0 mm wide: a hard equilibrium, not a spread.
#
# Both grip criteria in this project were isotropic balls about the drawing
# pose: 20 mm to count as captured, 30 mm to count as dropped. So 12 of the
# 20 mm and 12 of the 30 mm were spent by the interface doing its job before the
# policy acted, and 79% of what consumed the rest was more of the same feed.
# Measured on extract v17m130 at stage 0, 50 of 79 failures ended on that ball
# with the module 14.7 mm into a 525 mm pull -- at the first load transfer, with
# the pin seating, which is the one thing that must not read as a dropped
# module.
#
# The bounds below replace the ball with the three questions the pin actually
# asks, and each is a dimension of the pin rather than a number chosen to pass:
GRIP_SEATED_APPROACH_OFFSET_M = -0.0120
#: Length of pin the pads close on.
GRAPPLE_PIN_WEDGE_LENGTH_M = GRAPPLE_PIN_WEDGE_X[1] - GRAPPLE_PIN_WEDGE_X[0]
#: Pad length along the approach axis, from the measured span.
PAD_LENGTH_M = PAD_SPAN_FROM_FLANGE_M[1] - PAD_SPAN_FROM_FLANGE_M[0]
#: **Feeding in.** The pads may ride along the taper while at least half the
#: wedge stays under them. Past that the pads are running out of pin, and the
#: 2 N sin(alpha) the capacity argument rests on is being taken over a contact
#: that is disappearing. Half the wedge, from the wedge's own length.
GRIP_MAX_APPROACH_FEED_M = GRIP_SEATED_APPROACH_OFFSET_M - 0.5 * GRAPPLE_PIN_WEDGE_LENGTH_M
#: **Backing out.** The collar is a hard stop -- it is taller than the pads can
#: open, which is what makes it a depth stop rather than something a wide-open
#: gripper slides past -- and it sits exactly at the drawing pose, which is how
#: that pose was defined. So a module backing out past zero is not a tolerance
#: question: the collar is in the way, and anything past it means the pads have
#: left the pin. One PhysX contact envelope of slack, and no more.
#:
#: This end of the band is not slack for extraction, which never reaches it. It
#: is where the *insertion* sits: pushing the module home makes it lag the other
#: way and bear on the collar, so the same predicate has to admit both. That is
#: why the band is stated on the absolute offset rather than as a margin either
#: side of the seated pull equilibrium.
GRIP_MAX_APPROACH_BACKOUT_M = 0.005
#: **Across the pin.** The pads are 27 mm wide and the pin is 30 mm, so a pad
#: bears over its whole face until the offset passes the 1.5 mm the pin is
#: wider, and then loses one millimetre of face per millimetre of offset:
#: ``overlap(d) = pad_half + pin_half - d``. Half the face is gone when that
#: reaches half a pad width, which is at ``d = pin_half`` exactly -- the pad
#: widths cancel. So the bound is the pin's own half-width, and it is *tighter*
#: than the 30 mm ball it replaces by a factor of two, which is the point:
#: 30 mm was never a transverse budget, it was an axial one being spent
#: sideways. Measured on extract v17m130 at stage 0, 41 of 83 failures under the
#: ball were already past 13.5 mm across the pin, and its worst was 31.6 mm.
GRIP_MAX_TRANSVERSE_M = GRAPPLE_PIN_HALF_WIDTH_Y


def grip_offset_admissible(closing_m: float, third_m: float, approach_m: float) -> bool:
    """Whether a tool-to-grip offset, in tool axes, still describes a held pin.

    Signed ``approach_m``, because the two directions along the pin are not the
    same question: one is the taper taking load and the other is the pin leaving
    the pads.
    """

    transverse = (closing_m**2 + third_m**2) ** 0.5
    return bool(
        transverse <= GRIP_MAX_TRANSVERSE_M
        and GRIP_MAX_APPROACH_FEED_M <= approach_m <= GRIP_MAX_APPROACH_BACKOUT_M
    )


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
    "NON_FINGER_HALF_WIDTH_M",
    "GRAPPLE_PIN_COLLAR_HALF_HEIGHT",
    "GRAPPLE_PIN_COLLAR_X",
    "GRAPPLE_PIN_GRIP_OFFSET",
    "GRAPPLE_PIN_HALF_WIDTH_Y",
    "GRAPPLE_PIN_SHAFT_HALF_HEIGHT",
    "GRAPPLE_PIN_SHAFT_X",
    "GRAPPLE_PIN_WEDGE_HALF_HEIGHT",
    "GRAPPLE_PIN_WEDGE_LENGTH_M",
    "GRAPPLE_PIN_WEDGE_X",
    "GRIP_MAX_APPROACH_BACKOUT_M",
    "GRIP_MAX_APPROACH_FEED_M",
    "GRIP_MAX_TRANSVERSE_M",
    "GRIP_SEATED_APPROACH_OFFSET_M",
    "PAD_LENGTH_M",
    "grip_offset_admissible",
    "GRAPPLE_TOOL_OFFSET_POS",
    "MAX_CLEAR_OPENING_M",
    "PAD_HALF_WIDTH_M",
    "PAD_SPAN_FROM_FLANGE_M",
    "PALM_FACE_FROM_FLANGE_M",
    "RATED_GRIP_FORCE_N",
    "SLOT_ENTRY_RAMP_CATCH_M",
    "SLOT_ENTRY_RAMP_DEG",
    "SLOT_ENTRY_RAMP_LENGTH_M",
    "SLOT_ENTRY_RAMP_THICKNESS_M",
    "SLOT_ENTRY_RAMP_WIDTH_M",
    "SLOT_FLOOR_TOP_Z",
    "SLOT_LIP_BOTTOM_Z",
    "SLOT_MOUTH_X",
    "approach_clearance_per_side_m",
    "clear_opening_m",
    "drive_torque_for_grip_force_nm",
    "wedge_half_height_at",
    "wedge_taper_deg",
]
