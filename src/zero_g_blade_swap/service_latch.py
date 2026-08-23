"""The robot-side service latch: dimensions, and where they come from.

This module holds numbers, not simulator objects, and imports nothing from Isaac
Lab, for the same reason ``grapple_geometry`` does not: the latch's dimensional
contract is the part of this design most likely to be broken by a later edit,
and keeping it importable without a simulator means the test suite can defend it
on every commit.

**Why the latch is on the robot and not on the module.** Section 8.4 of
``docs/service_interface_spec.md`` sweeps the gripped section's half-height from
1 to 43 mm and finds that the volume immediately ahead of the finger pads --
between the palm face at 90 mm and the seated pads at 105 mm -- belongs to the
hand, so a serviceable module cannot carry a positive axial stop forward of the
pads. That measurement ends with the rule this module is built on:

    An axial lock therefore has to come from the *end-effector*: V-grooved
    fingers, or a powered latch. It cannot come from the module.

Two module-side attempts were built and refuted before that rule was written --
an anti-yaw yoke that cost the insert skill 67 points, and a keyed pin that
clamped beautifully and could not be installed because its nose flange overlaps
the palm by 45 mm at every closure. This is the third design, and it is the
first one on the correct side of the interface.

**Why it fits.** ``evidence/gripper_collision_envelope.json`` measures the whole
2F-85 as reaching no further than 162.11 mm from the flange, and no wider than
37.5 mm on the third axis. The module's shaft runs from 168 mm to 248 mm from
the flange at the seated grip. There is therefore an 80 mm length of 30 x 30 mm
shaft, immediately behind the collar, that no part of the hand can ever occupy,
and the latch lives in it. Every dimension below is a consequence of that
window, of the collar's own section, and of the rack channel the assembly has to
pass through -- none of them is chosen.

**What it is, mechanically.** A two-jaw powered latch on a carriage. The jaws
extend along the approach axis until their lips clear the collar's proximal
shoulder, then close on the shaft's flanks. That gives form closure in four of
the six degrees of freedom the pads leave open:

* pulling the module away from the flange is taken by the lips bearing on the
  collar shoulder, instead of by wedge friction rated at 69 N;
* pushing it toward the flange is taken by the collar on the pad leading faces,
  which is the depth stop the pin already had;
* rotation about the pin axis -- the rotation section 8 measured at 0.93 rad in
  failing extractions and could not fix from the module side -- is taken by two
  plane contacts on the shaft's flat flanks;
* lateral translation on the third axis is taken by the same two planes.

``scripts/check_service_latch_clearance.py`` derives every clearance below from
``evidence/gripper_collision_envelope.json`` and this file, with no simulator,
and refuses to pass if any of them closes.
"""

from __future__ import annotations

from zero_g_blade_swap.grapple_geometry import (
    GRAPPLE_PIN_COLLAR_X,
    GRAPPLE_PIN_GRIP_OFFSET,
    GRAPPLE_PIN_HALF_WIDTH_Y,
    GRAPPLE_PIN_SHAFT_HALF_HEIGHT,
    GRAPPLE_PIN_SHAFT_X,
    GRAPPLE_TOOL_OFFSET_POS,
    PAD_SPAN_FROM_FLANGE_M,
)

# ---------------------------------------------------------------------------
# Where the module's features sit in the hand's own frame, at the seated grip.
#
# A zero tool-to-grip error means the pads straddle the wedge with their leading
# faces on the collar, so the flange sits one tool offset behind the grip point
# and every pin feature can be expressed as a depth from the flange.

#: Blade-local x of the flange when the grip error is zero.
SEATED_FLANGE_BLADE_X = GRAPPLE_PIN_GRIP_OFFSET[0] - GRAPPLE_TOOL_OFFSET_POS[2]
#: Depth from the flange of the collar's proximal face -- the shoulder a lip has
#: to sit behind. 168 mm, six millimetres past the pads' 162 mm leading edge.
COLLAR_SHOULDER_FROM_FLANGE_M = GRAPPLE_PIN_COLLAR_X[1] - SEATED_FLANGE_BLADE_X
#: Depth from the flange of the module's own face, where the shaft ends.
MODULE_FACE_FROM_FLANGE_M = GRAPPLE_PIN_SHAFT_X[1] - SEATED_FLANGE_BLADE_X
#: The window the latch has to live inside: 80 mm of shaft that the hand cannot
#: reach, because the hand stops at 162.11 mm.
LATCH_WINDOW_FROM_FLANGE_M = (COLLAR_SHOULDER_FROM_FLANGE_M, MODULE_FACE_FROM_FLANGE_M)

# ---------------------------------------------------------------------------
# The jaws, engaged. Coordinates are the wrist_3_link frame measured in
# `evidence/gripper_collision_envelope.json`: x is the closing axis, y the third
# axis, z the approach axis out of the flange.

#: Length of a jaw along the approach axis. Short, because the load path is the
#: lip on the collar shoulder rather than a long clamp on the shaft, and a
#: shorter jaw leaves more of the seek stroke below.
JAW_LENGTH_M = 0.022
#: Thickness of the lip that bears on the collar shoulder.
LIP_THICKNESS_M = 0.0025
#: Half-gap between the closed jaw webs and the shaft flank. Half a millimetre
#: per side: this is a clamp, not a capture, because the pads have already
#: located the module by the time it closes.
WEB_INNER_HALF_GAP_M = GRAPPLE_PIN_HALF_WIDTH_Y + 0.0005
#: Outer face of a jaw web on the third axis.
WEB_OUTER_HALF_GAP_M = 0.046
#: Half-height of a jaw on the closing axis. Inside the collar's own 45 mm, so
#: the lip is fully backed by shoulder rather than overhanging its rim.
JAW_HALF_HEIGHT_M = 0.040
#: How far in from the centre line a lip reaches. The two opposing lips stop
#: 8 mm apart, which is clearance they need because nothing guarantees they
#: arrive simultaneously.
LIP_INNER_HALF_GAP_M = 0.004
#: The lip only exists above and below the shaft, because that is the only part
#: of the collar's proximal face that is not the shaft's own root. Its inner
#: edge clears the shaft by 3 mm.
LIP_INNER_HALF_HEIGHT_M = GRAPPLE_PIN_SHAFT_HALF_HEIGHT + 0.003
#: Bearing area of one lip band, on one jaw.
LIP_BEARING_AREA_M2 = (GRAPPLE_PIN_HALF_WIDTH_Y - LIP_INNER_HALF_GAP_M) * (
    JAW_HALF_HEIGHT_M - LIP_INNER_HALF_HEIGHT_M
)
#: Four of them: two jaws, each with a band above and below the shaft.
LIP_TOTAL_BEARING_AREA_M2 = 4.0 * LIP_BEARING_AREA_M2

# ---------------------------------------------------------------------------
# The two poses, and the strokes between them.

#: How far outward on the third axis a jaw travels to release. Sized so the
#: released lip clears the collar's own half-width by 20 mm, which is what lets
#: the module leave the hand at all. It also has to park the lip clear of the
#: fingers, and that is the tighter of the two: see
#: ``scripts/check_service_latch_clearance.py``, which resolves the hand's
#: half-extent by depth rather than taking its global maximum, because past
#: ``FINGERS_ONLY_DEPTH_FROM_FLANGE_M`` the only body left is a 13.5 mm-wide
#: inner finger.
CLOSE_STROKE_M = 0.031
#: How far back along the approach axis the carriage travels to stow. Sized by
#: the rack: see ``STOWED_DEPTH_FROM_FLANGE_M`` below.
EXTEND_STROKE_M = 0.025
#: Depth range a stowed jaw occupies. This is the number the rack constrains.
STOWED_DEPTH_FROM_FLANGE_M = (
    COLLAR_SHOULDER_FROM_FLANGE_M - EXTEND_STROKE_M,
    COLLAR_SHOULDER_FROM_FLANGE_M - EXTEND_STROKE_M + JAW_LENGTH_M,
)
#: Depth range an engaged jaw occupies at zero seek.
ENGAGED_DEPTH_FROM_FLANGE_M = (
    COLLAR_SHOULDER_FROM_FLANGE_M,
    COLLAR_SHOULDER_FROM_FLANGE_M + JAW_LENGTH_M,
)

#: The carriage seeks the collar rather than assuming it, and **both ends of its
#: travel are derived rather than chosen**.
#:
#: A tapered wedge does not seat a module at one depth. Where the pin sits along
#: the approach axis is set by where its thickness equals the pad opening, so it
#: moves with the closure command -- section 4 of the interface specification
#: measures about 12.5 mm of module travel on every capture, and a seated grip
#: offset between 12 and 19 mm. A latch authored at the nominal collar depth
#: would close on the collar's rim, or on air.
#:
#: The carriage therefore drives along the approach axis until it finds the
#: shoulder, exactly as a docking latch does, and the travel it used is recorded
#: in the report.
#:
#: *Near end.* An engaged jaw may not come back inside the hand. The deepest
#: gripper body measured anywhere in the closure range is the pad leading edge
#: at 162 mm, so the jaw's near face stops one millimetre past it.
#: *Far end.* An engaged jaw may not reach the module's own face, or it is
#: clamping the chassis instead of the shaft.
AXIAL_SEEK_MIN_M = (PAD_SPAN_FROM_FLANGE_M[1] + 0.001) - COLLAR_SHOULDER_FROM_FLANGE_M
AXIAL_SEEK_MAX_M = 0.040
AXIAL_SEEK_RANGE_M = (AXIAL_SEEK_MIN_M, AXIAL_SEEK_MAX_M)


def engaged_jaw_far_depth_m(seek_m: float = 0.0) -> float:
    """Depth from the flange reached by an engaged jaw at this seek travel."""

    return ENGAGED_DEPTH_FROM_FLANGE_M[1] + seek_m


def release_before_blade_centre_x_m(
    slot_mouth_x: float,
    blade_half_length_m: float,
    seek_m: float = 0.0,
) -> float:
    """Module-centre depth at which an engaged jaw would enter the slot mouth.

    The latch carries the module to the rack and the **rails take the last of
    the seating**, because the jaws sit behind the collar and the collar ends up
    five millimetres outside the mouth when the module is home. That is not a
    limitation to work around; it is the same rule section 3.1 of the interface
    specification applies to the gripper, applied to the mechanism bolted beside
    it. The workflow releases before this depth, and the number is derived here
    so the driver cannot hold a different one.
    """

    return slot_mouth_x + blade_half_length_m + (MODULE_FACE_FROM_FLANGE_M - engaged_jaw_far_depth_m(seek_m))


def seek_within_range(seek_m: float) -> bool:
    """Whether a measured collar depth is inside the carriage's travel."""

    return AXIAL_SEEK_RANGE_M[0] <= seek_m <= AXIAL_SEEK_RANGE_M[1]


def jaw_boxes(*, engaged: bool, seek_m: float = 0.0) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Return the latch's boxes as ``(name, centre, size)`` in the wrist frame.

    One list, used by the USD spawner, by the clearance check, and by the tests,
    so a dimension cannot be right in one of them and wrong in another. Only the
    ``+y`` jaw is returned; the ``-y`` jaw is its mirror.
    """

    if engaged:
        near = ENGAGED_DEPTH_FROM_FLANGE_M[0] + seek_m
        inner_web = WEB_INNER_HALF_GAP_M
        inner_lip = LIP_INNER_HALF_GAP_M
    else:
        near = STOWED_DEPTH_FROM_FLANGE_M[0]
        inner_web = WEB_INNER_HALF_GAP_M + CLOSE_STROKE_M
        inner_lip = LIP_INNER_HALF_GAP_M + CLOSE_STROKE_M
    outer_web = inner_web + (WEB_OUTER_HALF_GAP_M - WEB_INNER_HALF_GAP_M)
    boxes = [
        (
            "web",
            (0.0, 0.5 * (inner_web + outer_web), near + 0.5 * JAW_LENGTH_M),
            (2.0 * JAW_HALF_HEIGHT_M, outer_web - inner_web, JAW_LENGTH_M),
        )
    ]
    for sign, name in ((1.0, "lip_upper"), (-1.0, "lip_lower")):
        band_centre = sign * 0.5 * (LIP_INNER_HALF_HEIGHT_M + JAW_HALF_HEIGHT_M)
        boxes.append(
            (
                name,
                (band_centre, 0.5 * (inner_lip + inner_web), near + 0.5 * LIP_THICKNESS_M),
                (
                    JAW_HALF_HEIGHT_M - LIP_INNER_HALF_HEIGHT_M,
                    inner_web - inner_lip,
                    LIP_THICKNESS_M,
                ),
            )
        )
    return tuple(boxes)


#: The rail that carries a jaw carriage back to the flange. Outside the hand on
#: the third axis and inside the rack channel on it as well, so the assembly can
#: follow the module to the mouth without touching either.
RAIL_INNER_HALF_GAP_M = 0.046
RAIL_OUTER_HALF_GAP_M = 0.058
RAIL_HALF_HEIGHT_M = 0.018
RAIL_DEPTH_FROM_FLANGE_M = (0.030, STOWED_DEPTH_FROM_FLANGE_M[0])

# ---------------------------------------------------------------------------
# Ratings.

#: What the latch has to hold, from the interface specification: the worst-case
#: contact reaction of the promoted insertion policy. The passive wedge holds
#: 69 N against this, which is a 4% margin, and it is the reason a form lock
#: exists at all.
REQUIRED_AXIAL_CAPACITY_N = 66.4
#: Rated loads of the modelled latch. The fixed joint that implements it carries
#: these as PhysX break thresholds, so "the latch holds" is a number that a run
#: can fail rather than an assumption. They are swept, not chosen: see
#: ``scripts/run_robot_carried.sh sweep``.
RATED_FORCE_N = 600.0
RATED_TORQUE_NM = 30.0


def lip_bearing_stress_mpa(force_n: float = RATED_FORCE_N) -> float:
    """Bearing stress in the collar shoulder at a given axial load."""

    return 1.0e-6 * force_n / LIP_TOTAL_BEARING_AREA_M2


__all__ = [
    "AXIAL_SEEK_MAX_M",
    "AXIAL_SEEK_MIN_M",
    "AXIAL_SEEK_RANGE_M",
    "CLOSE_STROKE_M",
    "COLLAR_SHOULDER_FROM_FLANGE_M",
    "ENGAGED_DEPTH_FROM_FLANGE_M",
    "EXTEND_STROKE_M",
    "JAW_HALF_HEIGHT_M",
    "JAW_LENGTH_M",
    "LATCH_WINDOW_FROM_FLANGE_M",
    "LIP_BEARING_AREA_M2",
    "LIP_INNER_HALF_GAP_M",
    "LIP_INNER_HALF_HEIGHT_M",
    "LIP_THICKNESS_M",
    "LIP_TOTAL_BEARING_AREA_M2",
    "MODULE_FACE_FROM_FLANGE_M",
    "RAIL_DEPTH_FROM_FLANGE_M",
    "RAIL_HALF_HEIGHT_M",
    "RAIL_INNER_HALF_GAP_M",
    "RAIL_OUTER_HALF_GAP_M",
    "RATED_FORCE_N",
    "RATED_TORQUE_NM",
    "REQUIRED_AXIAL_CAPACITY_N",
    "SEATED_FLANGE_BLADE_X",
    "STOWED_DEPTH_FROM_FLANGE_M",
    "WEB_INNER_HALF_GAP_M",
    "WEB_OUTER_HALF_GAP_M",
    "engaged_jaw_far_depth_m",
    "jaw_boxes",
    "lip_bearing_stress_mpa",
    "release_before_blade_centre_x_m",
    "seek_within_range",
]
