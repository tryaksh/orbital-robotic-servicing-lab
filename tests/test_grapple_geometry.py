"""Dimensional contract for the head-on grapple pin.

The pin's dimensions are not free parameters. Each one is bounded by something
measured about the Robotiq 2F-85 or about the rack, and a later edit that
violates one of those bounds would produce an interface that silently cannot be
grasped, cannot be inserted, or slides straight out of the gripper.

These import only ``zero_g_blade_swap.grapple_geometry``, which pulls in no
simulator, so they run in CI on every commit rather than only on a machine with
Isaac Sim installed.

The measurements they defend live in ``evidence/gripper_collision_envelope.json``.
"""

from __future__ import annotations

import pytest

from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    CLOSING_RATE_M_PER_RAD,
    EXTRACTED_BLADE_CENTRE_X,
    FINGER_JOINT_RANGE_RAD,
    FINGERS_ONLY_DEPTH_FROM_FLANGE_M,
    GRAPPLE_PIN_COLLAR_HALF_HEIGHT,
    GRAPPLE_PIN_COLLAR_X,
    GRAPPLE_PIN_GRIP_OFFSET,
    GRAPPLE_PIN_HALF_WIDTH_Y,
    GRAPPLE_PIN_SHAFT_HALF_HEIGHT,
    GRAPPLE_PIN_SHAFT_X,
    GRAPPLE_PIN_KEY_HALF_HEIGHT,
    GRAPPLE_PIN_KEY_X,
    GRAPPLE_PIN_NOSE_HALF_HEIGHT,
    GRAPPLE_PIN_NOSE_X,
    GRAPPLE_TOOL_OFFSET_POS,
    GRAPPLE_YOKE_HALF_GAP_M,
    GRAPPLE_YOKE_HALF_HEIGHT,
    GRAPPLE_YOKE_MOUTH_HALF_GAP_M,
    GRAPPLE_YOKE_PARALLEL_X,
    GRAPPLE_YOKE_X,
    MAX_CLEAR_OPENING_M,
    NON_FINGER_HALF_WIDTH_M,
    PAD_HALF_WIDTH_M,
    PAD_SPAN_FROM_FLANGE_M,
    SLOT_FLOOR_TOP_Z,
    SLOT_LIP_BOTTOM_Z,
    SLOT_MOUTH_X,
    approach_clearance_per_side_m,
    clear_opening_m,
    drive_torque_for_grip_force_nm,
    key_seat_axial_travel_m,
    yoke_flare_deg,
    yoke_free_yaw_rad,
    yoke_lead_in_catch_m,
    yoke_mouth_depth_from_flange_m,
)

BLADE_CENTRE_Z = 0.72
BLADE_INSERTED_CENTRE_X = 0.75


def test_zero_is_fully_open() -> None:
    """The measured convention, which the contact task still has backwards."""

    lower, upper = FINGER_JOINT_RANGE_RAD
    assert clear_opening_m(lower) == pytest.approx(MAX_CLEAR_OPENING_M)
    assert clear_opening_m(upper) == pytest.approx(0.0, abs=1.0e-3)
    assert clear_opening_m(0.4) < clear_opening_m(0.2)


def test_pin_sections_are_contiguous_and_run_free_end_to_blade() -> None:
    assert GRAPPLE_PIN_NOSE_X[1] == GRAPPLE_PIN_KEY_X[0]
    assert GRAPPLE_PIN_KEY_X[1] == GRAPPLE_PIN_COLLAR_X[0]
    assert GRAPPLE_PIN_COLLAR_X[1] == GRAPPLE_PIN_SHAFT_X[0]
    assert GRAPPLE_PIN_SHAFT_X[1] == pytest.approx(-0.5 * BLADE_LENGTH_M)
    assert (
        GRAPPLE_PIN_NOSE_X[0]
        < GRAPPLE_PIN_KEY_X[0]
        < GRAPPLE_PIN_COLLAR_X[0]
        < GRAPPLE_PIN_SHAFT_X[0]
    )


def test_the_gripped_section_is_flat() -> None:
    """The whole design. A taper gives the pads a line contact, which resists no
    moment about the closing axis; a flat gives them a plane, which resists every
    moment in that plane. The module swinging end-for-end during the relocation
    transit is what a line contact costs, so a scalar half-height here is the
    contract, not an implementation detail."""

    assert isinstance(GRAPPLE_PIN_KEY_HALF_HEIGHT, float)
    assert GRAPPLE_PIN_KEY_HALF_HEIGHT == pytest.approx(GRAPPLE_PIN_SHAFT_HALF_HEIGHT)


def test_both_axial_stops_are_taller_than_the_aperture_can_open() -> None:
    """The whole point of the pocket, and the correction of a measured failure.

    A shoulder shorter than the aperture stops nothing: at 60 mm the module escaped
    32-55 mm axially and extraction scored 0.00%, because the drive yields under
    drag and the pads splay toward their 87 mm opening. Both walls now exceed it, so
    the pads are trapped whatever the jaws do.
    """

    assert 2.0 * GRAPPLE_PIN_NOSE_HALF_HEIGHT > MAX_CLEAR_OPENING_M
    assert 2.0 * GRAPPLE_PIN_COLLAR_HALF_HEIGHT > MAX_CLEAR_OPENING_M
    # Anything under a few millimetres a side makes the head-on approach a
    # coin toss for a learned policy.
    assert approach_clearance_per_side_m() >= 0.007


def test_the_pads_can_never_open_around_the_collar() -> None:
    """The collar is a depth stop only if it is wider than the full aperture."""

    assert 2.0 * GRAPPLE_PIN_COLLAR_HALF_HEIGHT > MAX_CLEAR_OPENING_M


def test_the_nose_flange_traps_a_closed_pad() -> None:
    """Axial capture is now a dimension, not a friction coefficient.

    The flange has to be taller than the key so a pad closed onto the key cannot
    ride off the free end, and the travel it allows has to be real but bounded --
    enough that a capture landing anywhere inside the skill's 10 mm grip
    tolerance still seats on the flat, and not so much that the module can wander.
    """

    assert GRAPPLE_PIN_NOSE_HALF_HEIGHT > GRAPPLE_PIN_KEY_HALF_HEIGHT
    # Bounded below by how accurately a capture places the pads axially, measured
    # at 0.15 mm on the settled grasp, and above by the 2 mm the pull gate calls a
    # slipped grip. A first attempt at 23 mm failed that gate at 51.6 N, because
    # friction alone carries about 52 N and the module simply slid until the
    # flange caught it.
    assert 0.005 <= key_seat_axial_travel_m() <= 0.012


def test_only_the_shaft_enters_the_slot_and_it_fits() -> None:
    collar_face_x = BLADE_INSERTED_CENTRE_X + GRAPPLE_PIN_COLLAR_X[1]
    assert collar_face_x <= SLOT_MOUTH_X, "the gripper would be inside the rack at full insertion"
    assert BLADE_CENTRE_Z - GRAPPLE_PIN_SHAFT_HALF_HEIGHT > SLOT_FLOOR_TOP_Z
    assert BLADE_CENTRE_Z + GRAPPLE_PIN_SHAFT_HALF_HEIGHT < SLOT_LIP_BOTTOM_Z


def test_the_pin_is_wide_enough_to_load_the_whole_pad() -> None:
    assert GRAPPLE_PIN_HALF_WIDTH_Y >= PAD_HALF_WIDTH_M


def test_tool_frame_and_grip_point_agree_with_the_measured_pad_span() -> None:
    """A zero tool-to-grip error must mean the pads are on the collar."""

    pad_length = PAD_SPAN_FROM_FLANGE_M[1] - PAD_SPAN_FROM_FLANGE_M[0]
    assert GRAPPLE_TOOL_OFFSET_POS[2] == pytest.approx(
        0.5 * (PAD_SPAN_FROM_FLANGE_M[0] + PAD_SPAN_FROM_FLANGE_M[1])
    )
    # Pads centred on the grip point put their leading face on the collar.
    leading_face = GRAPPLE_PIN_GRIP_OFFSET[0] + 0.5 * pad_length
    assert leading_face == pytest.approx(GRAPPLE_PIN_COLLAR_X[0])
    assert GRAPPLE_PIN_GRIP_OFFSET[1:] == (0.0, 0.0)


def test_seated_pads_sit_entirely_on_the_flat_key() -> None:
    """A pad half on the key and half on a stop is a vertex contact, which is the
    line-contact problem again in a different place."""

    pad_length = PAD_SPAN_FROM_FLANGE_M[1] - PAD_SPAN_FROM_FLANGE_M[0]
    trailing_edge = GRAPPLE_PIN_COLLAR_X[0] - pad_length
    assert trailing_edge > GRAPPLE_PIN_KEY_X[0], "the pads overhang the key's free end"
    assert key_seat_axial_travel_m() == pytest.approx(trailing_edge - GRAPPLE_PIN_KEY_X[0])


def test_extraction_target_actually_clears_the_blade() -> None:
    assert pytest.approx(SLOT_MOUTH_X) == EXTRACTED_BLADE_CENTRE_X + 0.5 * BLADE_LENGTH_M
    assert EXTRACTED_BLADE_CENTRE_X < BLADE_INSERTED_CENTRE_X


def test_grip_force_converts_to_drive_torque_by_virtual_work() -> None:
    assert drive_torque_for_grip_force_nm(235.0) == pytest.approx(235.0 * CLOSING_RATE_M_PER_RAD)
    # The inherited 10 N-m drive is worth roughly 100 N of pad force, which is
    # the number the rated-force experiment was argued against.
    assert 90.0 < 10.0 / CLOSING_RATE_M_PER_RAD < 110.0


# ---------------------------------------------------------------------------
# The anti-yaw yoke. Every bound here is a measured gripper number, so an edit
# that widens the walls, lengthens them, or moves them cannot silently produce a
# feature that fouls the gripper it is meant to engage.


def test_yoke_stays_inside_the_fingers_only_band() -> None:
    """Walls narrower than the knuckles must stay deeper than the knuckles reach.

    Measured over the whole 0 to 0.8203 rad closure range: no gripper body other
    than an inner finger reaches past 0.1245 m from the flange, and the inner
    knuckles reach 17.5 mm on the third axis against the fingers' 13.5 mm. Walls
    at a 15 mm half-gap therefore only work inside that band.
    """

    assert GRAPPLE_YOKE_HALF_GAP_M < NON_FINGER_HALF_WIDTH_M
    assert yoke_mouth_depth_from_flange_m() > FINGERS_ONLY_DEPTH_FROM_FLANGE_M, (
        f"the yoke mouth reaches {yoke_mouth_depth_from_flange_m():.4f} m from the flange, inside the "
        f"{FINGERS_ONLY_DEPTH_FROM_FLANGE_M:.4f} m band where an inner knuckle can reach "
        f"{NON_FINGER_HALF_WIDTH_M * 1000:.1f} mm and would foul a {GRAPPLE_YOKE_HALF_GAP_M * 1000:.1f} mm wall"
    )


def test_yoke_walls_clear_the_fingers_but_only_just() -> None:
    """Wide enough that a finger fits, narrow enough that yaw is constrained."""

    assert GRAPPLE_YOKE_HALF_GAP_M > PAD_HALF_WIDTH_M
    clearance = GRAPPLE_YOKE_HALF_GAP_M - PAD_HALF_WIDTH_M
    assert 0.0005 <= clearance <= 0.003, f"{clearance * 1000:.2f} mm per side is outside the useful band"


def test_yoke_sits_on_the_wedge_flanks_without_widening_the_pin() -> None:
    """The walls are the pin's own side faces raised, so the pin gains no width."""

    assert pytest.approx(GRAPPLE_PIN_HALF_WIDTH_Y) == GRAPPLE_YOKE_HALF_GAP_M


def test_yoke_lies_on_the_gripped_section_and_ends_at_the_collar() -> None:
    """The yoke is refuted and off by default, and the keyed section supersedes
    what it was built to do -- it blocks rotation by form, which is what the walls
    were trying to approximate. Its dimensional contract is kept because the code
    is still present and inert, and an inert feature with an unchecked contract is
    how a later session turns it on and gets a silently wrong scene."""

    mouth_x, collar_x = GRAPPLE_YOKE_X
    assert collar_x == pytest.approx(GRAPPLE_PIN_COLLAR_X[0])
    assert mouth_x > GRAPPLE_PIN_KEY_X[0], "the yoke must not overhang the key's free end"
    parallel_low, parallel_high = GRAPPLE_YOKE_PARALLEL_X
    assert mouth_x < parallel_low < parallel_high
    assert parallel_high == pytest.approx(collar_x)


def test_yoke_never_exceeds_the_collar_envelope() -> None:
    """The collar is already the absolute depth stop; the yoke must not out-reach it."""

    assert GRAPPLE_YOKE_HALF_HEIGHT <= GRAPPLE_PIN_COLLAR_HALF_HEIGHT


def test_yoke_mouth_catches_more_than_the_parallel_section_allows() -> None:
    """A lead-in, for the reason the rack has one: a blind 1.5 mm slot is a trap."""

    assert GRAPPLE_YOKE_MOUTH_HALF_GAP_M > GRAPPLE_YOKE_HALF_GAP_M
    assert yoke_lead_in_catch_m() > 3.0 * (GRAPPLE_YOKE_HALF_GAP_M - PAD_HALF_WIDTH_M)
    assert 10.0 <= yoke_flare_deg() <= 30.0


def test_yoke_predicts_a_large_reduction_in_free_yaw() -> None:
    """Geometry only. What the walls do under load is measured, not asserted."""

    measured_free_yaw_without_a_yoke_rad = 0.93
    assert yoke_free_yaw_rad() < 0.25 * measured_free_yaw_without_a_yoke_rad
