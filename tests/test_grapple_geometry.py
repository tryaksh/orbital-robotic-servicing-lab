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
    GRAPPLE_PIN_WEDGE_HALF_HEIGHT,
    GRAPPLE_PIN_WEDGE_X,
    GRAPPLE_TOOL_OFFSET_POS,
    MAX_CLEAR_OPENING_M,
    NON_FINGER_HALF_WIDTH_M,
    PAD_HALF_WIDTH_M,
    PAD_SPAN_FROM_FLANGE_M,
    PALM_FACE_FROM_FLANGE_M,
    SLOT_FLOOR_TOP_Z,
    SLOT_LIP_BOTTOM_Z,
    SLOT_MOUTH_X,
    approach_clearance_per_side_m,
    clear_opening_m,
    drive_torque_for_grip_force_nm,
    wedge_half_height_at,
    wedge_taper_deg,
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
    assert GRAPPLE_PIN_WEDGE_X[1] == GRAPPLE_PIN_COLLAR_X[0]
    assert GRAPPLE_PIN_COLLAR_X[1] == GRAPPLE_PIN_SHAFT_X[0]
    assert GRAPPLE_PIN_SHAFT_X[1] == pytest.approx(-0.5 * BLADE_LENGTH_M)
    assert GRAPPLE_PIN_WEDGE_X[0] < GRAPPLE_PIN_COLLAR_X[0] < GRAPPLE_PIN_SHAFT_X[0]


def test_wedge_thickens_toward_its_free_end() -> None:
    """Backwards, and pulling would squeeze the pin out instead of jamming it."""

    distal, proximal = GRAPPLE_PIN_WEDGE_HALF_HEIGHT
    assert distal > proximal
    assert wedge_half_height_at(GRAPPLE_PIN_WEDGE_X[0]) == pytest.approx(distal)
    assert wedge_half_height_at(GRAPPLE_PIN_WEDGE_X[1]) == pytest.approx(proximal)


def test_the_pads_can_open_around_the_wedge() -> None:
    assert 2.0 * GRAPPLE_PIN_WEDGE_HALF_HEIGHT[0] < MAX_CLEAR_OPENING_M
    # Anything under a few millimetres a side makes the head-on approach a
    # coin toss for a learned policy.
    assert approach_clearance_per_side_m() >= 0.007


def test_the_pads_can_never_open_around_the_collar() -> None:
    """The collar is a depth stop only if it is wider than the full aperture."""

    assert 2.0 * GRAPPLE_PIN_COLLAR_HALF_HEIGHT > MAX_CLEAR_OPENING_M


def test_the_wedge_is_steep_enough_to_be_worth_having() -> None:
    """Axial capacity goes as the sine of this angle."""

    assert 15.0 <= wedge_taper_deg() <= 35.0


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


def test_seated_pads_sit_entirely_on_the_wedge() -> None:
    """Their trailing edge must be on a sloped face, not off the free end.

    A pad face bearing on the wedge's rim is a vertex contact, whose normal has
    no axial component, so the wedge would carry no load at all.
    """

    pad_length = PAD_SPAN_FROM_FLANGE_M[1] - PAD_SPAN_FROM_FLANGE_M[0]
    trailing_edge = GRAPPLE_PIN_COLLAR_X[0] - pad_length
    assert trailing_edge > GRAPPLE_PIN_WEDGE_X[0], "the pads overhang the wedge's free end"


def test_extraction_target_actually_clears_the_blade() -> None:
    assert pytest.approx(SLOT_MOUTH_X) == EXTRACTED_BLADE_CENTRE_X + 0.5 * BLADE_LENGTH_M
    assert EXTRACTED_BLADE_CENTRE_X < BLADE_INSERTED_CENTRE_X


def test_grip_force_converts_to_drive_torque_by_virtual_work() -> None:
    assert drive_torque_for_grip_force_nm(235.0) == pytest.approx(235.0 * CLOSING_RATE_M_PER_RAD)
    # The inherited 10 N-m drive is worth roughly 100 N of pad force, which is
    # the number the rated-force experiment was argued against.
    assert 90.0 < 10.0 / CLOSING_RATE_M_PER_RAD < 110.0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The pin has to fit inside the hand that grips it.
#
# This is the bound the keyed redesign broke, and nothing caught it: five
# parameter corrections were made against a 0.00% extraction before anyone asked
# whether the new geometry was physically inside the gripper. It was, by 45.0 mm,
# at every closure including fully open. ``evidence/grapple_pin_keyed_interference.json``.


def test_pin_clears_the_gripper_at_every_closure_the_pin_allows() -> None:
    """No pin section may occupy the volume a non-pad gripper body occupies.

    The check runs at the closures the pin itself permits, because the pads stop
    on the pin and everything else in the hand is positioned by that stop. Asking
    about tighter closures asks whether the pin fits a hand closed on nothing,
    which condemned this very geometry when it was first written.
    """

    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "check_pin_gripper_clearance.py"
    spec = importlib.util.spec_from_file_location("check_pin_gripper_clearance", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    result = module.analyse(None)
    assert result["passed"], (
        f"{len(result['interferences'])} interfering slices, worst "
        f"{result['worst_interference_m'] * 1000:.1f} mm: {result['interferences'][:3]}"
    )


def test_no_pin_section_reaches_closer_to_the_flange_than_the_palm() -> None:
    """``PALM_FACE_FROM_FLANGE_M`` exists to be read, and the keyed pin did not.

    A seated grip puts the blade's grip point on the tool frame origin, so the
    pin's free end sits at ``TOOL_Z + GRIP_OFFSET_X - free_end_x`` from the
    flange. Anything closer than the palm face is inside the gripper.
    """

    from zero_g_blade_swap import grapple_geometry

    free_end = getattr(grapple_geometry, "GRAPPLE_PIN_NOSE_X", None) or GRAPPLE_PIN_WEDGE_X
    free_end_x = free_end[0]
    depth = free_end_x - GRAPPLE_PIN_GRIP_OFFSET[0] + GRAPPLE_TOOL_OFFSET_POS[2]
    assert depth >= PALM_FACE_FROM_FLANGE_M, (
        f"the pin's free end sits {depth * 1000:.1f} mm from the flange, inside the palm at "
        f"{PALM_FACE_FROM_FLANGE_M * 1000:.1f} mm"
    )
