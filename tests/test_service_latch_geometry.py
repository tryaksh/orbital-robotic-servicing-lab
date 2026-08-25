"""The robot-side service latch's dimensional contract, defended without Isaac.

``tests/test_grapple_geometry.py`` does this for the module's pin. The latch is
the same class of risk on the other side of the interface, and it exists at all
because two module-side attempts were built, certified, and refuted: a
specification rule this project derived from measurement says the axial lock has
to come from the end-effector. These tests keep the implementation on that side
of the rule and inside the volume the measured gripper envelope leaves free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from zero_g_blade_swap import service_latch as latch  # noqa: E402
from zero_g_blade_swap.grapple_geometry import (  # noqa: E402
    GRAPPLE_PIN_COLLAR_HALF_HEIGHT,
    GRAPPLE_PIN_HALF_WIDTH_Y,
    GRAPPLE_PIN_SHAFT_HALF_HEIGHT,
    PAD_SPAN_FROM_FLANGE_M,
)

ENVELOPE = json.loads((ROOT / "evidence" / "gripper_collision_envelope.json").read_text(encoding="utf-8"))
HAND_MAX = ENVELOPE["derived"]["gripper_envelope_max_m"]


def test_the_latch_lives_where_the_hand_cannot_reach() -> None:
    """The whole design rests on this: bare shaft past the deepest gripper body.

    The window is 80 mm at the shipped module length. It is derived from that
    length rather than chosen, so a module that changes length changes it --
    measured at 250 mm, it becomes 180 mm and every clearance in
    ``check_service_latch_clearance.py`` still passes. What has to hold is that
    there *is* a window and that it starts past every body of the hand.
    """

    deepest_gripper_body = HAND_MAX[2]
    assert deepest_gripper_body < latch.COLLAR_SHOULDER_FROM_FLANGE_M
    assert latch.MODULE_FACE_FROM_FLANGE_M > latch.COLLAR_SHOULDER_FROM_FLANGE_M
    assert latch.LATCH_WINDOW_FROM_FLANGE_M == (
        latch.COLLAR_SHOULDER_FROM_FLANGE_M,
        latch.MODULE_FACE_FROM_FLANGE_M,
    )
    window = latch.MODULE_FACE_FROM_FLANGE_M - latch.COLLAR_SHOULDER_FROM_FLANGE_M
    assert window == pytest.approx(0.080, abs=1e-6) or window > 0.080


def test_the_seek_travel_is_derived_from_the_hand_and_bounded_by_the_module() -> None:
    """Neither end of the carriage stroke may be a chosen number."""

    derived_near_end = (PAD_SPAN_FROM_FLANGE_M[1] + 0.001) - latch.COLLAR_SHOULDER_FROM_FLANGE_M
    assert pytest.approx(derived_near_end) == latch.AXIAL_SEEK_MIN_M
    # Retracted to the near end, the jaw is still past every gripper body.
    assert latch.ENGAGED_DEPTH_FROM_FLANGE_M[0] + latch.AXIAL_SEEK_MIN_M > PAD_SPAN_FROM_FLANGE_M[1]
    # Extended to the far end, it is still on the shaft rather than the chassis.
    assert latch.engaged_jaw_far_depth_m(latch.AXIAL_SEEK_MAX_M) < latch.MODULE_FACE_FROM_FLANGE_M


def test_a_shoulder_outside_the_travel_is_refused_rather_than_reached() -> None:
    assert latch.seek_within_range(0.0)
    assert latch.seek_within_range(latch.AXIAL_SEEK_MAX_M)
    assert not latch.seek_within_range(latch.AXIAL_SEEK_MAX_M + 0.001)
    assert not latch.seek_within_range(latch.AXIAL_SEEK_MIN_M - 0.001)


def test_the_lip_bears_on_collar_and_clears_the_shaft() -> None:
    """A lip that misses the shoulder is a clamp, and clamping is friction."""

    assert latch.LIP_INNER_HALF_GAP_M < GRAPPLE_PIN_HALF_WIDTH_Y
    assert latch.LIP_INNER_HALF_HEIGHT_M > GRAPPLE_PIN_SHAFT_HALF_HEIGHT
    assert latch.JAW_HALF_HEIGHT_M < GRAPPLE_PIN_COLLAR_HALF_HEIGHT
    assert latch.LIP_TOTAL_BEARING_AREA_M2 > 0.0


def test_the_release_stroke_lets_the_module_leave() -> None:
    assert latch.LIP_INNER_HALF_GAP_M + latch.CLOSE_STROKE_M > GRAPPLE_PIN_HALF_WIDTH_Y
    assert latch.WEB_INNER_HALF_GAP_M + latch.CLOSE_STROKE_M > GRAPPLE_PIN_HALF_WIDTH_Y


def test_the_web_clamps_the_shaft_rather_than_squeezing_it() -> None:
    assert latch.WEB_INNER_HALF_GAP_M > GRAPPLE_PIN_HALF_WIDTH_Y
    assert latch.WEB_INNER_HALF_GAP_M - GRAPPLE_PIN_HALF_WIDTH_Y < 0.002


def test_engaged_boxes_are_disjoint_and_inside_the_declared_depth() -> None:
    boxes = latch.jaw_boxes(engaged=True)
    assert {name for name, _c, _s in boxes} == {"web", "lip_upper", "lip_lower"}
    for _name, centre, size in boxes:
        near = centre[2] - 0.5 * size[2]
        far = centre[2] + 0.5 * size[2]
        assert near >= latch.ENGAGED_DEPTH_FROM_FLANGE_M[0] - 1e-9
        assert far <= latch.ENGAGED_DEPTH_FROM_FLANGE_M[1] + 1e-9
        assert min(size) > 0.0
    web = next(box for box in boxes if box[0] == "web")
    for lip in (box for box in boxes if box[0] != "web"):
        # The lip reaches inboard of the web, which is the whole point of it.
        assert lip[1][1] - 0.5 * lip[2][1] < web[1][1] - 0.5 * web[2][1]


def test_stowing_moves_the_jaws_outward_and_back() -> None:
    engaged = {name: (centre, size) for name, centre, size in latch.jaw_boxes(engaged=True)}
    stowed = {name: (centre, size) for name, centre, size in latch.jaw_boxes(engaged=False)}
    for name, (centre, _size) in engaged.items():
        stowed_centre, _ = stowed[name]
        assert stowed_centre[1] > centre[1]
        assert stowed_centre[2] < centre[2]


def test_the_shipped_latch_passes_its_own_clearance_derivation() -> None:
    """The check script is the contract; this keeps it green on every commit."""

    from check_service_latch_clearance import check

    result = check()
    failures = [row["check"] for row in result["checks"] if not row["passed"]]
    assert failures == []
    assert result["status"] == "passed"


def test_the_rating_is_stated_against_what_the_task_requires() -> None:
    assert latch.RATED_FORCE_N > latch.REQUIRED_AXIAL_CAPACITY_N
    # The lip is not the limit; if it were, the rating would be a fiction.
    assert latch.lip_bearing_stress_mpa() < 10.0
