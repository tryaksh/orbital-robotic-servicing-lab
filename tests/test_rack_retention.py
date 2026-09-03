import pytest

from zero_g_blade_swap import rack_retention
from zero_g_blade_swap.grapple_geometry import (
    BLADE_LENGTH_M,
    BLADE_WIDTH_M,
    SLOT_ENTRY_RAMP_CATCH_M,
)
from zero_g_blade_swap.service_latch import (
    RATED_FORCE_N as ROBOT_LATCH_RATED_FORCE_N,
)
from zero_g_blade_swap.service_latch import (
    RATED_TORQUE_NM as ROBOT_LATCH_RATED_TORQUE_NM,
)

SEATED = (0.67632, -0.22, 0.72)


def test_closed_pawls_are_clear_of_and_behind_the_module() -> None:
    boxes = rack_retention.pawl_tip_boxes(SEATED)
    rear_face_x = SEATED[0] - 0.5 * BLADE_LENGTH_M
    for _name, centre, size in boxes:
        assert centre[0] + 0.5 * size[0] == pytest.approx(
            rear_face_x - rack_retention.PAWL_FACE_CLEARANCE_M
        )
    left_inner = boxes[0][1][1] - 0.5 * boxes[0][2][1] - SEATED[1]
    right_inner = SEATED[1] - (boxes[1][1][1] + 0.5 * boxes[1][2][1])
    assert left_inner == pytest.approx(rack_retention.PAWL_CLOSED_INNER_HALF_GAP_M)
    assert right_inner == pytest.approx(rack_retention.PAWL_CLOSED_INNER_HALF_GAP_M)
    assert left_inner < 0.5 * BLADE_WIDTH_M


def test_open_pawls_clear_the_complete_lead_in_catch() -> None:
    opened_inner = (
        rack_retention.PAWL_CLOSED_INNER_HALF_GAP_M
        + rack_retention.pawl_translation(engaged=False, sign=1.0)[1]
    )
    assert opened_inner == pytest.approx(rack_retention.PAWL_OPEN_INNER_HALF_GAP_M)
    assert opened_inner == pytest.approx(0.5 * BLADE_WIDTH_M + SLOT_ENTRY_RAMP_CATCH_M)


def test_rack_accepts_the_rating_of_the_form_lock_that_hands_over_to_it() -> None:
    assert rack_retention.RATED_FORCE_N == ROBOT_LATCH_RATED_FORCE_N
    assert rack_retention.RATED_TORQUE_NM == ROBOT_LATCH_RATED_TORQUE_NM
    assert rack_retention.RATED_FORCE_N >= rack_retention.REQUIRED_AXIAL_CAPACITY_N
