"""The three regimes a measured manipulator forces on its interface.

The result these defend is that a design tool can say, before anything is built,
whether a passive channel is a solution at all. That is a strong claim, so the
cases where it must say *no* matter more than the cases where it says yes.
"""

from __future__ import annotations

import math

import pytest

from zero_g_blade_swap.servicing_design import (
    DEFAULT_LATERAL_SEATING_TOLERANCE_M,
    ManipulatorPerformance,
    interface_regime,
    lateral_clearance_window,
    passive_alignment_limit_rad,
)

MODULE_LENGTH_M = 0.450
STROKE_M = 0.529
UNRELIEVED_CLEARANCE_M = 0.011065


@pytest.fixture
def shipped() -> ManipulatorPerformance:
    """The manipulator this repository measured, with the shipped pad offset."""

    return ManipulatorPerformance(
        delivered_attitude_rad=0.046,
        seating_tolerance_rad=0.05236,
        pad_half_bearing_offset_m=0.015,
    )


def test_the_limit_is_where_the_two_requirements_meet() -> None:
    """Below it the corner sweep fits inside the gate; above it, it does not."""

    limit = passive_alignment_limit_rad(MODULE_LENGTH_M, DEFAULT_LATERAL_SEATING_TOLERANCE_M)
    assert limit == pytest.approx(2 * DEFAULT_LATERAL_SEATING_TOLERANCE_M / MODULE_LENGTH_M)
    # At exactly the limit the clearance a module needs equals what the gate allows.
    needed = 0.5 * limit * MODULE_LENGTH_M
    assert needed == pytest.approx(DEFAULT_LATERAL_SEATING_TOLERANCE_M)


def test_a_longer_module_is_harder(shipped: ManipulatorPerformance) -> None:
    """Length is the lever: the same arm serving a longer module loses passivity sooner."""

    short = passive_alignment_limit_rad(0.200, DEFAULT_LATERAL_SEATING_TOLERANCE_M)
    long = passive_alignment_limit_rad(0.900, DEFAULT_LATERAL_SEATING_TOLERANCE_M)
    assert short > long


def test_non_positive_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="module_length_m"):
        passive_alignment_limit_rad(0.0, DEFAULT_LATERAL_SEATING_TOLERANCE_M)
    with pytest.raises(ValueError, match="lateral_seating_tolerance_m"):
        passive_alignment_limit_rad(MODULE_LENGTH_M, 0.0)


def test_this_manipulator_needs_both_centring_and_correction(shipped: ManipulatorPerformance) -> None:
    """The measured system lands in the regime whose two requirements it implements.

    It has a guarded advance that centres and entry flares that square, and until
    the thresholds were written down nothing said why both were necessary.
    """

    regime = interface_regime(
        shipped,
        module_length_m=MODULE_LENGTH_M,
        seating_stroke_m=STROKE_M,
        clearance_per_side_m=UNRELIEVED_CLEARANCE_M,
    )
    assert regime["regime"] == "active_centring_and_correction"
    assert regime["passive_alignment_possible"] is False
    assert regime["correcting_lead_in_required"] is True
    assert regime["alignment_shortfall_m"] == pytest.approx(0.00785, abs=1.0e-5)


def test_a_gentle_arm_may_use_a_plain_channel() -> None:
    """Below the limit the tool must not demand machinery that is not needed."""

    gentle = ManipulatorPerformance(
        delivered_attitude_rad=0.005, seating_tolerance_rad=0.05236, pad_half_bearing_offset_m=0.015
    )
    regime = interface_regime(
        gentle,
        module_length_m=MODULE_LENGTH_M,
        seating_stroke_m=STROKE_M,
        clearance_per_side_m=UNRELIEVED_CLEARANCE_M,
    )
    assert regime["regime"] == "passive"
    assert regime["passive_alignment_possible"] is True
    assert regime["alignment_shortfall_m"] == 0.0


def test_the_middle_regime_exists(shipped: ManipulatorPerformance) -> None:
    """Centring without correction is a real state, not an unreachable label.

    An arm past the alignment limit whose engagement still reaches the seated
    plane needs a controller and no lead-in. Without this the three regimes would
    be two.
    """

    middle = ManipulatorPerformance(
        delivered_attitude_rad=0.020, seating_tolerance_rad=0.05236, pad_half_bearing_offset_m=0.015
    )
    regime = interface_regime(
        middle,
        module_length_m=MODULE_LENGTH_M,
        seating_stroke_m=STROKE_M,
        clearance_per_side_m=UNRELIEVED_CLEARANCE_M,
    )
    assert regime["regime"] == "active_centring"
    assert regime["passive_alignment_possible"] is False
    assert regime["correcting_lead_in_required"] is False


def test_the_lateral_limit_binds_before_the_orientation_one(shipped: ManipulatorPerformance) -> None:
    """The window's upper bound is not the binding constraint, and that is the point.

    `lateral_clearance_window` bounds the clearance above using the *orientation*
    tolerance. The lateral tolerance is four times tighter and the window never
    consults it, which is how a caller can read a feasible window for an
    interface that has no passive solution.
    """

    window = lateral_clearance_window(shipped, MODULE_LENGTH_M)
    assert window["lower_bound_m"] > DEFAULT_LATERAL_SEATING_TOLERANCE_M, (
        "the clearance needed to admit this arm already exceeds the lateral gate"
    )
    assert window["upper_bound_m"] > window["lower_bound_m"] > DEFAULT_LATERAL_SEATING_TOLERANCE_M


def test_the_limit_does_not_depend_on_the_arm() -> None:
    """It is a property of the module and the gate, which is what makes it a specification."""

    assert not math.isnan(passive_alignment_limit_rad(MODULE_LENGTH_M, DEFAULT_LATERAL_SEATING_TOLERANCE_M))
    for attitude in (0.001, 0.046, 0.052):
        performance = ManipulatorPerformance(
            delivered_attitude_rad=attitude, seating_tolerance_rad=0.05236, pad_half_bearing_offset_m=0.015
        )
        regime = interface_regime(
            performance,
            module_length_m=MODULE_LENGTH_M,
            seating_stroke_m=STROKE_M,
            clearance_per_side_m=UNRELIEVED_CLEARANCE_M,
        )
        assert regime["passive_alignment_limit_rad"] == pytest.approx(
            passive_alignment_limit_rad(MODULE_LENGTH_M, DEFAULT_LATERAL_SEATING_TOLERANCE_M)
        )
