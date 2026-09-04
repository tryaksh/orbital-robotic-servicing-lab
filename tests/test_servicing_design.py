"""The library and the certified check have to be the same arithmetic.

`servicing_design.py` exists so a spacecraft designer can compute a rack
requirement from *their* manipulator's measured numbers without running any of
this repository. That is only worth publishing if it is demonstrably the same
computation `scripts/check_workcell_geometry.py` performs for the shipped
workcell -- the one that validates itself against the simulator's recorded
configurations before it reports, and whose output the boundary validator binds
by hash.

So these tests drive the library with the shipped workcell's own inputs and
assert it lands on the shipped workcell's own numbers. If the two ever diverge,
one of them is wrong and the paper cannot cite either.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from zero_g_blade_swap.servicing_design import (
    ManipulatorPerformance,
    lateral_clearance_window,
    rack_requirement,
    rail_indexing_bound_m,
    section_verdict,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workcell():
    """Load the certified check as a module, without a simulator."""

    script = ROOT / "scripts" / "check_workcell_geometry.py"
    spec = importlib.util.spec_from_file_location("workcell_geometry_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["workcell_geometry_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shipped(workcell) -> ManipulatorPerformance:
    envelope = workcell.section_envelope()
    return ManipulatorPerformance(
        delivered_attitude_rad=float(envelope["delivered_attitude_rad"]),
        seating_tolerance_rad=float(workcell._literal("INSERTION_ORIENTATION_TOLERANCE_RAD", workcell.INSERTION)),
        pad_half_bearing_offset_m=float(envelope["pad_half_bearing_offset_m"]),
    )


def test_the_clearance_window_matches_the_certified_check(workcell, shipped) -> None:
    certified = workcell.lateral_clearance_window()
    library = lateral_clearance_window(shipped, float(workcell.BLADE_LENGTH_M))
    assert library["lower_bound_m"] == pytest.approx(float(certified["lower_bound_m"]), abs=1.0e-6)
    assert library["upper_bound_m"] == pytest.approx(float(certified["upper_bound_m"]), abs=1.0e-6)
    assert library["equal_margin_design_point_m"] == pytest.approx(float(certified["as_built_m"]), abs=1.0e-6)


def test_the_section_envelope_matches_the_certified_check(workcell, shipped) -> None:
    """Every cell of the certified grid, not just the accepted count."""

    certified = workcell.section_envelope()
    inner_face = float(certified["guide_inner_face_half_width_m"])
    channel_height = float(certified["channel_height_m"])
    relief = float(certified["destination_relief_per_side_m"])
    length = float(workcell.BLADE_LENGTH_M)

    checked = 0
    for row in certified["sections"]:
        verdict = section_verdict(
            shipped,
            module_length_m=length,
            module_width_m=float(row["width_m"]),
            module_height_m=float(row["height_m"]),
            channel_inner_face_half_width_m=inner_face,
            channel_height_m=channel_height,
            destination_relief_per_side_m=relief,
        )
        assert verdict.enters == bool(row["lead_ins_admit_the_delivered_attitude"]), row
        assert verdict.pads_can_follow == bool(row["pads_can_follow_the_corner"]), row
        assert verdict.accepted == bool(row["accepted"]), row
        assert verdict.lateral_half_gap_m == pytest.approx(float(row["lateral_half_gap_m"]), abs=1.0e-6)
        assert verdict.vertical_half_gap_m == pytest.approx(float(row["vertical_half_gap_m"]), abs=1.0e-6)
        checked += 1
    assert checked == int(certified["evaluated_count"])


def test_the_shipped_grip_margin_matches_the_certified_check(workcell, shipped) -> None:
    certified = workcell.section_envelope()
    requirement = rack_requirement(
        shipped,
        module_length_m=float(workcell.BLADE_LENGTH_M),
        module_width_m=float(workcell._literal("BLADE_SIZE")[1]),
        module_height_m=float(workcell._literal("BLADE_SIZE")[2]),
        channel_height_m=float(certified["channel_height_m"]),
        destination_relief_per_side_m=float(certified["destination_relief_per_side_m"]),
    )
    assert requirement.shipped_section.accepted
    assert requirement.shipped_section.grip_margin_m == pytest.approx(
        float(certified["grip_margin_of_the_shipped_section_m"]), abs=1.0e-6
    )
    # And the channel the library places from the manipulator's numbers alone is
    # the channel this workcell was built with.
    assert requirement.channel_inner_face_half_width_m == pytest.approx(
        float(certified["guide_inner_face_half_width_m"]), abs=1.0e-6
    )
    assert len(requirement.accepted_sections) == int(certified["accepted_count"])


def test_the_rail_bound_is_what_is_left_of_the_pads_reach(workcell, shipped) -> None:
    certified = workcell.section_envelope()
    lateral = float(certified["guide_inner_face_half_width_m"]) - 0.5 * float(workcell._literal("BLADE_SIZE")[1])
    vertical = 0.5 * (float(certified["channel_height_m"]) - float(workcell._literal("BLADE_SIZE")[2]))
    bound = rail_indexing_bound_m(shipped, lateral_half_gap_m=lateral, vertical_half_gap_m=vertical)
    expected = math.sqrt(shipped.pad_half_bearing_offset_m**2 - vertical**2) - lateral
    assert bound == pytest.approx(expected, abs=1.0e-9)
    # A millimetre-scale requirement, which is the point: it is far tighter than
    # a reader expects from an eleven-millimetre channel.
    assert 0.0 < bound < 0.005


def test_a_manipulator_that_cannot_meet_its_own_interface_is_rejected() -> None:
    with pytest.raises(ValueError):
        ManipulatorPerformance(
            delivered_attitude_rad=0.100,
            seating_tolerance_rad=0.052,
            pad_half_bearing_offset_m=0.015,
        )


def test_a_better_manipulator_earns_a_tighter_channel() -> None:
    """The direction of derivation, asserted: the arm sizes the rack."""

    coarse = ManipulatorPerformance(
        delivered_attitude_rad=0.046,
        seating_tolerance_rad=0.05236,
        pad_half_bearing_offset_m=0.015,
    )
    precise = ManipulatorPerformance(
        delivered_attitude_rad=0.010,
        seating_tolerance_rad=0.05236,
        pad_half_bearing_offset_m=0.015,
    )
    coarse_window = lateral_clearance_window(coarse, 0.450)
    precise_window = lateral_clearance_window(precise, 0.450)
    assert precise_window["lower_bound_m"] < coarse_window["lower_bound_m"]
    assert precise_window["window_width_m"] > coarse_window["window_width_m"]

    # And a wider window is a larger family of admissible modules, which is the
    # sentence a designer actually wants out of this.
    common = dict(
        module_length_m=0.450,
        module_width_m=0.130,
        module_height_m=0.020,
        channel_height_m=0.036,
    )
    assert len(rack_requirement(precise, **common).accepted_sections) >= len(
        rack_requirement(coarse, **common).accepted_sections
    )


def test_the_wedge_law_is_the_prior_art_form() -> None:
    from zero_g_blade_swap.servicing_design import engagement_depth_limit_m

    assert engagement_depth_limit_m(0.046, 0.011065) == pytest.approx(2 * 0.011065 / 0.046, rel=1e-12)
    # A square module is not bounded, and a zero-clearance channel admits nothing.
    assert engagement_depth_limit_m(0.0, 0.011) == math.inf
    assert engagement_depth_limit_m(0.046, 0.0) == 0.0


def test_this_bay_is_predicted_to_need_a_correcting_lead_in_and_has_one() -> None:
    """The rule fires on the shipped workcell, before any simulation.

    At the hand-over attitude an 11.065 mm channel admits 481 mm of a 529 mm
    stroke, so a module pushed straight home would wedge 48 mm short. The bay
    carries a 12-degree entry flare, and the corrected clearance sweep seats
    36 of 64 at 6 mm per side where the shortfall is 268 mm -- so the correction
    is real and a designer may not size the channel as though the module carried
    its hand-over attitude to the seated plane.
    """

    from zero_g_blade_swap.servicing_design import requires_a_correcting_lead_in

    performance = ManipulatorPerformance(
        delivered_attitude_rad=0.046,
        seating_tolerance_rad=0.0523599,
        pad_half_bearing_offset_m=0.015,
    )
    verdict = requires_a_correcting_lead_in(
        performance, seating_stroke_m=0.529, clearance_per_side_m=0.011065
    )
    assert verdict["required"] is True
    assert verdict["engagement_depth_limit_m"] == pytest.approx(0.4811, abs=5.0e-4)
    assert verdict["shortfall_m"] == pytest.approx(0.0479, abs=5.0e-4)
    # And the attitude at which no lead-in would be needed, which is the number a
    # designer trades against the arm.
    assert verdict["attitude_that_would_not_need_one_rad"] == pytest.approx(0.0418, abs=5.0e-4)

    # Tighter clearance makes the requirement stronger, not weaker: the sweep's
    # 6 mm point falls 268 mm short and still seats, in the bay that has a flare.
    tight = requires_a_correcting_lead_in(performance, seating_stroke_m=0.529, clearance_per_side_m=0.006)
    assert tight["required"] is True
    assert tight["shortfall_m"] > verdict["shortfall_m"]


def test_the_admissible_attitude_inverts_the_entry_bound(shipped) -> None:
    """A section admitted at exactly its admissible attitude has zero entry margin.

    The inverse is the number a designer hands a controls team when the section
    cannot change: not "this fails by 3.92 mm of gap" but "the arm must deliver
    at or below this angle". The two have to be the same statement, so the test
    is that evaluating at the returned attitude lands exactly on the boundary.
    """

    from dataclasses import replace

    # 140x26, whose inverse is 22.72 mrad and so below the seating tolerance.
    # The 120x16 section returns 64.94 mrad, above the 52.36 mrad tolerance, and
    # ManipulatorPerformance rightly refuses to be built at it -- which is the
    # documented case where entry is not the binding constraint.
    verdict = section_verdict(
        shipped,
        module_length_m=0.450,
        module_width_m=0.140,
        module_height_m=0.026,
        channel_inner_face_half_width_m=0.0705,
        channel_height_m=0.036,
        destination_relief_per_side_m=0.0046125,
    )

    at_the_limit = replace(shipped, delivered_attitude_rad=verdict.admissible_delivered_attitude_rad)
    boundary = section_verdict(
        at_the_limit,
        module_length_m=0.450,
        module_width_m=0.140,
        module_height_m=0.026,
        channel_inner_face_half_width_m=0.0705,
        channel_height_m=0.036,
        destination_relief_per_side_m=0.0046125,
    )
    assert boundary.entry_margin_m == pytest.approx(0.0, abs=1.0e-12)
    assert boundary.enters


def test_headroom_and_margin_agree_on_the_verdict(shipped) -> None:
    """Positive headroom and positive entry margin must never disagree.

    They are the same bound in different units -- radians of attitude and metres
    of gap -- so a section that enters must have headroom and one that does not
    must not. A sign disagreement would mean the inverse is not the inverse.
    """

    for width, height in ((0.120, 0.016), (0.130, 0.020), (0.140, 0.026)):
        verdict = section_verdict(
            shipped,
            module_length_m=0.450,
            module_width_m=width,
            module_height_m=height,
            channel_inner_face_half_width_m=0.0705,
            channel_height_m=0.036,
            destination_relief_per_side_m=0.0046125,
        )
        assert verdict.enters == (verdict.attitude_headroom_rad >= -1.0e-12), (width, height)


def test_the_inverse_can_exceed_the_seating_tolerance_and_says_so(shipped) -> None:
    """Entry is not always the binding constraint, and the tool must not pretend it is.

    For the shipped bay's 120x16 section the entry bound admits 64.94 mrad while
    the interface accepts 52.36. Handing 64.94 to a controls team as a
    requirement would specify an arm the interface still rejects. The value is
    correct for what it names -- the entry bound alone -- and the caller takes
    the minimum.
    """

    verdict = section_verdict(
        shipped,
        module_length_m=0.450,
        module_width_m=0.120,
        module_height_m=0.016,
        channel_inner_face_half_width_m=0.0705,
        channel_height_m=0.036,
        destination_relief_per_side_m=0.0046125,
    )
    assert verdict.enters
    assert verdict.admissible_delivered_attitude_rad > shipped.seating_tolerance_rad
    usable = min(verdict.admissible_delivered_attitude_rad, shipped.seating_tolerance_rad)
    assert usable == pytest.approx(shipped.seating_tolerance_rad)
