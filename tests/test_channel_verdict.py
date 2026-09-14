"""The measured rack, pinned: it holds a module outside the gate it judges by.

`channel_verdict` is the entry point a hardware engineer runs before anyone cuts
metal, and the thing that makes it worth running is that this repository's own bay
fails it. So the bay is pinned here, at both of its clearances, against the
report that measured it -- if the scene moves, or the criterion moves, or the law
is implemented differently, this test says so.

CPU only. No simulator, no policy, no checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_g_blade_swap.servicing_design import channel_verdict

ROOT = Path(__file__).resolve().parents[1]
MEASURED = ROOT / "evidence" / "destination_channel_geometry.json"

#: The destination bay as the chain runs it: the scene's own guides, relieved by
#: `--destination_channel_relief_m 0.0046125` on every chain invocation.
SHIPPED = {
    "module_length_m": 0.45,
    "module_width_m": 0.13,
    "module_height_m": 0.02,
    # `GUIDE_CENTER_OFFSET_Y` 0.085065 places the guide BODY centre; the face the
    # module runs against is half the guide's 18 mm thickness inboard of it.
    # Reading the centre as the face is the mistake that turns a 0.75 mm channel
    # into a 9.75 mm one.
    "channel_inner_face_half_width_m": 0.076065,
    "channel_height_m": 0.036,
    "seating_stroke_m": 0.529,
    "seating_tolerance_rad": 0.0523599,
}
RELIEF_M = 0.0046125


def _measured_arm() -> dict:
    """The chain's arm out of the report that measured the bay in the simulator."""

    report = json.loads(MEASURED.read_text(encoding="utf-8"))
    arms = [arm for arm in report["arms"] if arm["arm"] == "chain"]
    assert arms, "destination_channel_geometry.json no longer records a chain arm"
    return arms[0]


def test_the_shipped_bay_holds_the_module_outside_its_own_criterion() -> None:
    """The finding, as one assertion: the bay demands what its geometry forbids.

    Both axes exceed the acceptance tolerance, and the lateral offset exceeds it by
    more than six times. A controller cannot close either, because the module is
    already at rest when the gate is read.
    """

    verdict = channel_verdict(**SHIPPED, relief_per_side_m=RELIEF_M)

    # Exact, to the micrometre: 0.076065 - 0.065 + 0.0046125 laterally, and
    # 0.5 * (0.036 - 0.020) + 0.0046125 vertically.
    assert verdict.lateral_clearance_per_side_m == pytest.approx(0.0156775, abs=1e-9)
    assert verdict.vertical_clearance_per_side_m == pytest.approx(0.0126125, abs=1e-9)
    assert verdict.resting_yaw_rad == pytest.approx(2.0 * 0.0156775 / 0.45, abs=1e-9)
    assert verdict.resting_pitch_rad == pytest.approx(2.0 * 0.0126125 / 0.45, abs=1e-9)

    # The demand is the LOOSER axis. A resting module may take either attitude, so
    # a gate that only accepts the tighter one rejects a module the channel is
    # entitled to leave it in.
    assert verdict.attitude_the_gate_must_accept_rad == pytest.approx(0.069678, abs=5e-6)
    assert verdict.limiting_axis == "lateral"

    assert verdict.seating_tolerance_rad == pytest.approx(0.0523599)
    assert verdict.attitude_margin_rad == pytest.approx(0.0523599 - 0.069678, abs=5e-6)
    assert not verdict.attitude_is_compatible
    assert not verdict.offset_is_compatible
    assert not verdict.compatible


def test_the_library_and_the_simulator_measurement_agree_on_the_bay() -> None:
    """Bind the closed form to the report that measured the same bay in the scene.

    The report is `evidence/destination_channel_geometry.json`, generated from the
    scene the chain builds. Two independent routes to the same four numbers, which
    is the standard every geometric requirement here is held to.
    """

    arm = _measured_arm()
    verdict = channel_verdict(**SHIPPED, relief_per_side_m=RELIEF_M)

    # The report stores six decimals, so agreement is asserted at its own
    # resolution rather than tighter than it can express. Half a micrometre would
    # be asserting against a rounding artefact.
    assert verdict.lateral_clearance_per_side_m == pytest.approx(
        arm["lateral_clearance_per_side_m"], abs=1e-6
    )
    assert verdict.vertical_clearance_per_side_m == pytest.approx(
        arm["vertical_clearance_per_side_m"], abs=1e-6
    )
    assert verdict.resting_yaw_rad == pytest.approx(arm["resting_yaw_rad"], abs=1e-6)
    assert verdict.resting_pitch_rad == pytest.approx(arm["resting_pitch_rad"], abs=1e-6)
    # The report reaches the same verdict field by field; neither axis is inside.
    assert arm["yaw_inside_the_success_criterion"] is False
    assert arm["pitch_inside_the_success_criterion"] is False
    assert not verdict.compatible


def test_the_design_point_fixes_the_attitude_and_not_the_offset() -> None:
    """Removing the relief is not enough, and that is the useful half.

    At the 11.065 mm design point the attitude comes inside the tolerance with
    3.18 mrad in hand -- which is `lateral_clearance_window`'s upper bound being
    respected. The lateral *offset* does not: 11.065 mm against a 2.5 mm gate. So a
    plain channel cannot guarantee seating at any clearance that also admits the
    module, which is what `interface_regime` returns as `active_centring`, in
    millimetres instead of regimes.
    """

    verdict = channel_verdict(**SHIPPED, relief_per_side_m=0.0)

    assert verdict.lateral_clearance_per_side_m == pytest.approx(0.011065, abs=5e-7)
    assert verdict.resting_yaw_rad == pytest.approx(0.049178, abs=5e-6)
    assert verdict.attitude_is_compatible
    assert verdict.attitude_margin_rad == pytest.approx(0.003182, abs=5e-6)
    assert not verdict.offset_is_compatible
    assert verdict.offset_margin_m == pytest.approx(0.0025 - 0.011065, abs=5e-7)
    assert not verdict.compatible


def test_a_channel_tight_enough_to_hold_the_gate_will_not_admit_the_module() -> None:
    """The trade, both sides of it, so neither can be quoted without the other.

    Narrow the channel until a resting module is inside the 2.5 mm gate and the
    entry bound closes: at 0.5 mm per side the module reaches 21.7 mm of a 529 mm
    stroke before it wedges. This is `no passive rack satisfies this interface`
    stated as two runs of one tool.
    """

    tight = channel_verdict(
        **{**SHIPPED, "channel_inner_face_half_width_m": 0.0655, "channel_height_m": 0.021},
        relief_per_side_m=0.0,
        delivered_attitude_rad=0.046,
        pad_half_bearing_offset_m=0.015,
    )

    assert tight.lateral_clearance_per_side_m == pytest.approx(0.0005, abs=5e-7)
    assert tight.compatible
    assert tight.correcting_lead_in_required is True
    assert tight.engagement_depth_limit_m == pytest.approx(0.0217, abs=5e-4)
    assert tight.regime == "active_centring_and_correction"


def test_the_entry_half_needs_a_real_pad_offset_rather_than_a_stand_in() -> None:
    """A delivered attitude without the gripper number is refused, not guessed."""

    with pytest.raises(ValueError, match="pad_half_bearing_offset_m is required"):
        channel_verdict(**SHIPPED, relief_per_side_m=0.0, delivered_attitude_rad=0.046)


def test_the_channel_question_answers_without_any_manipulator() -> None:
    """Whether a channel can hold its module inside its gate needs no arm at all.

    That is the whole reason this is separate from `interface_regime`: the answer
    is a property of the channel and the module, so it is available while the slot
    is still a drawing and nobody has chosen a robot.
    """

    verdict = channel_verdict(**SHIPPED, relief_per_side_m=RELIEF_M)

    assert verdict.delivered_attitude_rad is None
    assert verdict.regime is None
    assert verdict.correcting_lead_in_required is None
    assert verdict.compatible is False
