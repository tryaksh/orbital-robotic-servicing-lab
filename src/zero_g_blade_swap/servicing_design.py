"""The rack requirement for a measured manipulator, in closed form, on the CPU.

**This is the project's actual output, packaged so it can be used without any of
the rest of it.** The claim the work is built on is that given a manipulator
whose performance has been measured, the rack requirement for autonomous
servicing can be computed before a policy is trained -- and the four `check_*`
scripts in this repository are that computation, each answering one question and
validating itself against the simulator's recorded configurations before it
reports. What they did not have was a surface a spacecraft designer could call
with *their* manipulator's numbers instead of this one's.

That surface is here. Nothing in this module reads an asset, a checkpoint, a
report or a simulator: it takes four measured quantities and returns the
requirement they imply.

    from zero_g_blade_swap.servicing_design import ManipulatorPerformance, rack_requirement

    requirement = rack_requirement(
        ManipulatorPerformance(
            delivered_attitude_rad=0.046,      # what transit hands the insertion over at
            seating_tolerance_rad=0.05236,     # what a seated module is accepted at
            pad_half_bearing_offset_m=0.015,   # how far off-centre a pad still keeps half its face
        ),
        module_length_m=0.450,
        module_width_m=0.130,
        module_height_m=0.020,
        channel_height_m=0.036,
    )

`tests/test_servicing_design.py` asserts that this reproduces the numbers
`scripts/check_workcell_geometry.py` derives for the shipped workcell, to the
last decimal. The library and the certified check are the same arithmetic, and
that test is what keeps them so.

## The four requirements, and where each comes from

**Lateral clearance is bounded on both sides, and that is the non-obvious part.**
A channel too tight refuses a module arriving at the attitude the manipulator
actually delivers; a channel too loose lets a resting module sit at an attitude
the seating criterion rejects. Both bounds are `half the module length times an
angle`, and the design point that maximises the smaller margin is the midpoint in
*attitude* rather than in clearance.

**The pads set a second, independent bound.** A module in the corner of a
channel with no relief is offset from the pin by `hypot(lateral, vertical)`, and
a parallel-jaw pad that has slid more than `pad_half_bearing_offset_m` off the
pin is no longer bearing on it. This is the criterion that governs the *source*
bay, where the pull happens, and it is the one that rejects sections a designer
would otherwise think generous: making a module narrower makes its channel
looser, and a looser channel is a corner the pads cannot follow.

**The section envelope is the intersection of the two**, evaluated over a grid.

**The rail indexing requirement follows from the pad bound** with the stop error
added to the lateral offset the pads must absorb. It is a *geometric* bound: it
says how far the base may be parked from nominal before a module in the channel
corner is outside the pads' reach. It is not a statement about what a policy
trained at one base position tolerates, and this repository's own sweep shows
those are different numbers -- see `docs/sim_to_real.md`.

Every quantity is a length in metres or an angle in radians. Nothing is tuned.

## What this library does not bound, and it matters

**Every criterion here is static.** Each asks whether a module *placed* somewhere
is admissible; none asks what the manipulator's motion leaves behind. That is a
real gap and this repository has measured it: the rail stop-error axis fails at a
threshold three times looser than the pad bound predicts, and the failing
episodes are extracted and still gripped at a normal offset. What they fail is a
*settling* condition -- they carry 16 to 30 mm/s against a 14.29 mm/s limit that
is itself derived, from the capture tolerance over the settling window.

So a designer taking `rail_indexing_bound_m` gets a safe number for the wrong
reason. Bounding that axis properly needs the residual velocity an off-axis pull
imparts, which is a dynamic quantity and is not computed here. `docs/NOW.md`
carries the measurement and `docs/paper_position.md` carries what it means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ManipulatorPerformance",
    "engagement_depth_limit_m",
    "requires_a_correcting_lead_in",
    "RackRequirement",
    "SectionVerdict",
    "lateral_clearance_window",
    "rack_requirement",
    "rail_indexing_bound_m",
    "section_verdict",
]


def engagement_depth_limit_m(attitude_rad: float, clearance_per_side_m: float) -> float:
    """Return ``2c/theta``: how deep a module held at this attitude may go.

    Whitney's quasi-static wedging geometry, applied to a long flat module in a
    rectangular channel. It is prior art and this library cites it rather than
    claiming it; what is being published is its use as a *design gate* on the
    bay, before either the bay or the controller exists.
    """

    if attitude_rad <= 0.0:
        return math.inf
    if clearance_per_side_m <= 0.0:
        return 0.0
    return 2.0 * clearance_per_side_m / attitude_rad


def requires_a_correcting_lead_in(
    performance: ManipulatorPerformance,
    *,
    seating_stroke_m: float,
    clearance_per_side_m: float,
) -> dict[str, float | bool]:
    """Decide whether the bay has to square the module on its way in.

    A module pushed straight home at the attitude the manipulator hands over
    reaches ``2c/theta`` and wedges. If that is shorter than the seating stroke,
    the bay cannot be a plain channel: something -- a funnel, or a controller
    that refuses to advance while the module is cocked -- has to reduce the
    attitude *during* the stroke.

    **Sizing a lead-in from an initial misalignment is chamfer-crossing analysis
    and is classical.** This is that analysis used as a gate on a manipulator
    whose delivered attitude was *measured* rather than specified, which is the
    only part of it this project claims. See ``docs/paper_position.md``.

    **This repository is the case that motivates the rule.** Its stroke is
    529 mm and its hand-over attitude 46 mrad, so an 11.065 mm channel admits
    481 mm and falls short. The chain seats anyway, and it seats at 6 mm of
    clearance too, where the shortfall is far larger -- which is the measurement
    that says the correction is real and that a designer may not size the channel
    as though the module carried its hand-over attitude to the seated plane.
    """

    limit = engagement_depth_limit_m(performance.delivered_attitude_rad, clearance_per_side_m)
    return {
        "seating_stroke_m": seating_stroke_m,
        "clearance_per_side_m": clearance_per_side_m,
        "engagement_depth_limit_m": limit,
        "shortfall_m": max(0.0, seating_stroke_m - limit),
        "required": bool(limit < seating_stroke_m),
        "attitude_that_would_not_need_one_rad": (
            2.0 * clearance_per_side_m / seating_stroke_m if seating_stroke_m > 0.0 else math.inf
        ),
    }


@dataclass(frozen=True)
class ManipulatorPerformance:
    """The three numbers a rack requirement follows from.

    Two of them are design inputs and one is a measurement, and the difference
    decides what may be claimed about ordering. ``seating_tolerance_rad`` is a
    property of the interface and ``pad_half_bearing_offset_m`` is read off the
    gripper's collision geometry, so both are known before anything is trained.
    ``delivered_attitude_rad`` is not: it is what the arm actually achieves, and
    something has to move the module before it can be measured. **A rack
    requirement is therefore computable before the seating policy is trained,
    not before any policy is trained.** The second claim is circular and this
    class is where it would start.

    ``delivered_attitude_rad`` is what the manipulator actually hands the
    insertion over at, not what it is specified to. In this repository it is
    reported as ``handoff_attitude_rad`` in every robot-carried run.

    ``seating_tolerance_rad`` is the acceptance criterion for a seated module --
    a property of the interface, not of the arm -- and it enters because a
    channel loose enough to let a resting module exceed it has moved the
    acceptance problem into the rack.

    ``pad_half_bearing_offset_m`` is the offset at which a gripper pad still
    keeps half its face on the capture feature. Measure it from the gripper's
    collision geometry, not from body origins; reading body origins as pad
    locations produced a retracted claim in this project once already.
    """

    delivered_attitude_rad: float
    seating_tolerance_rad: float
    pad_half_bearing_offset_m: float

    def __post_init__(self) -> None:
        if not self.delivered_attitude_rad > 0.0:
            raise ValueError("delivered_attitude_rad must be positive")
        if not self.seating_tolerance_rad > 0.0:
            raise ValueError("seating_tolerance_rad must be positive")
        if not self.pad_half_bearing_offset_m > 0.0:
            raise ValueError("pad_half_bearing_offset_m must be positive")
        if self.seating_tolerance_rad < self.delivered_attitude_rad:
            raise ValueError(
                "seating_tolerance_rad is below delivered_attitude_rad: the manipulator hands over "
                "at an attitude the interface does not accept, and no channel clearance can fix that"
            )


def lateral_clearance_window(
    performance: ManipulatorPerformance,
    module_length_m: float,
) -> dict[str, float]:
    """Return the two-sided lateral clearance bound and the equal-margin point.

    The lower bound is what the channel has to admit: a module arriving at the
    delivered attitude sweeps ``0.5 * theta * L`` at its corners. The upper bound
    is what a resting module may not exceed: a channel that lets it lie at more
    than the seating tolerance has made the rack responsible for an acceptance
    the interface already specifies.
    """

    if module_length_m <= 0.0:
        raise ValueError("module_length_m must be positive")
    half_length = 0.5 * module_length_m
    lower = half_length * performance.delivered_attitude_rad
    upper = half_length * performance.seating_tolerance_rad
    # The design point is the midpoint in *attitude*, not in clearance: that is
    # what maximises the smaller of the two margins, and sitting on either bound
    # is the defect this replaces.
    derived = half_length * 0.5 * (performance.delivered_attitude_rad + performance.seating_tolerance_rad)
    return {
        "lower_bound_m": lower,
        "upper_bound_m": upper,
        "equal_margin_design_point_m": derived,
        "window_width_m": upper - lower,
    }


@dataclass(frozen=True)
class SectionVerdict:
    """Whether one module cross-section is admissible, and by how much."""

    width_m: float
    height_m: float
    #: The delivered attitude this verdict was computed at, kept so the verdict
    #: can state its own headroom without the caller re-supplying the input.
    delivered_attitude_rad: float
    lateral_half_gap_m: float
    vertical_half_gap_m: float
    channel_corner_m: float
    entry_margin_m: float
    grip_margin_m: float
    #: The largest delivered attitude at which this section would still enter,
    #: in radians. A property of the section and the channel, not of what the arm
    #: currently delivers, which is what makes it the number to hand a controls
    #: team. **It is the entry bound alone and is not a specification by
    #: itself**: it can exceed ``seating_tolerance_rad`` -- for the 120x16
    #: section in the shipped bay it returns 64.94 mrad against a 52.36 mrad
    #: tolerance -- and where it does, entry is not the binding constraint and
    #: the usable requirement is the smaller of the two.
    admissible_delivered_attitude_rad: float

    @property
    def enters(self) -> bool:
        return self.entry_margin_m >= -1.0e-9

    @property
    def pads_can_follow(self) -> bool:
        return self.grip_margin_m >= -1.0e-9

    @property
    def accepted(self) -> bool:
        return self.enters and self.pads_can_follow

    @property
    def attitude_headroom_rad(self) -> float:
        """How much delivered attitude the entry bound still has in hand.

        Negative means the section is inadmissible on entry and by how much in
        the quantity a controls team can act on, which ``entry_margin_m`` states
        in millimetres of gap instead. Same fact, the other unit.
        """

        return self.admissible_delivered_attitude_rad - self.delivered_attitude_rad

    @property
    def limiting_criterion(self) -> str | None:
        """Which bound rejects this section, which is what a designer changes."""

        if self.accepted:
            return None
        if not self.enters and not self.pads_can_follow:
            return "entry_and_grip"
        return "entry" if not self.enters else "grip"


def section_verdict(
    performance: ManipulatorPerformance,
    *,
    module_length_m: float,
    module_width_m: float,
    module_height_m: float,
    channel_inner_face_half_width_m: float,
    channel_height_m: float,
    destination_relief_per_side_m: float = 0.0,
) -> SectionVerdict:
    """Apply both closed-form bounds to one module cross-section.

    ``channel_inner_face_half_width_m`` is the face the module runs against, not
    the guide body's centre. Reading the centre as the face is the mistake that
    turns a 0.75 mm channel into a 9.75 mm one.
    """

    lateral = channel_inner_face_half_width_m - 0.5 * module_width_m
    vertical = 0.5 * (channel_height_m - module_height_m)
    needed = 0.5 * performance.delivered_attitude_rad * module_length_m
    corner = math.hypot(lateral, vertical) if min(lateral, vertical) > 0.0 else math.inf
    # The inverse of the entry bound. ``needed = 0.5 * theta * L`` must not
    # exceed the admitting gap, so the largest attitude that still enters is
    # ``2 * gap / L``. Stating it turns the tool from a checker into a
    # specification: a designer who cannot change the section can read off what
    # the arm would have to deliver instead.
    admitting_gap = min(lateral, vertical) + destination_relief_per_side_m
    admissible_attitude = 2.0 * admitting_gap / module_length_m if module_length_m > 0.0 else math.inf
    return SectionVerdict(
        width_m=module_width_m,
        height_m=module_height_m,
        delivered_attitude_rad=performance.delivered_attitude_rad,
        lateral_half_gap_m=lateral,
        vertical_half_gap_m=vertical,
        channel_corner_m=corner,
        entry_margin_m=admitting_gap - needed,
        grip_margin_m=performance.pad_half_bearing_offset_m - corner,
        admissible_delivered_attitude_rad=admissible_attitude,
    )


def rail_indexing_bound_m(
    performance: ManipulatorPerformance,
    *,
    lateral_half_gap_m: float,
    vertical_half_gap_m: float,
) -> float:
    """Return how far the parked base may sit from nominal, geometrically.

    A stop error adds directly to the lateral offset a pad has to absorb, so the
    bound is what is left of the pad's reach once the channel's own corner is
    accounted for. A negative result means the section is already outside the
    pads' reach with the base exactly on nominal.

    **This is a geometric bound and not a policy tolerance.** A policy trained at
    a single base position can fail well inside it, and in this repository it
    does; `docs/sim_to_real.md` states that separation, and a designer should
    take this number rather than a swept one.
    """

    reach = performance.pad_half_bearing_offset_m**2 - vertical_half_gap_m**2
    if reach <= 0.0:
        return -math.inf
    return math.sqrt(reach) - lateral_half_gap_m


#: The lateral offset a seated module may carry, from the insertion criterion.
#: Imported as a number rather than from the task package, which needs Isaac to
#: import; the test asserts the two agree.
DEFAULT_LATERAL_SEATING_TOLERANCE_M = 0.0025


def passive_alignment_limit_rad(module_length_m: float, lateral_seating_tolerance_m: float) -> float:
    """The delivered attitude above which no plain channel can guarantee seating.

    A channel must be at least ``(L/2) * theta`` per side to admit a module of
    length ``L`` arriving at attitude ``theta`` without squaring it. A module
    resting in that channel may sit anywhere up to the clearance off the centre
    line, and the seating gate accepts only ``t_lat``. The two requirements meet
    at

        theta = 2 * t_lat / L

    and above it there is no clearance that satisfies both: widen the channel to
    let the module in and it can rest outside the gate; narrow it to hold the
    gate and the module will not enter. **Something other than the channel has to
    close the difference.**

    **This bites harder in orbit than on a bench.** On the ground a released part
    is centred for free -- gravity plus a chamfer takes up the slack -- so the
    clearance is an upper bound something else closes. In zero gravity a module
    released at an offset stays there. The clearance is the whole error budget.
    """

    if module_length_m <= 0.0:
        raise ValueError("module_length_m must be positive")
    if lateral_seating_tolerance_m <= 0.0:
        raise ValueError("lateral_seating_tolerance_m must be positive")
    return 2.0 * lateral_seating_tolerance_m / module_length_m


def interface_regime(
    performance: ManipulatorPerformance,
    *,
    module_length_m: float,
    seating_stroke_m: float,
    clearance_per_side_m: float,
    lateral_seating_tolerance_m: float = DEFAULT_LATERAL_SEATING_TOLERANCE_M,
) -> dict[str, object]:
    """Which class of interface a measured manipulator forces, before anything is built.

    Three regimes, in order of what they cost a designer:

    ``passive``
        Below ``2 * t_lat / L``. A plain channel works: pick a clearance in the
        window and geometry alone guarantees seating.

    ``active_centring``
        Above that limit. The channel admits the module but cannot guarantee it
        is within the lateral gate, and **no choice of clearance fixes it**. A
        controller has to centre the module.

    ``active_centring_and_correction``
        Also above ``2c/stroke``, so the module wedges before reaching the seated
        plane. The bay must additionally square it during the stroke -- a lead-in,
        or a controller that refuses to advance while cocked.

    The regime is the useful output. A clearance window says how wide to cut a
    slot; the regime says whether cutting a slot is a solution at all.
    """

    alignment_limit = passive_alignment_limit_rad(module_length_m, lateral_seating_tolerance_m)
    lead_in = requires_a_correcting_lead_in(
        performance, seating_stroke_m=seating_stroke_m, clearance_per_side_m=clearance_per_side_m
    )
    passive_alignment = performance.delivered_attitude_rad <= alignment_limit
    needs_correction = bool(lead_in["required"])

    if passive_alignment and not needs_correction:
        regime = "passive"
    elif needs_correction:
        regime = "active_centring_and_correction"
    else:
        regime = "active_centring"

    return {
        "regime": regime,
        "delivered_attitude_rad": performance.delivered_attitude_rad,
        "passive_alignment_limit_rad": alignment_limit,
        "passive_alignment_possible": passive_alignment,
        "alignment_shortfall_m": max(
            0.0, 0.5 * performance.delivered_attitude_rad * module_length_m - lateral_seating_tolerance_m
        ),
        "correcting_lead_in_required": needs_correction,
        "lead_in": lead_in,
        "what_the_channel_cannot_do": (
            "A module released in this channel may rest anywhere up to the clearance off the centre "
            "line, and in zero gravity nothing returns it. Above the alignment limit the geometry "
            "cannot guarantee the lateral gate and the controller that placed the module is the only "
            "thing that can."
            if not passive_alignment
            else "Within the alignment limit a plain channel guarantees the lateral gate on its own."
        ),
    }


@dataclass(frozen=True)
class RackRequirement:
    """Everything the closed form says about a rack for one manipulator."""

    performance: ManipulatorPerformance
    module_length_m: float
    module_width_m: float
    module_height_m: float
    channel_height_m: float
    lateral_clearance_window_m: dict[str, float]
    channel_inner_face_half_width_m: float
    shipped_section: SectionVerdict
    rail_indexing_bound_m: float
    accepted_sections: tuple[SectionVerdict, ...] = field(default_factory=tuple)

    def describe(self) -> dict[str, object]:
        """A JSON-safe record, for a report that has to say where a number came from."""

        window = self.lateral_clearance_window_m
        return {
            "measured_manipulator_performance": {
                "delivered_attitude_rad": self.performance.delivered_attitude_rad,
                "seating_tolerance_rad": self.performance.seating_tolerance_rad,
                "pad_half_bearing_offset_m": self.performance.pad_half_bearing_offset_m,
            },
            "module_m": {
                "length": self.module_length_m,
                "width": self.module_width_m,
                "height": self.module_height_m,
            },
            "rack_requirement": {
                "lateral_clearance_per_side_m": window,
                "channel_inner_face_half_width_m": self.channel_inner_face_half_width_m,
                "channel_height_m": self.channel_height_m,
                "rail_indexing_bound_m": self.rail_indexing_bound_m,
            },
            "shipped_section": {
                "accepted": self.shipped_section.accepted,
                "limiting_criterion": self.shipped_section.limiting_criterion,
                "entry_margin_m": self.shipped_section.entry_margin_m,
                "grip_margin_m": self.shipped_section.grip_margin_m,
            },
            "accepted_section_count": len(self.accepted_sections),
        }


def rack_requirement(
    performance: ManipulatorPerformance,
    *,
    module_length_m: float,
    module_width_m: float,
    module_height_m: float,
    channel_height_m: float,
    destination_relief_per_side_m: float = 0.0,
    section_widths_m: tuple[float, ...] = (0.110, 0.120, 0.130, 0.140, 0.150, 0.160),
    section_heights_m: tuple[float, ...] = (0.014, 0.018, 0.020, 0.024, 0.030, 0.035),
) -> RackRequirement:
    """Compute a rack requirement from measured manipulator performance.

    The channel is placed at the equal-margin design point rather than on either
    bound, and every other figure follows from that placement, so a designer
    changes the *manipulator* numbers and reads off a different rack rather than
    tuning the rack directly.
    """

    window = lateral_clearance_window(performance, module_length_m)
    inner_face = 0.5 * module_width_m + window["equal_margin_design_point_m"]

    def verdict(width_m: float, height_m: float) -> SectionVerdict:
        return section_verdict(
            performance,
            module_length_m=module_length_m,
            module_width_m=width_m,
            module_height_m=height_m,
            channel_inner_face_half_width_m=inner_face,
            channel_height_m=channel_height_m,
            destination_relief_per_side_m=destination_relief_per_side_m,
        )

    shipped = verdict(module_width_m, module_height_m)
    accepted = tuple(
        row
        for width in section_widths_m
        for height in section_heights_m
        if (row := verdict(width, height)).accepted
    )
    return RackRequirement(
        performance=performance,
        module_length_m=module_length_m,
        module_width_m=module_width_m,
        module_height_m=module_height_m,
        channel_height_m=channel_height_m,
        lateral_clearance_window_m=window,
        channel_inner_face_half_width_m=inner_face,
        shipped_section=shipped,
        rail_indexing_bound_m=rail_indexing_bound_m(
            performance,
            lateral_half_gap_m=shipped.lateral_half_gap_m,
            vertical_half_gap_m=shipped.vertical_half_gap_m,
        ),
        accepted_sections=accepted,
    )
