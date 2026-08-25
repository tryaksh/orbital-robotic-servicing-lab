"""Answer the workcell's geometry questions before anything starts a simulator.

``docs/archive/next_session_handoff.md`` asks four of them, in this order:

* can the arm reach the grasp pose in bay 1 and hold the module square there;
* can it pull the module straight out without folding up;
* can it reach bay 2 and hold the module square there too;
* does anything collide on the way.

``scripts/solve_workcell.py`` answered a weaker version of the first three with
one Isaac Sim launch per candidate base. This script answers them on the CPU in
about a second, and adds the two questions that sweep never asked.

**How much authority does the arm have left at each pose**, rather than whether
a 3000-step servo eventually converges. The chain's differential IK is DLS with
``DLS_LAMBDA``, so a commanded twist ``v`` is realised as
``J J^T (J J^T + lambda^2 I)^-1 v``. Where the arm is well conditioned that
matrix is the identity and the controller gets what it asks for; near a
singularity its rotational block collapses and a proportional loop with a
per-step clamp never converges inside a leg's step budget. That is the
mechanism section 6a of ``docs/service_interface_spec.md`` measured from the
outside, and it is an eigenvalue here.

**What attitude the destination channel actually admits.** A rigid part of
width ``w`` in a channel of width ``w + 2c``, engaged over a length ``l``,
sweeps ``w + l*sin(theta)`` across the channel. So the channel accepts
``theta <= 2c/l`` and nothing looser, whatever a success predicate says. This is
the requirement the delivered attitude has to meet, and it is not the one the
chain checks.

Kinematics are validated against the simulator's own answers before anything
else is reported: every solved configuration in
``evidence/workcell_reach_solution.json`` is run through this module's forward
kinematics and has to reproduce the tool pose the simulator recorded. If it does
not, the script fails instead of reporting.

Run with no arguments to check the shipped workcell. ``--report`` writes the
result as evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zero_g_blade_swap.grapple_geometry import (  # noqa: E402
    BLADE_LENGTH_M,
    FLARE_LEADING_X,
    GRIP_MAX_TRANSVERSE_M,
    SLOT_FLOOR_TOP_Z,
    SLOT_LIP_BOTTOM_Z,
    SLOT_MOUTH_X,
    TRANSIT_CLEAR_BLADE_CENTRE_X,
)

#: ``TRANSIT_FLARE_CLEARANCE_M`` in ``scripts/run_workflow_demo.py``: the margin
#: the retreat leg keeps behind the flare leading plane.
TRANSIT_FLARE_CLEARANCE_M = 0.010

#: How far the module's leading corner stands proud of half its length, measured
#: rather than assumed: ``measured_front_overhang_m`` in every robot-carried
#: report is 231.99 mm against the 225 mm half-length.
MEASURED_FRONT_OVERHANG_EXCESS_M = 0.006993

#: Where the crossing leg was actually observed to sit, from the tool_x_m column
#: of artifacts/robotcarried/diag_a025_trace.npz while leg 1 was stalled. It is
#: 10 mm deeper than the derivation above, because the tool hangs off the module
#: through an attitude that is not square while the leg is still squaring it.
OBSERVED_CROSSING_TOOL_X_M = -0.2197

#: Where the side guides stood before ``GUIDE_CENTER_OFFSET_Y`` was derived: the
#: value inherited from the 160 mm module, kept so the "before" row of
#: :func:`rail_constraint_change` compares two whole workcells rather than one
#: module against the other's rack.
HISTORIC_GUIDE_CENTER_OFFSET_Y = 0.08975

#: The attitude the chain measurably hands the insertion over at. Reported in
#: every robot-carried report as ``handoff_attitude_rad``; about 46 mrad.
DELIVERED_ATTITUDE_RAD = 0.046

#: ``--destination_channel_relief_m`` in the shipped preset. The destination bay
#: is relieved and the source bay is not, which is why a section has to be
#: judged against a different channel at each end of the job.
DESTINATION_RELIEF_M = 0.0046125

#: The module the recorded clearance sweep was taken on, before it was shortened.
#: Preserved evidence is checked against the geometry it was measured in.
SWEPT_MODULE_SIZE_M = (0.45, 0.16, 0.035)

ASSETS = PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py"
REACH_SOLUTION = PROJECT_ROOT / "evidence" / "workcell_reach_solution.json"
INSERTION = (
    PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "insertion.py"
)

# ---------------------------------------------------------------------------
# UR10e kinematics.
#
# The Denavit-Hartenberg parameters, the tool offset, the DLS damping and the
# solver live in ``zero_g_blade_swap.arm_kinematics``, because the driver needs
# the same solver batched in Torch to command the scripted transit legs and two
# copies of a kinematic chain is two chains. ``_validate_against_simulator``
# below proves this one against the simulator's own recorded configurations, so
# none of it has to be taken on trust, and ``tests/test_arm_kinematics.py``
# proves the Torch half against this one.
from zero_g_blade_swap.arm_kinematics import (  # noqa: E402
    DLS_LAMBDA,
    TOOL_OFFSET_Z,
    quaternion_to_matrix,
    realised_authority,
    rotation_vector,
    solve_ik,
    tool_pose,
)

#: ``GRAPPLE_HEAD_ON_TOOL_ROT``: the tool z axis down the rack's +x.
HEAD_ON_QUAT_WXYZ = (0.0, 0.7071068, 0.0, 0.7071068)
HEAD_ON = quaternion_to_matrix(*HEAD_ON_QUAT_WXYZ)

#: Length of a slot side guide, the ``_slot_guide_cfg`` default.
GUIDE_LENGTH_M = 0.60

#: Thickness of a side guide along y, from ``_slot_guide_cfg``.
GUIDE_THICKNESS_Y_M = 0.018

#: PhysX contact envelope authored on the side guides, from ``_slot_guide_cfg``.
GUIDE_CONTACT_OFFSET_M = 0.003


# ---------------------------------------------------------------------------
# Workcell constants that live in the Isaac Lab asset module, read as literals
# so an edit there moves this check rather than leaving it stale.


def _literal(name: str, path: Path = ASSETS) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not a literal assignment in {path}")


def channel_acceptance(relief_m: float = 0.0) -> dict[str, object]:
    """Return the attitude the destination channel admits, from its dimensions.

    A rigid part of half-width ``w/2`` inside a channel of half-width
    ``w/2 + c``, engaged over a length ``l`` measured from the mouth to its
    leading face, sweeps ``w + l*sin(theta)`` across the channel. It fits only
    while ``theta <= 2c/l``. Two independent limits apply at once, because the
    channel is rectangular and its two clearances differ.
    """

    blade_size = _literal("BLADE_SIZE")
    guide_offset = float(_literal("GUIDE_CENTER_OFFSET_Y"))
    # ``GUIDE_CENTER_OFFSET_Y`` places the guide *body* centre. The face the
    # module runs against is half the guide's own thickness inboard of it, and
    # reading the centre as the face is the mistake that turns a 0.75 mm
    # channel into a 9.75 mm one.
    guide_inner_face = guide_offset - 0.5 * GUIDE_THICKNESS_Y_M
    # ``service_destination_channel_relief_m`` opens both axes by the same
    # amount per side: the guides move outboard, the floor drops, the lips rise.
    lateral_clearance = guide_inner_face - 0.5 * float(blade_size[1]) + relief_m
    vertical_clearance = 0.5 * ((SLOT_LIP_BOTTOM_Z - SLOT_FLOOR_TOP_Z) - float(blade_size[2])) + relief_m
    seated_centre_x = float(_literal("BLADE_INSERTED_POS")[0])
    seated_engagement = min(BLADE_LENGTH_M, seated_centre_x + 0.5 * BLADE_LENGTH_M - SLOT_MOUTH_X)

    def accepted(engaged: float, clearance: float) -> float:
        return float(2.0 * clearance / engaged) if engaged > 0 else float("inf")

    depths = [0.05, 0.1, 0.2, 0.362, seated_engagement]
    def deepest_engagement(attitude: float) -> float:
        """How far a module held at ``attitude`` can enter before it wedges."""

        tightest = min(lateral_clearance, vertical_clearance)
        if attitude <= 0.0:
            return BLADE_LENGTH_M
        return min(BLADE_LENGTH_M, 2.0 * tightest / attitude)

    return {
        "channel_relief_per_side_m": round(relief_m, 6),
        "module_size_m": list(blade_size),
        "guide_body_centre_offset_m": round(guide_offset, 6),
        "guide_inner_face_half_width_m": round(guide_inner_face, 6),
        "physx_contact_offset_m": GUIDE_CONTACT_OFFSET_M,
        "guide_length_m": GUIDE_LENGTH_M,
        "lateral_clearance_per_side_m": round(lateral_clearance, 6),
        "vertical_clearance_per_side_m": round(vertical_clearance, 6),
        "seated_engagement_m": round(seated_engagement, 6),
        "law": "theta_max = 2 * clearance_per_side / engaged_length",
        "accepted_attitude_rad": [
            {
                "engaged_length_m": round(depth, 6),
                "lateral_yaw_rad": round(accepted(depth, lateral_clearance), 6),
                "vertical_pitch_rad": round(accepted(depth, vertical_clearance), 6),
            }
            for depth in depths
        ],
        "seated_requirement_rad": {
            "yaw": round(accepted(BLADE_LENGTH_M, lateral_clearance), 6),
            "pitch": round(accepted(BLADE_LENGTH_M, vertical_clearance), 6),
        },
        # Read the other way: what a delivered attitude costs in seating depth.
        # This is the curve the chain has been walking down without naming it.
        "deepest_engagement_for_attitude_m": [
            {"attitude_rad": attitude, "engagement_m": round(deepest_engagement(attitude), 6)}
            for attitude in (0.002, 0.005, 0.010, 0.0187, 0.0228, 0.050)
        ],
    }


def rail_constraint(
    section_m: tuple[float, float, float] | None = None,
    guide_offset_m: float | None = None,
) -> dict[str, object]:
    """Return what the rack's own channel still holds while the module is in it.

    The extraction skill's docstring says the rails constrain five of six
    motions, and nothing anywhere checks that. It was true of the module those
    tasks were certified on, so when the cross-section changed nothing said so.

    Three freedoms, from the same rectangle:

    * **lateral and vertical** translation, the half-gaps directly;
    * **pitch and yaw**, the ``2c/l`` law of :func:`channel_acceptance`, at the
      engagement the module still has -- deepest at the start of a pull, and
      opening up as it comes out;
    * **roll**, which no other function in this project computes. A section
      ``w`` by ``h`` rolled by ``theta`` about its long axis stands
      ``w*sin(theta) + h*cos(theta)`` tall and ``w*cos(theta) + h*sin(theta)``
      wide, and has to fit both ways at once.

    Roll is the one that matters and the one nothing measured, because roll is
    also the axis a pair of flat pads on a pin cannot resist: their contact
    normals lie along it. While the channel held roll to a few milliradians the
    grip never had to, and the moment the channel stopped holding it the grip
    inherited a job it has no geometry for.
    """

    blade_size = tuple(float(value) for value in _literal("BLADE_SIZE")) if section_m is None else section_m
    guide_offset = float(_literal("GUIDE_CENTER_OFFSET_Y")) if guide_offset_m is None else guide_offset_m
    guide_inner_face = guide_offset - 0.5 * GUIDE_THICKNESS_Y_M
    channel_width = 2.0 * guide_inner_face
    channel_height = SLOT_LIP_BOTTOM_Z - SLOT_FLOOR_TOP_Z
    width, height = float(blade_size[1]), float(blade_size[2])
    lateral = 0.5 * (channel_width - width)
    vertical = 0.5 * (channel_height - height)

    def max_roll_rad() -> float:
        """Largest roll a rectangle can take inside a rectangular channel."""

        def fits(theta: float) -> bool:
            sin, cos = np.sin(theta), np.cos(theta)
            return bool(
                width * sin + height * cos <= channel_height + 1e-12
                and width * cos + height * sin <= channel_width + 1e-12
            )

        if not fits(0.0):
            return 0.0
        # Both bounds are monotonic in theta over the range that can fit here,
        # so bisecting on "does it still fit" is exact to the tolerance.
        low, high = 0.0, 0.5 * float(np.pi)
        for _ in range(80):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if fits(middle) else (low, middle)
        return float(low)

    engagements = [BLADE_LENGTH_M, 0.30, 0.20, 0.10]
    return {
        "module_size_m": [round(value, 6) for value in blade_size],
        "channel_width_m": round(channel_width, 6),
        "channel_height_m": round(channel_height, 6),
        "lateral_half_gap_m": round(lateral, 6),
        "vertical_half_gap_m": round(vertical, 6),
        "max_roll_rad": round(max_roll_rad(), 6),
        "roll_law": "w*sin(t) + h*cos(t) <= H and w*cos(t) + h*sin(t) <= W",
        "pitch_yaw_by_engagement_rad": [
            {
                "engaged_length_m": round(engaged, 6),
                "pitch_rad": round(2.0 * vertical / engaged, 6),
                "yaw_rad": round(2.0 * lateral / engaged, 6),
            }
            for engaged in engagements
        ],
    }


def lateral_clearance_window() -> dict[str, object]:
    """Return the band the channel's lateral clearance has to lie in, and why.

    Two independent requirements bound it from opposite sides and neither was
    ever written down, so the number between them was inherited from a module
    that no longer exists.

    **From below**, the lead-ins have to admit the attitude the transit
    delivers. The chain delivers about 46 mrad, and ``2c/l`` says that needs
    ``c >= 46e-3 * l / 2``.

    **From above**, the grip has to be able to follow the module anywhere the
    channel lets it go. A pair of flat pads keeps half its face on the pin while
    the offset stays inside the pin's half-width, and a module in the *corner*
    of its channel is offset by ``hypot(lateral, vertical)``. The vertical gap
    is spent on the hand-off pitch requirement, so what is left for lateral is
    ``sqrt(pin_half^2 - vertical^2)``.

    The channel as it stood was outside the upper bound, which is what
    ``GUIDE_CENTER_OFFSET_Y`` now derives itself from.
    """

    entry = channel_acceptance(0.0)
    vertical = float(entry["vertical_clearance_per_side_m"])
    lateral = float(entry["lateral_clearance_per_side_m"])
    lower = 0.5 * DELIVERED_ATTITUDE_RAD * BLADE_LENGTH_M
    upper = float(np.sqrt(max(GRIP_MAX_TRANSVERSE_M**2 - vertical**2, 0.0)))
    return {
        "question": "how much lateral clearance may the channel have",
        "delivered_attitude_rad": DELIVERED_ATTITUDE_RAD,
        "pad_half_bearing_offset_m": GRIP_MAX_TRANSVERSE_M,
        "vertical_clearance_per_side_m": round(vertical, 6),
        "lower_bound_m": round(lower, 6),
        "lower_bound_reason": "the lead-ins must admit the attitude the transit delivers, 2c/l",
        "upper_bound_m": round(upper, 6),
        "upper_bound_reason": (
            "a module in the corner of its channel must stay inside the pads' half-bearing "
            "offset, so hypot(lateral, vertical) <= the pin's half-width"
        ),
        "as_built_m": round(lateral, 6),
        "inside_the_window": bool(lower - 1.0e-6 <= lateral <= upper + 1.0e-6),
        "historic_lateral_m": round(HISTORIC_GUIDE_CENTER_OFFSET_Y - 0.5 * GUIDE_THICKNESS_Y_M - 0.5 * 0.13, 6),
    }


def section_envelope(
    widths_m: tuple[float, ...] = (0.110, 0.120, 0.130, 0.140, 0.150, 0.160),
    heights_m: tuple[float, ...] = (0.014, 0.018, 0.020, 0.024, 0.030, 0.035),
) -> dict[str, object]:
    """Which module cross-sections this rack accepts, with no simulator.

    The chain is certified for one module. A cell that serves a family of them
    has to say which, and both bounds are closed form:

    * **entry** -- the module has to reach the seated plane at the attitude the
      transit delivers, which is ``2c/L >= theta`` on the destination channel,
      *with* its relief, because that is the surface the last 450 mm of the
      stroke runs against;
    * **grip** -- a module in the corner of the **source** channel, which has no
      relief, has to stay inside the offset at which a pad still keeps half its
      face on the pin, so ``hypot(lateral, vertical) <= pin_half_width``. That
      bay is where the pull happens and nothing was ever checked there.

    A section that fails the first jams on the way in. A section that fails the
    second is one the rack can move further than the gripper can follow, which
    is not a jam at all -- it shows up as lost grips during the *pull*, in the
    source bay, where nothing was looking.

    This is the map ``BLADE_SIZE`` was moved across blind: 450 x 160 x 35 mm
    fails entry and 450 x 130 x 20 mm failed grip, in the same rack, and the
    only thing that ever measured either was a training run.
    """

    guide_offset = float(_literal("GUIDE_CENTER_OFFSET_Y"))
    inner_face = guide_offset - 0.5 * GUIDE_THICKNESS_Y_M
    channel_height = SLOT_LIP_BOTTOM_Z - SLOT_FLOOR_TOP_Z
    needed = 0.5 * DELIVERED_ATTITUDE_RAD * BLADE_LENGTH_M

    rows: list[dict[str, object]] = []
    for width in widths_m:
        for height in heights_m:
            lateral = inner_face - 0.5 * width
            vertical = 0.5 * (channel_height - height)
            corner = float(np.hypot(lateral, vertical)) if min(lateral, vertical) > 0 else float("inf")
            enters = bool(
                lateral + DESTINATION_RELIEF_M >= needed - 1.0e-6
                and vertical + DESTINATION_RELIEF_M >= needed - 1.0e-6
            )
            holds = bool(min(lateral, vertical) > 0.0 and corner <= GRIP_MAX_TRANSVERSE_M + 1.0e-6)
            rows.append(
                {
                    "width_m": round(width, 6),
                    "height_m": round(height, 6),
                    "lateral_half_gap_m": round(lateral, 6),
                    "vertical_half_gap_m": round(vertical, 6),
                    "channel_corner_m": None if corner == float("inf") else round(corner, 6),
                    "lead_ins_admit_the_delivered_attitude": enters,
                    "pads_can_follow_the_corner": holds,
                    "accepted": bool(enters and holds),
                }
            )
    accepted = [row for row in rows if row["accepted"]]
    return {
        "question": "which module cross-sections does this rack accept",
        "delivered_attitude_rad": DELIVERED_ATTITUDE_RAD,
        "required_half_gap_m": round(needed, 6),
        "destination_relief_per_side_m": DESTINATION_RELIEF_M,
        "pad_half_bearing_offset_m": GRIP_MAX_TRANSVERSE_M,
        "guide_inner_face_half_width_m": round(inner_face, 6),
        "channel_height_m": round(channel_height, 6),
        "accepted_count": len(accepted),
        "grip_margin_of_the_shipped_section_m": round(
            GRIP_MAX_TRANSVERSE_M
            - float(
                np.hypot(
                    inner_face - 0.5 * float(_literal("BLADE_SIZE")[1]),
                    0.5 * (channel_height - float(_literal("BLADE_SIZE")[2])),
                )
            ),
            6,
        ),
        "note_on_the_shipped_section": (
            "GUIDE_CENTER_OFFSET_Y is derived as the *largest* lateral clearance the pads can "
            "follow, so the shipped module sits exactly on that bound with no margin. The window "
            "runs from 5.738 mm, and a value in the middle of it would leave a few millimetres on "
            "both sides. That is a measurement this session did not have the clock to take."
        ),
        "evaluated_count": len(rows),
        "sections": rows,
    }


def rail_constraint_change() -> dict[str, object]:
    """The same numbers for the section this project used to run, side by side.

    ``BLADE_SIZE`` went from 450 x 160 x 35 mm to 450 x 130 x 20 mm to buy
    clearance for the *destination* seating, and it was measured there. What was
    never measured is what it did to the *source* bay, where the extraction
    happens, and the answer is that it took the rails out of the load path.
    """

    before = rail_constraint(SWEPT_MODULE_SIZE_M, HISTORIC_GUIDE_CENTER_OFFSET_Y)
    after = rail_constraint()
    return {
        "question": "what did thinning the module do to what the rack holds during a pull",
        "before": before,
        "after": after,
        "roll_freedom_multiplier": round(float(after["max_roll_rad"]) / max(float(before["max_roll_rad"]), 1e-9), 2),
        "lateral_freedom_multiplier": round(
            float(after["lateral_half_gap_m"]) / max(float(before["lateral_half_gap_m"]), 1e-9), 2
        ),
        "vertical_freedom_multiplier": round(
            float(after["vertical_half_gap_m"]) / max(float(before["vertical_half_gap_m"]), 1e-9), 2
        ),
        "why_it_matters": (
            "a pair of flat pads closing on a pin cannot resist a moment about the closing axis, "
            "which is roll. While the channel held roll to a few milliradians the grip never had "
            "to; the extract skill's own docstring still says the rails constrain five of six "
            "motions, and for this section they no longer do"
        ),
        "guide_offset_is_not_derived": (
            "GUIDE_CENTER_OFFSET_Y is documented as '1.5 mm total clearance around the 160 mm "
            "blade' and was not moved when the blade stopped being 160 mm wide; the same is true "
            "of SLOT_UPPER_LIP_CENTER_Z, whose comment quotes 1.0 mm of lift for a 35 mm module"
        ),
    }


# ---------------------------------------------------------------------------
# The poses the whole job needs, and the sweep over where the arm stands.

SEED_JOINTS = [
    np.array([-0.30, -1.10, 1.60, 2.60, -1.35, -1.57]),
    np.array([-0.50, 0.42, -1.58, -1.99, -1.08, -1.57]),
    np.array([-1.00, -2.36, 2.65, 2.86, -0.57, -1.57]),
    np.array([-1.43, -0.11, -2.15, -4.03, 0.14, 1.57]),
]


def handoff_attitude_requirement() -> dict[str, object]:
    """Return the attitude a module may be handed to the insertion at.

    **The relief does not help here, and that is the whole point of computing it
    separately.** ``service_destination_channel_relief_m`` moves the side guides
    outboard, drops the floor and raises the lips, but the lead-in ramps and
    flares are authored from ``SLOT_LIP_BOTTOM_Z`` and ``SLOT_FLOOR_TOP_Z`` and
    are deliberately left where they are -- section 6 of the interface
    specification, and measured: lead-ins moved out with the relief stop touching
    the module in time to square it. So the entry gap is the *unrelieved* one
    however wide the channel behind it is, and a module that satisfies the seated
    fit can still be too crooked to get in.

    Same law as ``channel_acceptance``, evaluated at zero relief and over the
    full engagement, because a module that has to reach the seated plane has to
    survive every engagement on the way there and the tightest is the last.
    """

    at_the_lead_ins = channel_acceptance(0.0)
    lateral = float(at_the_lead_ins["lateral_clearance_per_side_m"])
    vertical = float(at_the_lead_ins["vertical_clearance_per_side_m"])
    pitch = 2.0 * vertical / BLADE_LENGTH_M
    yaw = 2.0 * lateral / BLADE_LENGTH_M
    return {
        "question": "at what attitude may the transit hand a module to the insertion",
        "lead_in_vertical_half_gap_m": round(vertical, 6),
        "lead_in_lateral_half_gap_m": round(lateral, 6),
        "law": "theta_max = 2 * half_gap / module_length",
        "required_pitch_rad": round(pitch, 6),
        "required_yaw_rad": round(yaw, 6),
        "requirement_rad": round(min(pitch, yaw), 6),
        "why_the_relief_does_not_relax_it": (
            "the relief moves the channel surfaces and leaves the lead-ins at the nominal "
            "surfaces, so the gap a module has to enter through is the unrelieved one"
        ),
        "what_the_chain_used_to_gate_on_rad": float(
            _literal("INSERTION_ORIENTATION_TOLERANCE_RAD", INSERTION)
        ),
        "seated_success_check_is_not_an_entry_requirement": (
            "INSERTION_ORIENTATION_TOLERANCE_RAD is the seated success predicate. Gating a "
            "hand-off on it lets the transit deliver a module that cannot enter, while every "
            "condition in the report reads true"
        ),
    }


def explain_seating_sweep() -> dict[str, object] | None:
    """Check the recorded clearance sweep against the acceptance law.

    ``evidence/robot_carried_seating_sweep.json`` was read as a trade: opening
    the channel buys travel and costs squareness, at about 3.5 mrad per
    millimetre, because "the channel was what was squaring the module".

    The law says something stricter and simpler. A module fully inside a channel
    with ``c`` of clearance per side cannot be tilted past ``2c/L``, so a module
    pushed in until it wedges *stops at* ``2c/L`` -- and the slope of that curve
    is ``2/L``, 4.44 mrad per millimetre on a 450 mm module, which is a property
    of the module's length and of nothing else. If the recorded attitudes track
    it, the sweep was measuring the module rather than the lead-ins.
    """

    sweep = PROJECT_ROOT / "evidence" / "robot_carried_seating_sweep.json"
    if not sweep.exists():
        return None
    # The sweep was recorded on the 450 x 160 x 35 mm module, so it is checked
    # against *that* module's numbers. Re-reading it through the current
    # geometry would be checking a preserved measurement against a rack it was
    # never taken in.
    vertical_clearance = 0.5 * ((SLOT_LIP_BOTTOM_Z - SLOT_FLOOR_TOP_Z) - SWEPT_MODULE_SIZE_M[2])
    points = json.loads(sweep.read_text(encoding="utf-8"))["points"]
    rows = []
    for point in points:
        relief = float(point["channel_relief_per_side_m"])
        measured = float(point["terminal_orientation_error_rad"])
        predicted = 2.0 * (vertical_clearance + relief) / SWEPT_MODULE_SIZE_M[0]
        rows.append(
            {
                "channel_relief_per_side_m": relief,
                "measured_terminal_attitude_rad": round(measured, 6),
                "predicted_by_acceptance_law_rad": round(predicted, 6),
                "ratio": round(measured / predicted, 4),
            }
        )
    ratios = [row["ratio"] for row in rows]
    return {
        "law": "a module fully inside the channel wedges at 2 * clearance_per_side / module_length",
        "slope_rad_per_m_of_relief": round(2.0 / SWEPT_MODULE_SIZE_M[0], 4),
        "published_slope_rad_per_m_of_relief": 3.5,
        "points": rows,
        "worst_ratio": round(min(ratios), 4),
        "best_ratio": round(max(ratios), 4),
    }


def executed_retreat_tool_x(installed_tool_x: float, retreat_offset: float) -> dict[str, object]:
    """Return the depth the crossing is actually flown at, not the certified one.

    ``evidence/workcell_reach_solution.json`` swept a pose called ``retreated``
    at the *nominal* clear centre, ``TRANSIT_CLEAR_BLADE_CENTRE_X``, and the
    base at -0.65 was adopted because that pose solved with 33 mm to spare.

    The chain does not fly that pose. ``_plan_relocation`` takes the module's
    **measured** front overhang -- 232 mm rather than the nominal 225, because
    the grapple pin and the module's own corner stand proud of its centre plane
    -- and then subtracts ``TRANSIT_FLARE_CLEARANCE_M`` on top. Both are
    deliberate and both push the retreat deeper. The gap is small in millimetres
    and not small in what it costs, which is the point of computing it here.
    """

    overhang = 0.5 * BLADE_LENGTH_M + MEASURED_FRONT_OVERHANG_EXCESS_M
    clear_centre_x = FLARE_LEADING_X - overhang - TRANSIT_FLARE_CLEARANCE_M
    # Taken from the sweep's own pair rather than from a nominal blade pose: the
    # certified ``retreated`` tool x and the module centre it corresponds to.
    nominal_retreat_tool_x = installed_tool_x + retreat_offset
    tool_to_module_x = nominal_retreat_tool_x - TRANSIT_CLEAR_BLADE_CENTRE_X
    return {
        "nominal_clear_blade_centre_x_m": round(TRANSIT_CLEAR_BLADE_CENTRE_X, 6),
        "measured_front_overhang_m": round(overhang, 6),
        "flare_clearance_margin_m": TRANSIT_FLARE_CLEARANCE_M,
        "executed_clear_blade_centre_x_m": round(clear_centre_x, 6),
        "tool_to_module_x_m": round(tool_to_module_x, 6),
        "nominal_retreat_tool_x_m": round(nominal_retreat_tool_x, 6),
        "executed_retreat_tool_x_m": round(clear_centre_x + tool_to_module_x, 6),
        "deeper_by_m": round(TRANSIT_CLEAR_BLADE_CENTRE_X - clear_centre_x, 6),
    }


def crossing_authority(base: tuple[float, float, float], tool_x: float, tool_z: float, bays: list[float]) -> list[dict[str, object]]:
    """Return the DLS authority along the bay-to-bay crossing at one depth.

    The crossing is the leg that has to hold the module square while the tool
    translates the bay pitch sideways, and it is the leg where holding it is
    hardest: it happens at the retreat depth, which is the folded end of this
    arm's envelope.
    """

    rows: list[dict[str, object]] = []
    for bay_y in np.linspace(max(bays), min(bays), 13):
        target = np.array([tool_x, float(bay_y), tool_z]) - np.array(base)
        joints, position_residual, attitude_residual = solve_ik(target, HEAD_ON, SEED_JOINTS)
        entry: dict[str, object] = {
            "tool_y_m": round(float(bay_y), 6),
            "position_residual_m": round(position_residual, 9),
            "attitude_residual_rad": round(attitude_residual, 9),
        }
        entry.update({key: round(value, 6) for key, value in realised_authority(joints).items()})
        rows.append(entry)
    return rows


def _required_poses() -> tuple[dict[str, float], float, list[float]]:
    solution = json.loads(REACH_SOLUTION.read_text(encoding="utf-8"))
    installed_tool_x = float(solution["installed_tool_x_local_m"])
    offsets = {name: float(value) for name, value in solution["required_pose_offsets_m"].items()}
    return offsets, installed_tool_x, [float(y) for y in solution["bays_y_m"]]


def _validate_against_simulator(tool_z: float) -> dict[str, object]:
    """Reproduce every configuration the simulator solved, or refuse to report."""

    solution = json.loads(REACH_SOLUTION.read_text(encoding="utf-8"))
    root = np.array(solution["solution"]["robot_root_local_m"], dtype=float)
    worst_position = 0.0
    worst_attitude = 0.0
    for pose in solution["solution"]["poses"]:
        joints = np.array(pose["arm_joint_pos_rad"], dtype=float)
        position, rotation = tool_pose(joints)
        position = position + root
        worst_position = max(
            worst_position,
            abs(position[0] - float(pose["reached_tool_x_local_m"])),
            abs(position[1] - float(pose["bay_y_m"])),
            abs(position[2] - tool_z),
        )
        worst_attitude = max(worst_attitude, float(np.linalg.norm(rotation_vector(HEAD_ON @ rotation.T))))
    return {
        "configurations_checked": len(solution["solution"]["poses"]),
        "worst_tool_position_disagreement_m": round(worst_position, 9),
        "worst_head_on_attitude_disagreement_rad": round(worst_attitude, 9),
        "position_tolerance_m": 1.0e-4,
        "attitude_tolerance_rad": 1.0e-4,
        "passed": bool(worst_position < 1.0e-4 and worst_attitude < 1.0e-4),
    }


def sweep_bases(bases: list[tuple[float, float, float]], tool_z: float) -> list[dict[str, object]]:
    offsets, installed_tool_x, bays = _required_poses()
    rows: list[dict[str, object]] = []
    for base in bases:
        poses: list[dict[str, object]] = []
        for bay_y in bays:
            for name, offset in offsets.items():
                target = np.array([installed_tool_x + offset, bay_y, tool_z]) - np.array(base)
                joints, position_residual, attitude_residual = solve_ik(target, HEAD_ON, SEED_JOINTS)
                entry: dict[str, object] = {
                    "pose": name,
                    "bay_y_m": bay_y,
                    "target_tool_x_local_m": round(installed_tool_x + offset, 6),
                    "position_residual_m": round(position_residual, 9),
                    "attitude_residual_rad": round(attitude_residual, 9),
                    "solved": bool(position_residual < 0.002 and attitude_residual < 0.010),
                }
                entry.update({key: round(value, 6) for key, value in realised_authority(joints).items()})
                entry["arm_joint_pos_rad"] = [round(float(value), 6) for value in joints]
                poses.append(entry)
        rows.append(
            {
                "robot_root_local_m": list(base),
                "all_required_poses_solved": all(pose["solved"] for pose in poses),
                "worst_position_residual_m": max(pose["position_residual_m"] for pose in poses),
                "worst_attitude_residual_rad": max(pose["attitude_residual_rad"] for pose in poses),
                "worst_rotational_authority": min(pose["authority_worst_rotation_axis"] for pose in poses),
                "worst_any_axis_authority": min(pose["authority_worst_any_axis"] for pose in poses),
                "worst_jacobian_min_singular_value": min(pose["jacobian_min_singular_value"] for pose in poses),
                "worst_joint_travel_used_fraction": max(pose["joint_travel_used_fraction"] for pose in poses),
                "poses": poses,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Workcell geometry check, no simulator.")
    parser.add_argument("--report", type=Path, default=None, help="Write the result as evidence JSON.")
    parser.add_argument("--tool_z", type=float, default=0.72, help="Grip point height, the module centre line.")
    parser.add_argument(
        "--relief_m",
        type=float,
        nargs="*",
        default=[0.0, 0.0046125],
        help="Per-side channel relief to report the acceptance at. Default: as built, and the shipped preset.",
    )
    parser.add_argument(
        "--base_x",
        type=float,
        nargs="*",
        default=[-0.65, -0.70, -0.75, -0.85, -0.95],
        help="Base x positions to profile the bay-to-bay crossing at.",
    )
    arguments = parser.parse_args()

    validation = _validate_against_simulator(arguments.tool_z)
    if not validation["passed"]:
        print("KINEMATIC MODEL DISAGREES WITH THE SIMULATOR -- refusing to report")
        print(json.dumps(validation, indent=2))
        return 1

    shipped = tuple(float(value) for value in _literal("GRAPPLE_ROBOT_ROOT_POS"))
    second_bay_y = float(_literal("SECOND_SLOT_CENTER_Y"))
    bases = [
        shipped,
        (shipped[0], 0.5 * second_bay_y, shipped[2]),
        (shipped[0], second_bay_y, shipped[2]),
        (-0.45, 0.0, shipped[2]),
    ]
    rows = sweep_bases([tuple(base) for base in bases], arguments.tool_z)
    acceptance = [channel_acceptance(relief) for relief in arguments.relief_m]
    handoff = handoff_attitude_requirement()
    sweep_explanation = explain_seating_sweep()
    offsets, installed_tool_x, bays = _required_poses()
    retreat = executed_retreat_tool_x(installed_tool_x, offsets["retreated"])
    # Two depths, because the derived one and the one the chain was observed at
    # differ by the tool's own attitude offset and the curve is steep here.
    depths = [float(retreat["executed_retreat_tool_x_m"]), OBSERVED_CROSSING_TOOL_X_M]
    crossing = {
        f"base_x={candidate:.2f},tool_x={depth:.4f}": crossing_authority(
            (candidate, 0.0, shipped[2]), depth, arguments.tool_z, bays
        )
        for depth in depths
        for candidate in arguments.base_x
    }

    print(
        f"kinematics validated against {validation['configurations_checked']} simulator configurations: "
        f"{validation['worst_tool_position_disagreement_m'] * 1000:.4f} mm, "
        f"{validation['worst_head_on_attitude_disagreement_rad'] * 1000:.4f} mrad"
    )
    print()
    header = f"{'base x':>7} {'base y':>7} {'solved':>7} {'pos mm':>8} {'att mrad':>9}"
    print(f"{header} {'min rot authority':>18} {'min sigma':>10} {'joint use':>10}")
    for row in rows:
        base = row["robot_root_local_m"]
        print(
            f"{base[0]:7.3f} {base[1]:7.3f} {str(row['all_required_poses_solved']):>7} "
            f"{row['worst_position_residual_m'] * 1000:8.4f} "
            f"{row['worst_attitude_residual_rad'] * 1000:9.5f} "
            f"{row['worst_rotational_authority']:18.4f} "
            f"{row['worst_jacobian_min_singular_value']:10.4f} "
            f"{row['worst_joint_travel_used_fraction']:10.3f}"
        )
    print()
    print(
        f"{'relief/side':>11} {'lateral mm':>10} {'vertical mm':>11} "
        f"{'seated yaw mrad':>15} {'seated pitch mrad':>17}"
    )
    for entry in acceptance:
        seated = entry["seated_requirement_rad"]
        print(
            f"{entry['channel_relief_per_side_m'] * 1000:11.3f} "
            f"{entry['lateral_clearance_per_side_m'] * 1000:10.3f} "
            f"{entry['vertical_clearance_per_side_m'] * 1000:11.3f} "
            f"{seated['yaw'] * 1000:15.2f} {seated['pitch'] * 1000:17.2f}"
        )
    print()
    rails = rail_constraint_change()
    header = f"{'module section':>22} {'lateral mm':>11} {'vertical mm':>12}"
    print(f"{header} {'max roll mrad':>14} {'pitch at full mrad':>19}")
    for label in ("before", "after"):
        entry = rails[label]
        section = entry["module_size_m"]
        name = f"{section[1] * 1000:.0f} x {section[2] * 1000:.0f} mm ({label})"
        print(
            f"{name:>22} "
            f"{entry['lateral_half_gap_m'] * 1000:11.3f} "
            f"{entry['vertical_half_gap_m'] * 1000:12.3f} "
            f"{entry['max_roll_rad'] * 1000:14.2f} "
            f"{entry['pitch_yaw_by_engagement_rad'][0]['pitch_rad'] * 1000:19.2f}"
        )
    print(
        f"the source bay now leaves {rails['roll_freedom_multiplier']:.0f}x the roll it did; "
        "roll is the axis the pads cannot resist"
    )
    window = lateral_clearance_window()
    print(
        f"lateral clearance window: {window['lower_bound_m'] * 1000:.2f} mm "
        f"(lead-ins must admit {window['delivered_attitude_rad'] * 1000:.0f} mrad) to "
        f"{window['upper_bound_m'] * 1000:.2f} mm (the pads must be able to follow the corner); "
        f"as built {window['as_built_m'] * 1000:.3f} mm, "
        f"inside={window['inside_the_window']}; it was {window['historic_lateral_m'] * 1000:.3f} mm"
    )

    print()
    envelope = section_envelope()
    print(
        f"module sections this rack accepts ({envelope['accepted_count']} of "
        f"{envelope['evaluated_count']} evaluated), '.' rejected"
    )
    heights = sorted({row["height_m"] for row in envelope["sections"]})
    widths = sorted({row["width_m"] for row in envelope["sections"]})
    lookup = {(row["width_m"], row["height_m"]): row for row in envelope["sections"]}
    print("  width \\ height " + " ".join(f"{height * 1000:6.0f}" for height in heights))
    for width in widths:
        marks = []
        for height in heights:
            row = lookup[(width, height)]
            if row["accepted"]:
                marks.append("    ok")
            elif not row["lead_ins_admit_the_delivered_attitude"]:
                marks.append(" entry")
            else:
                marks.append("  grip")
        print(f"  {width * 1000:13.0f} " + " ".join(marks))

    print()
    print(
        "hand-off attitude requirement at the lead-ins: "
        f"{handoff['requirement_rad'] * 1000:.2f} mrad "
        f"(pitch {handoff['required_pitch_rad'] * 1000:.2f}, "
        f"yaw {handoff['required_yaw_rad'] * 1000:.2f}); "
        f"the chain used to gate on {handoff['what_the_chain_used_to_gate_on_rad'] * 1000:.2f}"
    )
    print()
    print(
        f"crossing depth: certified tool x {retreat['nominal_retreat_tool_x_m']:.4f}, "
        f"executed {retreat['executed_retreat_tool_x_m']:.4f} "
        f"({retreat['deeper_by_m'] * 1000:.1f} mm deeper)"
    )
    print(f"{'crossing profile':>34} {'unreached':>10} {'worst authority':>16} {'at tool y':>10} {'min sigma':>10}")
    for name, profile in crossing.items():
        unreached = [
            entry
            for entry in profile
            if entry["position_residual_m"] >= 0.002 or entry["attitude_residual_rad"] >= 0.010
        ]
        reachable = [entry for entry in profile if entry not in unreached]
        if not reachable:
            print(f"{name:>34} {len(unreached):>4}/{len(profile):<5}   no pose holds attitude")
            continue
        worst = min(reachable, key=lambda entry: entry["authority_worst_any_axis"])
        print(
            f"{name:>34} {len(unreached):>4}/{len(profile):<5} "
            f"{worst['authority_worst_any_axis']:16.4f} "
            f"{worst['tool_y_m']:10.3f} {worst['jacobian_min_singular_value']:10.4f}"
        )

    if sweep_explanation is not None:
        print()
        print(
            "recorded seating sweep against the acceptance law: measured attitude is "
            f"{sweep_explanation['worst_ratio']:.2f} to {sweep_explanation['best_ratio']:.2f} "
            f"of 2c/L across {len(sweep_explanation['points'])} points; the sweep's published "
            f"{sweep_explanation['published_slope_rad_per_m_of_relief']:.1f} mrad per mm slope "
            f"is 2/L = {sweep_explanation['slope_rad_per_m_of_relief']:.2f} mrad per mm"
        )
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "title": "Where the arm stands, what authority it has there, and what the channel admits",
            "evidence_type": "geometry_check_no_simulator",
            "generated_utc": datetime.now(UTC).isoformat(),
            "question": (
                "Is the destination bay's seating failure caused by where the robot stands, and what attitude "
                "does the destination channel actually accept?"
            ),
            "method": (
                "scripts/check_workcell_geometry.py. Closed-form UR10e kinematics validated against every "
                "configuration evidence/workcell_reach_solution.json recorded from the simulator, then a "
                "damped-least-squares IK at the head-on attitude for every required pose in both bays, with the "
                "realised authority of the chain's own DLS controller reported at each. The channel's angular "
                "acceptance is closed form from the rack dimensions."
            ),
            "kinematic_validation": validation,
            "dls_lambda": DLS_LAMBDA,
            "tool_offset_z_m": TOOL_OFFSET_Z,
            "shipped_robot_root_local_m": list(shipped),
            "bases": rows,
            "executed_crossing_depth": retreat,
            "crossing_authority_by_base_x": crossing,
            "destination_channel_acceptance": acceptance,
            "source_bay_rail_constraint": rail_constraint_change(),
            "lateral_clearance_window": lateral_clearance_window(),
            "module_section_envelope": section_envelope(),
            "recorded_seating_sweep_against_the_law": sweep_explanation,
            "scope_and_limitations": (
                "Kinematics and rigid-body geometry only. It says what poses the arm can hold and what attitude a "
                "straight channel admits; it says nothing about contact resolution, joint drive dynamics, or the "
                "compliant mount, all of which need the simulator."
            ),
        }
        arguments.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
