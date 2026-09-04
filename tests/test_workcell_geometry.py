"""Defend the two numbers ``scripts/check_workcell_geometry.py`` produces.

The first is that the closed-form kinematics agree with the simulator. Without
that, every other number the script prints is a guess with decimals on it.

The second is the finding: the destination channel admits far less attitude than
the seating check tolerates, so a chain gated on the seating check cannot see a
module that is too crooked to enter. That is a claim about the rack's own
dimensions, and it should break loudly if either of those dimensions moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py"
)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_workcell_geometry import (  # noqa: E402
    DLS_LAMBDA,
    GUIDE_THICKNESS_Y_M,
    HEAD_ON,
    OBSERVED_CROSSING_TOOL_X_M,
    _literal,
    _required_poses,
    _validate_against_simulator,
    channel_acceptance,
    crossing_authority,
    executed_retreat_tool_x,
    explain_seating_sweep,
    handoff_attitude_requirement,
    lateral_clearance_window,
    parked_base_offset_profile,
    rail_constraint,
    rail_constraint_change,
    realised_authority,
    rotation_vector,
    section_envelope,
    solve_ik,
    sweep_bases,
)

#: The seating check's own tolerance, read as a literal rather than imported:
#: ``mdp/insertion.py`` pulls in Isaac Lab, and the point of this check is that
#: it needs no simulator.
INSERTION_ORIENTATION_TOLERANCE_RAD = float(
    _literal(
        "INSERTION_ORIENTATION_TOLERANCE_RAD",
        PROJECT_ROOT / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "mdp" / "insertion.py",
    )
)

TOOL_Z = 0.72


def test_the_kinematics_reproduce_every_configuration_the_simulator_solved():
    validation = _validate_against_simulator(TOOL_Z)
    assert validation["configurations_checked"] >= 8
    assert validation["passed"], validation


def test_the_shipped_base_holds_the_head_on_attitude_at_every_required_pose():
    shipped = tuple(float(value) for value in _literal("GRAPPLE_ROBOT_ROOT_POS"))
    row = sweep_bases([shipped], TOOL_Z)[0]
    assert row["all_required_poses_solved"], row
    # Not merely solvable: solvable with the differential IK's authority intact.
    # Below about 0.9 a proportional loop with a per-step clamp stops converging
    # inside a leg's step budget, which is the failure section 6a measured.
    assert row["worst_rotational_authority"] > 0.9, row["worst_rotational_authority"]


def test_the_old_base_still_reproduces_the_measured_reach_boundary():
    # evidence/relocation_reach_boundary.json recorded 166.95 mm of shortfall at
    # the retreat with the base at -0.45. If the kinematics here cannot
    # reproduce a known failure they cannot be trusted on a new success.
    row = sweep_bases([(-0.45, 0.0, 0.15)], TOOL_Z)[0]
    assert not row["all_required_poses_solved"]
    assert row["worst_position_residual_m"] == pytest.approx(0.16695, abs=5.0e-4)


#: What the scripted legs deliver. A controller floor, not a tuning gap: the
#: squaring leg limit-cycles at about one action scale, and a smaller gain makes
#: it diverge. Measured at 14.8 to 21.5 mrad across the rail runs.
DELIVERED_ATTITUDE_RAD = 0.0215


def test_the_rack_as_built_admits_the_attitude_this_arm_delivers():
    """The requirement the whole branch was failing, now met by geometry.

    A rigid module of length *L* fits a channel with *c* per side only while its
    attitude is under ``2c/L``. At 450 x 160 x 35 mm that was 2.22 mrad vertical
    against a delivery of 21.5, which is why nothing seated and why no relief
    could fix it -- widening the channel widens the tilt it permits by the same
    ratio. Shortening and thinning the module moves both sides of the inequality
    the right way at once.
    """

    acceptance = channel_acceptance()
    seated = acceptance["seated_requirement_rad"]
    # 35.6 mrad of pitch and 70.0 of yaw, against a delivery of 21.5: a factor
    # of 1.65 on the tighter axis. Not luxurious, and stated rather than
    # rounded up, because the entry transient the mating imparts spends some of
    # it. The axis that is tight is the one the channel is tight on, and 20 mm
    # of module in a 36 mm channel is as thin as the pin boss allows.
    assert seated["pitch"] > 1.5 * DELIVERED_ATTITUDE_RAD, seated["pitch"]
    # Yaw went 70.0 -> 56.4 -> 49.2 mrad, and the last step is the one that
    # matters: it is now under the attitude a seated module is accepted at, so
    # a module that merely rests in this channel passes rather than failing on
    # the wall. It was 4.04 mrad outside that criterion, which is what the
    # insert skill measured and what no controller could have fixed.
    assert seated["yaw"] > DELIVERED_ATTITUDE_RAD, seated["yaw"]
    assert seated["yaw"] < INSERTION_ORIENTATION_TOLERANCE_RAD, seated["yaw"]
    assert seated["yaw"] > seated["pitch"], (seated["yaw"], seated["pitch"])


def test_the_lateral_clearance_is_inside_the_window_two_requirements_leave_it():
    """The rack may not hold a module outside its own acceptance criterion.

    Three independent requirements bound the channel's lateral clearance and
    none had ever been written down, so the number between them was left
    wherever the previous module's cross-section put it.

    From below, the lead-ins have to admit the attitude the transit delivers.
    From above, a module that merely *rests* in the channel wedges at ``2c/L``
    and cannot be squarer, so the channel may not be wider than the attitude a
    seated module is accepted at. A third bound -- a module in the corner of its
    channel staying inside the offset at which a pad still keeps half its face
    on the pin -- used to be the binding one and no longer is.

    Both of the values this replaces sat *on* a bound. The inherited 15.750 mm
    was outside the pads' 12.689 and cost measured grips: 65 of 92 extract
    failures at stage 0 ended with the grip more than 13.5 mm across the pin.
    The 12.689 mm that replaced it sat exactly on the pads' bound and 4.04 mrad
    outside the seated criterion, which is what the insert skill then measured.
    """

    window = lateral_clearance_window()
    assert window["inside_the_window"], window
    assert window["lower_bound_m"] < window["as_built_m"] <= window["upper_bound_m"] + 1.0e-6
    # The criterion binds from above now, and the pads' bound does not.
    assert not window["pad_bound_still_binds"], window
    assert window["upper_bound_m"] < window["pad_bound_m"], window
    # Both superseded values are outside the window, which is what makes it
    # worth checking rather than a restatement of the constant.
    assert window["historic_lateral_m"] > window["upper_bound_m"], window
    assert window["superseded_lateral_m"] > window["upper_bound_m"], window
    # And a module resting in the channel now passes the criterion it is
    # judged by, which is the whole point of the change.
    assert window["as_built_resting_attitude_rad"] < INSERTION_ORIENTATION_TOLERANCE_RAD, window
    # The corner the pads have to follow gained margin as a consequence, not as
    # a target: it was exactly GRIP_MAX_TRANSVERSE_M and is now inside it.
    assert window["as_built_channel_corner_m"] < window["pad_half_bearing_offset_m"], window


def test_the_guide_offset_is_the_derivation_and_not_a_remembered_number():
    """``GUIDE_CENTER_OFFSET_Y`` must reproduce the window's derived value.

    Not a bound. Sitting on a bound is what both previous values did and what
    cost this project a training run each time, so the derivation places the
    clearance where the two margins are equal.
    """

    window = lateral_clearance_window()
    thickness = GUIDE_THICKNESS_Y_M
    blade = _literal("BLADE_SIZE")
    expected = 0.5 * float(blade[1]) + window["derived_m"] + 0.5 * thickness
    assert abs(float(_literal("GUIDE_CENTER_OFFSET_Y")) - expected) < 1.0e-5, expected
    # Equal margins, which is what "derived" means here.
    below = window["as_built_resting_attitude_rad"] - window["delivered_attitude_rad"]
    above = window["seated_orientation_tolerance_rad"] - window["as_built_resting_attitude_rad"]
    assert abs(below - above) < 1.0e-4, (below, above)
    assert below > 0.003, below


def test_the_rails_stopped_holding_the_module_when_it_was_thinned():
    """The finding the extract skill's docstring still contradicts.

    Extraction's docstring says the rails constrain five of six motions. That
    was true of 450 x 160 x 35 mm in this channel and is not true of
    450 x 130 x 20 mm, and the axis it is least true of is roll -- which is also
    the axis a pair of flat pad normals cannot resist, because the normals lie
    along it.
    """

    change = rail_constraint_change()
    assert change["before"]["max_roll_rad"] < 0.010, change["before"]
    assert change["after"]["max_roll_rad"] > 0.100, change["after"]
    assert change["roll_freedom_multiplier"] > 10.0, change
    # Roll is unbounded by engagement, unlike pitch and yaw: it is the same
    # number at the mouth and at the seated plane, so a pull never escapes it.
    deep = rail_constraint()["pitch_yaw_by_engagement_rad"]
    assert deep[0]["engaged_length_m"] > deep[-1]["engaged_length_m"]
    assert deep[0]["pitch_rad"] < deep[-1]["pitch_rad"]


def test_the_leading_corner_clears_the_lead_in_at_the_delivered_attitude():
    """What actually jammed: the corner, not the channel.

    The module enters nose first, so the quantity that has to fit is the dip of
    its leading corner, ``(L/2) * theta``, against the vertical half-gap it has
    left after whatever centring error it arrives with.
    """

    acceptance = channel_acceptance()
    half_length = 0.5 * float(acceptance["module_size_m"][0])
    dip = half_length * DELIVERED_ATTITUDE_RAD
    centring_error = 0.0013  # measured at the hand-off, in every rail run
    assert dip + centring_error < acceptance["vertical_clearance_per_side_m"], (
        dip,
        acceptance["vertical_clearance_per_side_m"],
    )


def test_the_seating_check_is_no_longer_the_looser_of_the_two_tests():
    """The predicate used to be 24 times looser than the geometry.

    It no longer is, which means a run that passes the seating check is a run
    whose module physically fits -- the property the chain needed all along.
    """

    seated = channel_acceptance()["seated_requirement_rad"]
    assert min(seated["yaw"], seated["pitch"]) < INSERTION_ORIENTATION_TOLERANCE_RAD
    assert min(seated["yaw"], seated["pitch"]) > INSERTION_ORIENTATION_TOLERANCE_RAD / 4.0


def test_a_base_opposite_the_bay_makes_that_bay_identical_to_the_first():
    # What a rail would buy, as a number rather than an argument: parked
    # opposite a bay, the arm's configuration there is the one it has at bay 1,
    # so a skill certified in bay 1 is certified in every bay the rail reaches.
    second_bay_y = float(_literal("SECOND_SLOT_CENTER_Y"))
    shipped = tuple(float(value) for value in _literal("GRAPPLE_ROBOT_ROOT_POS"))
    head_on = sweep_bases([(shipped[0], 0.0, shipped[2])], TOOL_Z)[0]
    railed = sweep_bases([(shipped[0], second_bay_y, shipped[2])], TOOL_Z)[0]
    bay_one = {p["pose"]: p for p in head_on["poses"] if p["bay_y_m"] == 0.0}
    bay_two = {p["pose"]: p for p in railed["poses"] if p["bay_y_m"] == second_bay_y}
    for name, reference in bay_one.items():
        assert bay_two[name]["authority_worst_rotation_axis"] == pytest.approx(
            reference["authority_worst_rotation_axis"], abs=1.0e-5
        )
        assert bay_two[name]["jacobian_min_singular_value"] == pytest.approx(
            reference["jacobian_min_singular_value"], abs=1.0e-4
        )


def test_a_ten_millimetre_rail_stop_error_is_still_kinematically_feasible():
    shipped = tuple(float(value) for value in _literal("GRAPPLE_ROBOT_ROOT_POS"))
    profile = parked_base_offset_profile(
        base_x_m=shipped[0], stop_error_y_m=0.010, tool_z_m=TOOL_Z, base_z_m=shipped[2]
    )
    assert profile["required_poses"] == 8
    assert profile["maximum_position_residual_m"] < 1.0e-4
    assert profile["maximum_attitude_residual_rad"] < 1.0e-4
    assert profile["minimum_worst_axis_authority"] > 0.9
    assert all(
        pose["base_y_m"] == pytest.approx(pose["bay_y_m"] + 0.010, abs=1.0e-6)
        for pose in profile["poses"]
    )


def test_the_jacobian_matches_a_finite_difference_of_the_forward_kinematics():
    from zero_g_blade_swap.arm_kinematics import tool_jacobian, tool_pose

    joints = np.array([-0.30, -1.10, 1.60, 2.60, -1.35, -1.57])
    analytic = tool_jacobian(joints)
    step = 1.0e-6
    for index in range(6):
        moved = joints.copy()
        moved[index] += step
        position_a, rotation_a = tool_pose(joints)
        position_b, rotation_b = tool_pose(moved)
        numeric = np.concatenate(
            [(position_b - position_a) / step, rotation_vector(rotation_b @ rotation_a.T) / step]
        )
        assert numeric == pytest.approx(analytic[:, index], abs=1.0e-4)


def test_the_head_on_attitude_is_reachable_at_the_deepest_pose_in_the_second_bay():
    shipped = np.array([float(value) for value in _literal("GRAPPLE_ROBOT_ROOT_POS")])
    second_bay_y = float(_literal("SECOND_SLOT_CENTER_Y"))
    target = np.array([0.379975, second_bay_y, TOOL_Z]) - shipped
    seeds = [np.array([-0.50, 0.42, -1.58, -1.99, -1.08, -1.57])]
    joints, position_residual, attitude_residual = solve_ik(target, HEAD_ON, seeds)
    assert position_residual < 1.0e-5
    assert attitude_residual < 1.0e-5
    assert realised_authority(joints)["authority_worst_rotation_axis"] > 0.9


def test_the_executed_crossing_depth_is_deeper_than_the_pose_the_base_was_chosen_for():
    # The base was adopted because the *nominal* retreat solved with margin.
    # The chain flies a deeper one, derived at run time from the module's
    # measured front overhang plus the flare clearance margin. If that stops
    # being true this check should say so, because the base position rests on it.
    offsets, installed_tool_x, _ = _required_poses()
    retreat = executed_retreat_tool_x(installed_tool_x, offsets["retreated"])
    assert retreat["deeper_by_m"] > 0.010
    assert retreat["executed_retreat_tool_x_m"] < retreat["nominal_retreat_tool_x_m"]


def test_the_crossing_loses_authority_at_the_shipped_base_and_keeps_it_further_back():
    """Record the open defect as a measurement, on both sides of the trade.

    This is not a property the shipped workcell has. The crossing is the leg
    that has to keep the module square while the tool translates the bay pitch
    sideways, and it happens at the folded end of the envelope. Endpoint
    reachability is not enough to survive it: at -0.65 every pose on the path
    solves and the realised authority still falls to 0.72, which is what stalled
    the destination squaring leg for 380 control steps at 144 mrad.

    Asserted in both directions so that either fix -- moving the base, or a rail
    that removes the lateral translation from the arm -- breaks this test and
    forces the number to be re-read rather than assumed.
    """

    shipped = tuple(float(value) for value in _literal("GRAPPLE_ROBOT_ROOT_POS"))
    _, _, bays = _required_poses()
    at_shipped = crossing_authority(shipped, OBSERVED_CROSSING_TOOL_X_M, TOOL_Z, bays)
    further_back = crossing_authority(
        (shipped[0] - 0.10, shipped[1], shipped[2]), OBSERVED_CROSSING_TOOL_X_M, TOOL_Z, bays
    )
    # Every pose on the crossing is reachable at both. Reach is not the fault.
    for profile in (at_shipped, further_back):
        assert all(entry["position_residual_m"] < 0.002 for entry in profile)
        assert all(entry["attitude_residual_rad"] < 0.010 for entry in profile)
    worst_shipped = min(entry["authority_worst_any_axis"] for entry in at_shipped)
    worst_back = min(entry["authority_worst_any_axis"] for entry in further_back)
    sigma_shipped = min(entry["jacobian_min_singular_value"] for entry in at_shipped)
    assert worst_shipped == pytest.approx(0.72, abs=0.03), worst_shipped
    assert sigma_shipped < 2.0 * DLS_LAMBDA, sigma_shipped
    assert worst_back > 0.99, worst_back
    # And it is the destination bay that pays, not the source.
    worst_entry = min(at_shipped, key=lambda entry: entry["authority_worst_any_axis"])
    assert worst_entry["tool_y_m"] == pytest.approx(min(bays), abs=1.0e-6)


def test_the_recorded_seating_sweep_is_the_acceptance_law():
    """The published clearance sweep was measuring the module's length.

    Every row of ``evidence/robot_carried_seating_sweep.json`` stopped within
    13% of ``2c/L``, and the slope it was read as a controller trade -- 3.5 mrad
    of squareness per millimetre of relief -- is ``2/L``, 4.44 mrad per
    millimetre on a 450 mm module. A module pushed into a channel until it
    wedges stops at the largest tilt that channel admits, by definition.
    """

    explanation = explain_seating_sweep()
    assert explanation is not None, "the recorded sweep is a preserved result"
    assert len(explanation["points"]) >= 8
    assert explanation["worst_ratio"] > 0.85
    assert explanation["best_ratio"] < 1.10
    # The published slope and the geometric one, in the same units.
    assert explanation["slope_rad_per_m_of_relief"] == pytest.approx(4.44, abs=0.01)


def test_the_handoff_attitude_requirement_is_tighter_than_the_seating_check():
    """The gate the transit hands over on is an entry requirement, not a success one.

    ``INSERTION_ORIENTATION_TOLERANCE_RAD`` says whether a *seated* module counts
    as seated. What decides whether a module can get in is the lead-in gap over
    the length it has to travel, and it is the smaller of the two. A chain gated
    on the success check delivers modules that wedge, with every condition in the
    report reading true -- which is what happened: handed over at 52.4 mrad, into
    lead-ins that admit 35.6, wedged 53 mm short of seated.
    """

    requirement = handoff_attitude_requirement()
    seating = requirement["what_the_chain_used_to_gate_on_rad"]
    assert requirement["requirement_rad"] < seating
    # And it is deliberately conservative rather than exact: the module enters
    # through the lead-in gap and sits in the relieved channel behind it, which is
    # wider, so modules measurably seat above this bound. 46.7 mrad, observed.
    assert channel_acceptance(0.0046125)["seated_requirement_rad"]["pitch"] > 0.0467
    # And the tighter of the two axes is the vertical one, because the module is
    # thin and the lips are close.
    assert requirement["requirement_rad"] == requirement["required_pitch_rad"]
    assert requirement["required_pitch_rad"] == pytest.approx(0.03556, abs=1.0e-4)


def test_the_channel_relief_does_not_relax_the_entry_requirement():
    """The lead-ins stay at the nominal surfaces when the channel is relieved."""

    requirement = handoff_attitude_requirement()
    relieved = channel_acceptance(0.0046125)
    # The seated fit gets looser with relief and the entry requirement does not.
    assert relieved["seated_requirement_rad"]["pitch"] > requirement["required_pitch_rad"]
    assert requirement["lead_in_vertical_half_gap_m"] == pytest.approx(
        channel_acceptance(0.0)["vertical_clearance_per_side_m"]
    )


def test_the_rack_states_which_module_sections_it_accepts():
    """A cell that serves a family of modules has to say which family.

    Both bounds are closed form and neither existed before this session, which
    is how ``BLADE_SIZE`` was moved across the map blind: 450 x 160 x 35 mm
    fails entry and 450 x 130 x 20 mm failed grip, in the same rack, and the
    only thing that ever measured either was a training run.
    """

    envelope = section_envelope()
    lookup = {(row["width_m"], row["height_m"]): row for row in envelope["sections"]}
    assert lookup[(0.130, 0.020)]["accepted"], lookup[(0.130, 0.020)]
    # The section this project used to run fails on the way in, which is the
    # finding that moved it, and it should stay visible.
    assert not lookup[(0.160, 0.035)]["lead_ins_admit_the_delivered_attitude"]
    # Thin and narrow is not free: it is the corner of the channel that grows.
    assert not lookup[(0.110, 0.014)]["pads_can_follow_the_corner"]
    assert 0 < envelope["accepted_count"] < envelope["evaluated_count"]
    # And the shipped section now sits *inside* the grip bound rather than on
    # it. It sat exactly on it while the guide offset was derived as the largest
    # clearance the pads can follow; the seated criterion binds from above now,
    # and this margin is a consequence of that rather than its purpose.
    margin = float(envelope["grip_margin_of_the_shipped_section_m"])
    assert 0.001 < margin < 0.002, envelope


def test_the_lead_ins_move_with_the_rails_they_continue():
    """A lead-in that does not track its own channel surface is a step, not a lead-in.

    ``_FLARE_CENTER_Y`` was an authored literal placed so the flare's inner face
    met the rail face exactly at the mouth. When ``GUIDE_CENTER_OFFSET_Y`` was
    derived and the rails moved inboard 3.061 mm, the flares stayed. And
    ``_RAMP_SURFACE_OFFSET`` is the *difference* between the two, written that
    way so "the two lead-ins cannot drift apart", so the vertical ramps moved
    3.061 mm the other way at the same time.

    Measured, before it was caught: the chain scored 0.00% over 32 episodes with
    the module arriving 1.0 mm from the seated plane and 47.1 mrad square. It
    was 4.04 mm of lateral against a 2.5 mm tolerance, against 1.85 mm on the
    same chain the day before.
    """

    assets = ASSETS.read_text(encoding="utf-8")
    guide = float(_literal("GUIDE_CENTER_OFFSET_Y"))
    # A round() call, not a literal, which is the point: it tracks the rail.
    flare = round(guide - 0.5 * GUIDE_THICKNESS_Y_M + float(_literal("FLARE_CENTRE_OUTSIDE_RAIL_FACE_M")), 6)
    outside = float(_literal("FLARE_CENTRE_OUTSIDE_RAIL_FACE_M"))
    rail_face = guide - 0.5 * GUIDE_THICKNESS_Y_M

    # The flare's inner face meets the rail face at the mouth, which is what the
    # lead-in is for.
    assert abs((flare - outside) - rail_face) < 1.0e-6, (flare, outside, rail_face)
    # And it is derived from the rail rather than remembered, so the next time a
    # channel dimension moves the lead-ins move with it.
    assert "_FLARE_CENTER_Y = round(GUIDE_CENTER_OFFSET_Y" in assets
    # The vertical ramp offset is then the same constant, which is what its own
    # comment claims and what stopped being true when the guides moved alone.
    assert "_RAMP_SURFACE_OFFSET = _FLARE_CENTER_Y - (GUIDE_CENTER_OFFSET_Y - 0.009)" in assets
    assert abs((flare - (guide - 0.009)) - outside) < 1.0e-6
