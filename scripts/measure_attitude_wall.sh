#!/usr/bin/env bash
# How far off the base's own plane does a target have to be before the arm can
# hold the head-on capture attitude at the depths the relocation needs?
#
# `evidence/relocation_reach_boundary.json` measured the boundary on the base's
# centre line and read it as a depth: the tool parks 0.4242 m forward of the base
# and the extraction end and transit retreat are 88.7 mm and 167 mm past that.
# `evidence/workcell_reach_solution.json` then found the same two poses
# converging to 0.01 mm in the second bay, 220 mm to the side, at full attitude
# authority -- same depth, same commanded attitude, same arm.
#
# Read across all 64 cells of that sweep the two are one boundary: every failing
# cell lies within 110 mm of the base centre line and every cell at 220 mm or
# beyond succeeds. So the region has a derived DEPTH -- 0.4242 m, moving one for
# one with the base -- and a WIDTH known only to lie between 110 and 220 mm,
# because no candidate was swept in between. This resolves the width: both
# failing depths, one environment per (depth, lateral offset, wrist seed, start
# pose), in a single app launch.
#
# **The base is pinned at the OLD cell, -0.45, and that is the whole point.**
# GRAPPLE_ROBOT_ROOT_POS is now -0.65, where both of these depths already solve
# on the centre line -- so run at the default and every cell converges and the
# probe measures nothing. The width of the region can only be measured where the
# region exists. Rule 6: prove a probe moves what it measures.
#
# Moving the TARGET off the centre line and moving
# the BASE off it are the same displacement measured from opposite ends, and this
# way one launch measures the whole profile instead of one launch per base.
#
# Usage: scripts/measure_attitude_wall.sh

set -u

PYTHON="C:/isaac-sim/python.bat"
OUT="${OUT:-artifacts/workcell}"
mkdir -p "$OUT" evidence

# Extraction end and transit retreat, as offsets from the installed module pose.
# Derived in scripts/solve_workcell.py from EXTRACTED_BLADE_CENTRE_X and
# TRANSIT_CLEAR_BLADE_CENTRE_X; restated nowhere.
DEPTHS="${DEPTHS:--0.494475 -0.572722}"
LATERAL="${LATERAL:-0.0 -0.05 -0.10 -0.15 -0.20 -0.22 -0.30 -0.40}"

echo "[$(date +%H:%M:%S)] LATERAL PROFILE of the attitude wall"
"$PYTHON" scripts/calibrate_grasp_pose.py --headless \
    --task Isaac-ZeroG-Blade-GrapplePin-Capture-v0 \
    --steps 3000 --pin_blade --finger_joint 0.02 --stages 0 \
    --sweep_offset_x $DEPTHS \
    --sweep_offset_y $LATERAL \
    --robot_base_x -0.45 --robot_base_y 0.0 --robot_base_z 0.15 \
    --alt_start_joint_pos 0 -1.5708 0 -1.5708 0 0 \
    --report evidence/attitude_wall_lateral_profile.json \
    > "$OUT/attitude_wall_lateral.log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] exit=$rc -> evidence/attitude_wall_lateral_profile.json"
