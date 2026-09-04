#!/usr/bin/env bash
# What breaks this chain first?
#
# The chain is certified at one point in a space a real servicing cell varies
# across: one module, one rack, one robot position. "It is brittle" is a guess
# until something says *to what*, so this sweeps one variable at a time around
# the certified point and reports where the rate falls off.
#
# Sixteen environments and sixteen episodes per point, one seed. That is a
# coarse instrument on purpose -- a 16-episode Wilson interval is about twenty
# points wide -- and it is the right one for the question, which is ranking
# which knob breaks the chain soonest rather than measuring any of them
# precisely. Whatever it ranks first gets the 96-episode treatment.
#
# Bay pitch is swept through --robot_base_y rather than by moving the rack: the
# robot rides a rail to the destination bay, so a bay that is not where the rail
# stops and a rail that does not stop where the bay is are the same error, and
# only one of them is a scene rebuild.
#
# SWEEP_EXTRA passes one driver flag to every point, so a sweep can be re-run
# with one thing changed and compared against itself. It exists for
# --rack_clearance_scope: the published clearance points moved the side guides
# and left the entry flares, and re-measuring that needs the same points with
# the same seed under one changed flag, not a different script.

set -u
PYTHON="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
OUT="${OUT:-artifacts/robustness}"
mkdir -p "$OUT" evidence

GRASP_CKPT="${GRASP_CKPT:?set GRASP_CKPT}"
EXTRACT_CKPT="${EXTRACT_CKPT:?set EXTRACT_CKPT}"
INSERT_CKPT="${INSERT_CKPT:?set INSERT_CKPT}"
TASK="Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0"
ENVS="${ENVS:-16}"
EPISODES="${EPISODES:-16}"
SEED="${SEED:-4070}"

# POINTS selects a subset by tag, for the follow-up this script's own header
# promises: "whatever it ranks first gets the 96-episode treatment". The
# boundary validator reads five of these points and the nominal, so re-measuring
# a mismatch does not need the other four.
point() {
  tag="$1"; shift
  if [ -n "${POINTS:-}" ] && [[ " ${POINTS} " != *" ${tag} "* ]]; then
    return
  fi
  if [ -f "$OUT/${tag}.npz" ] && [ "${RESUME:-1}" = "1" ]; then
    echo "[$(date +%H:%M:%S)] $tag already done, skipping"
    return
  fi
  echo "[$(date +%H:%M:%S)] $tag  $*"
  "$PYTHON" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 --task "$TASK" \
      --grasp_checkpoint "$GRASP_CKPT" --extract_checkpoint "$EXTRACT_CKPT" \
      --insert_checkpoint "$INSERT_CKPT" \
      --num_envs "$ENVS" --episodes "$EPISODES" --seed "$SEED" --steps "${STEPS:-5000}" \
      --robot_rail_on_relocation \
      --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 \
      --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m "${RELIEF:-0.0046125}" \
      --mating_mode compliant --mating_force_cap_n 1000 \
      ${SWEEP_EXTRA:-} \
      "$@" \
      --report "$OUT/${tag}_report.json" --episode_metrics "$OUT/${tag}.npz" \
      > "$OUT/${tag}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)]   exit=$rc $(grep -oE '"success_rate": [0-9.]+' "$OUT/${tag}_report.json" | head -1)"
}

point nominal

# Module mass. The interface specification has to state a payload range and has
# never had one measured on a chain that carries the module by contact.
for kg in ${MASSES:-20 40}; do point "mass_${kg}kg" --module_mass_kg "$kg"; done

# Module cross-section. Every clearance in the cell is derived from it, and the
# last time it moved it cost the extract skill 22 points without anything saying so.
point "section_120x16" --module_cross_section_m 0.120 0.016
point "section_140x26" --module_cross_section_m 0.140 0.026

# Rack lateral tolerance around whatever module is fitted.
for mm in ${RACK_LAT:-6 16}; do point "rack_lat_${mm}mm" --rack_lateral_clearance_mm "$mm"; done

# Destination channel relief, the one rack tolerance this project already sweeps.
point "relief_0mm" --destination_channel_relief_m 0.0

# Where the robot stands. Along x this is the trade section 6a of the interface
# specification measures; across y it is the rail's own stopping error.
point "base_x_-0.70" --robot_base_x -0.70
# The rail's stopping error, as a ladder rather than as one point. The
# published sweep measured +10 mm and lost the chain to 1.6%, with 60 of 63
# failures timing out inside the *learned* phases and the channel untouched.
# One point cannot separate a geometric bound from a policy trained at one
# base position; the shape of the curve between 0 and 10 mm can. The default
# is the single published point, so nothing already measured moves.
for mm in ${BASE_Y_MM:-10}; do
  point "base_y_+${mm}mm" --robot_base_y "$(awk -v m="$mm" 'BEGIN{printf "%.6f", m/1000}')"
done

echo "[$(date +%H:%M:%S)] DONE"
