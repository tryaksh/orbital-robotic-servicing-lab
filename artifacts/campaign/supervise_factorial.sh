#!/usr/bin/env bash
# The complete 2x2x2 behind the camera-driven result.
#
# Three changes take the camera-driven chain from 4/24 to 13/16, and the arms
# measured so far say the combination is doing something none of the parts does:
# the retrained extraction alone scores 1/16, the kinematic velocity channel
# alone 1/16, and the guard on the flare's catch alone 12/24. Reporting "the
# combination works" from four of eight cells is a story. The factorial is a
# decomposition, and it is the difference between a result a reader can reuse on
# another system and one they can only admire.
#
#   N  extraction retrained on the estimator's error
#   K  module velocity from the robot's encoders rather than differenced camera poses
#   L  guarded advance admitting on the entry flare's catch rather than the estimator's noise bound
#
# Already measured at the current commit: N00 (N alone), 0K0, NKL.
# This runs the five that are missing: 000, 00L, 0KL, N0L, NK0.
#
# **Each cell aggregates immediately after its own three seeds, and retries once
# if the aggregation refuses.** `aggregate_evaluation.py` rejects a cohort whose
# runs came from different source commits, which is correct and which is exactly
# what happens if the repository is committed to while a cell is in flight. The
# retry re-runs the cell rather than asking anyone to coordinate.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED="$ROOT/grapple_extract_l0_seed70_v19noised/nn/last_zero_g_blade_insertion_contact_ep_14600_rew_166.19054.pth"
OUT=artifacts/campaign/factorial
mkdir -p "$OUT"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

say () { echo "[$(date +%H:%M:%S)] $*"; }

until grep -q "gate re-run done" artifacts/campaign/supervise_gate_rerun.log 2>/dev/null; do sleep 180; done
say "gate closed; filling in the rest of the factorial"

# cell <name> <extract checkpoint> [extra flags...]
cell () {
  name="$1"; extract="$2"; shift 2
  for attempt in 1 2; do
    rows=()
    for seed in 4070 5070 6070; do
      out="$OUT/${name}_seed${seed}"
      "$PY" scripts/run_workflow_demo.py --headless \
          --workflow relocate --curriculum_stage 0 \
          --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
          --grasp_checkpoint "$G" --extract_checkpoint "$extract" --insert_checkpoint "$I" \
          --num_envs 8 --seed "$seed" --steps 1900 \
          --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
          --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
          --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
          --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
          --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
          "$@" \
          --report "${out}_report.json" --episode_metrics "${out}.npz" \
          > "${out}.log" 2>&1
      rc=$?
      say "  $name seed $seed exit=$rc"
      [ -f "${out}.npz" ] && rows+=("${out}.npz")
    done
    if [ "${#rows[@]}" -ne 3 ]; then say "$name: only ${#rows[@]} of 3 seeds produced episodes; giving up on this cell"; return 1; fi
    ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
        --output "evidence/workflow_robot_carried_vision_factorial_${name}_certification.json" \
        --title "RGB-D chain, factorial cell ${name}" \
        --scope "$SCOPE_1" "$SCOPE_2" \
          "One cell of the 2x2x2 over the retrained extraction, the kinematic velocity channel and the lead-in guard bound. Every other term is the published camera-driven configuration." \
          "$SCOPE_3" \
        > "$OUT/aggregate_${name}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      say "$name aggregate ok"
      tail -4 "$OUT/aggregate_${name}.log"
      return 0
    fi
    say "$name aggregate refused on attempt $attempt:"
    tail -3 "$OUT/aggregate_${name}.log"
    [ "$attempt" -eq 1 ] && say "$name: re-running the cell so its three seeds share one commit"
  done
  return 1
}

# 000 -- the published configuration, re-measured at this commit so the factorial
# has an internally consistent baseline rather than one quoted from an older run.
cell base_000 "$E"

# 00L -- the guard bound alone.
cell guard_00L "$E" --fiducial_guard_bounds lead_in

# 0KL -- velocity from the robot, plus the guard.
cell velguard_0KL "$E" --module_velocity_source kinematics --fiducial_guard_bounds lead_in

# N0L -- retrained extraction, plus the guard.
cell noisedguard_N0L "$NOISED" --fiducial_guard_bounds lead_in

# NK0 -- both channel fixes, without the guard. The cell that says whether the
# guard is necessary or merely sufficient.
cell bothchannels_NK0 "$NOISED" --module_velocity_source kinematics

say "factorial done"
