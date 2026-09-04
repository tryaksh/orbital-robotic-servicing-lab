#!/usr/bin/env bash
# The RGB-D cohorts that decide the 50% gate. Three held-out seeds each, and
# every arm is published beside the 4/24 it is trying to beat.
#
#   A  noised extract          one change from the published cohort: the
#                              extraction policy saw the estimator's error in
#                              training. Needs a retrain.
#   B  kinematic velocity      one change from the published cohort, and the
#                              *published* checkpoints: the module-velocity
#                              channel comes from the robot's own encoders
#                              rather than from differencing camera poses.
#                              Needs no retrain at all.
#   C  both, plus the lead-in  the best configuration this repository can field.
#      guard                   Three changes, each of which is measured alone
#                              elsewhere, so the combination is interpretable.
#
# A and B are the interesting pair: the channel attribution says restoring either
# observation channel recovers most of a 41-point loss, and these are the two
# ways to restore one -- train the policy on the noise, or stop generating it.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"

until ls "$ROOT/grapple_extract_l0_seed70_v19noised/nn/"*ep_14600_*.pth >/dev/null 2>&1; do sleep 120; done
until grep -q "master queue done" artifacts/campaign/master.log 2>/dev/null; do sleep 120; done
NOISED=$(ls -t "$ROOT/grapple_extract_l0_seed70_v19noised/nn/"*ep_14600_*.pth 2>/dev/null | head -1)
if [ -z "$NOISED" ]; then echo "no final noised checkpoint; nothing to certify"; exit 1; fi
echo "[$(date +%H:%M:%S)] RGB-D cohorts; noised extract is $NOISED"
mkdir -p artifacts/campaign/rgbdcohorts

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

run_arm () {
  tag="$1"; title="$2"; extract="$3"; note="$4"; shift 4
  rows=()
  for seed in 4070 5070 6070; do
    out="artifacts/campaign/rgbdcohorts/${tag}_seed${seed}"
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
    echo "[$(date +%H:%M:%S)]   $tag seed $seed exit=$rc"
    # **Fail an arm after one seed, not after three.** A configuration error
    # writes {"error": ...} into the report and produces no episodes -- that is
    # how the clearance-scope defect surfaced -- and spending the other two seeds
    # on it buys nothing. The arm is abandoned and the queue moves on, so one
    # broken arm cannot take the others down with it.
    # Corrected 2026-09-03: this tested for the *presence* of an "error" key.
    # Every successful report carries `"error": null`, so the screen failed all
    # three arms after they had run to completion. Test the value.
    ERR=$(./.venv/Scripts/python.exe -c "import json,sys; print(json.load(open(sys.argv[1])).get('error') or '')" "${out}_report.json" 2>/dev/null)
    if [ -n "$ERR" ] || [ ! -f "${out}.npz" ]; then
      echo "[$(date +%H:%M:%S)] $tag FAILED on its first seed; skipping the rest of this arm"
      grep -o '"error": "[^"]*"' "${out}_report.json" 2>/dev/null | head -1
      return 1
    fi
    rows+=("${out}.npz")
  done
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
      --output "evidence/workflow_robot_carried_vision_${tag}_certification.json" \
      --title "$title" \
      --scope "$SCOPE_1" "$SCOPE_2" "$note" "$SCOPE_3" \
      > "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $tag aggregate exit=$rc"
  tail -5 "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log"
}

run_arm noised_extract \
  "RGB-D chain with extraction trained on the estimator's error" \
  "$NOISED" \
  "One change from the published camera-driven cohort: the extraction policy resumed the certified v18pin checkpoint on a task whose module-derived observations carry the deployed estimator's certified residual, sample-and-hold and miss rate. Capture and the seating controller are unchanged."

run_arm kinematic_velocity \
  "RGB-D chain with the module velocity taken from the robot rather than the cameras" \
  "$E" \
  "One change from the published camera-driven cohort, and no retrain: every policy is the published checkpoint, and the module-velocity channel reports zero before capture and the wrist's own velocity after it. That is encoder and forward-kinematics information; no module state is read." \
  --module_velocity_source kinematics

run_arm noised_extract_kinematic_leadin \
  "RGB-D chain, best available configuration: trained on the estimator's error, velocity from the robot, guard on the flare's catch" \
  "$NOISED" \
  "Three changes from the published camera-driven cohort, each of which is measured alone in its own certification: the retrained extraction policy, the kinematic velocity channel, and the guarded advance admitting on the entry flare's catch rather than on the estimator's own noise bound." \
  --module_velocity_source kinematics --fiducial_guard_bounds lead_in

echo "[$(date +%H:%M:%S)] RGB-D cohorts done"
