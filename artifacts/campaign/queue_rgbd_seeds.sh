#!/usr/bin/env bash
# The two seeds each RGB-D arm never got.
#
# All three arms ran seed 4070 to completion on 2026-09-03 and were then thrown
# away by their own screening guard, which tested for the *presence* of an
# "error" key rather than a non-null value. Every report carries `"error": null`
# on success, so the guard fired on all three. The npz files were on disk the
# whole time. Nothing about the arms was wrong; the screen was.
#
# Seed 4070 says arm C is worth finishing: 7/8 against the published 4/24. That
# is one seed, it is not a certification, and it is not to be quoted until this
# queue has produced all three.
#
# This queue takes the evaluation slot ahead of the datum-pair work, because the
# pooled camera-driven rate is the submission gate and the datum-pair
# certificate is not. It relaunches that queue at the end, so the chain behind it
# is unchanged.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED=$(ls -t "$ROOT/grapple_extract_l0_seed70_v19noised/nn/"*ep_14600_*.pth 2>/dev/null | head -1)

until grep -q "noised skill certification done" artifacts/campaign/noised_skill_cert.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] evaluation slot free; finishing the RGB-D arms"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

finish_arm () {
  tag="$1"; title="$2"; extract="$3"; note="$4"; shift 4
  for seed in 5070 6070; do
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
    # The corrected screen: a real failure writes a *string* into "error", and a
    # successful report writes null. Test the value, and require the episodes.
    ERR=$(./.venv/Scripts/python.exe -c "import json,sys; print(json.load(open(sys.argv[1])).get('error') or '')" "${out}_report.json" 2>/dev/null)
    if [ -n "$ERR" ] || [ ! -f "${out}.npz" ]; then
      echo "[$(date +%H:%M:%S)] $tag seed $seed FAILED: ${ERR:-no episode metrics}"
      return 1
    fi
  done
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py \
      --episodes "artifacts/campaign/rgbdcohorts/${tag}_seed4070.npz" \
                 "artifacts/campaign/rgbdcohorts/${tag}_seed5070.npz" \
                 "artifacts/campaign/rgbdcohorts/${tag}_seed6070.npz" \
      --output "evidence/workflow_robot_carried_vision_${tag}_certification.json" \
      --title "$title" \
      --scope "$SCOPE_1" "$SCOPE_2" "$note" "$SCOPE_3" \
      > "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $tag aggregate exit=$rc"
  tail -6 "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log"
}

finish_arm noised_extract \
  "RGB-D chain with extraction trained on the estimator's error" \
  "$NOISED" \
  "One change from the published camera-driven cohort: the extraction policy resumed the certified v18pin checkpoint on a task whose module-derived observations carry the deployed estimator's certified residual, sample-and-hold and miss rate. Capture and the seating controller are unchanged."

finish_arm kinematic_velocity \
  "RGB-D chain with the module velocity taken from the robot rather than the cameras" \
  "$E" \
  "One change from the published camera-driven cohort, and no retrain: every policy is the published checkpoint, and the module-velocity channel reports zero before capture and the wrist's own velocity after it. That is encoder and forward-kinematics information; no module state is read." \
  --module_velocity_source kinematics

finish_arm noised_extract_kinematic_leadin \
  "RGB-D chain, best available configuration: trained on the estimator's error, velocity from the robot, guard on the flare's catch" \
  "$NOISED" \
  "Three changes from the published camera-driven cohort, each of which is measured alone in its own certification: the retrained extraction policy, the kinematic velocity channel, and the guarded advance admitting on the entry flare's catch rather than on the estimator's own noise bound." \
  --module_velocity_source kinematics --fiducial_guard_bounds lead_in

echo "[$(date +%H:%M:%S)] RGB-D seeds done"
nohup bash artifacts/campaign/queue_datum_pair_perception.sh > artifacts/campaign/datum_pair_perception.log 2>&1 &
echo "[$(date +%H:%M:%S)] datum-pair queue relaunched behind it"
