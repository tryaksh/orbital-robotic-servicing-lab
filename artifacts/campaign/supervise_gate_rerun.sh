#!/usr/bin/env bash
# Re-run seed 4070 for the three RGB-D arms so each cohort is internally consistent.
#
# All six new runs succeeded and `aggregate_evaluation.py` refused all three
# cohorts, correctly. Seed 4070 was measured last night at commit 6139fdac and
# seeds 5070 and 6070 today at 9669a8f9, and for the two noised arms the
# checkpoints differ too: `ls -t | head -1` picked
# `..._ep_14600_rew__166.19054_.pth` last night, and the explicit path today
# names `..._ep_14600_rew_166.19054.pth`. Both are epoch 14,600 and they are
# different weights -- policy sets 75403F3B and 1AFE41D3. Every queue in this
# repository that selects a checkpoint filters `_rew__` out for exactly this
# reason; the one that resolved it by modification time did not.
#
# So the guard did its job and the fix is not to relax it. Re-run 4070 on today's
# commit with the same checkpoint the other two seeds used, then aggregate.
#
# One wait loop, not twelve. The evaluation supervisor is mid-stage and this
# takes the slot after it.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
NOISED="$ROOT/grapple_extract_l0_seed70_v19noised/nn/last_zero_g_blade_insertion_contact_ep_14600_rew_166.19054.pth"

SCOPE_1="Simulation only. No result here was produced on real hardware."
SCOPE_2="Every module-state channel the policies and the guard consume is camera-derived unless this report's scope says otherwise."
SCOPE_3="Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path."

say () { echo "[$(date +%H:%M:%S)] $*"; }

until grep -q "evaluation supervisor done" artifacts/campaign/supervise_evaluation.log 2>/dev/null; do sleep 180; done
say "evaluation slot free; re-running seed 4070 on the current commit"

run_4070 () {
  tag="$1"; extract="$2"; shift 2
  out="artifacts/campaign/rgbdcohorts/${tag}_seed4070"
  mv -f "${out}.npz" "${out}_stale_6139fdac.npz" 2>/dev/null
  mv -f "${out}_report.json" "${out}_stale_6139fdac_report.json" 2>/dev/null
  "$PY" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 \
      --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
      --grasp_checkpoint "$G" --extract_checkpoint "$extract" --insert_checkpoint "$I" \
      --num_envs 8 --seed 4070 --steps 1900 \
      --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
      "$@" \
      --report "${out}_report.json" --episode_metrics "${out}.npz" \
      > "${out}.log" 2>&1
  rc=$?
  say "  $tag seed 4070 exit=$rc"
}

aggregate () {
  tag="$1"; title="$2"; note="$3"
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py \
      --episodes "artifacts/campaign/rgbdcohorts/${tag}_seed4070.npz" \
                 "artifacts/campaign/rgbdcohorts/${tag}_seed5070.npz" \
                 "artifacts/campaign/rgbdcohorts/${tag}_seed6070.npz" \
      --output "evidence/workflow_robot_carried_vision_${tag}_certification.json" \
      --title "$title" --scope "$SCOPE_1" "$SCOPE_2" "$note" "$SCOPE_3" \
      > "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log" 2>&1
  rc=$?
  say "$tag aggregate exit=$rc"
  tail -6 "artifacts/campaign/rgbdcohorts/aggregate_${tag}.log"
}

run_4070 noised_extract "$NOISED"
aggregate noised_extract \
  "RGB-D chain with extraction trained on the estimator's error" \
  "One change from the published camera-driven cohort: the extraction policy resumed the certified v18pin checkpoint on a task whose module-derived observations carry the deployed estimator's certified residual, sample-and-hold and miss rate. Capture and the seating controller are unchanged."

run_4070 kinematic_velocity "$E" --module_velocity_source kinematics
aggregate kinematic_velocity \
  "RGB-D chain with the module velocity taken from the robot rather than the cameras" \
  "One change from the published camera-driven cohort, and no retrain: every policy is the published checkpoint, and the module-velocity channel reports zero before capture and the wrist's own velocity after it. That is encoder and forward-kinematics information; no module state is read."

run_4070 noised_extract_kinematic_leadin "$NOISED" --module_velocity_source kinematics --fiducial_guard_bounds lead_in
aggregate noised_extract_kinematic_leadin \
  "RGB-D chain, best available configuration: trained on the estimator's error, velocity from the robot, guard on the flare's catch" \
  "Three changes from the published camera-driven cohort, each of which is measured alone in its own certification: the retrained extraction policy, the kinematic velocity channel, and the guarded advance admitting on the entry flare's catch rather than on the estimator's own noise bound."

say "gate re-run done"
