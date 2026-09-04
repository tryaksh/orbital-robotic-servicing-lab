#!/usr/bin/env bash
# After the RGB-D probe: the guard-bounds A/B, then n on the section axis.
set -u
cd /d/6axis-space-robotics || exit 1
while [ ! -f artifacts/campaign/visionprobe/probe_report.json ]; do sleep 60; done
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"

# The guard-bounds A/B. Same three seeds, same checkpoints, same task, same
# eight environments; only the admissibility test the guarded advance uses
# changes, from the estimator's own noise bound to the entry flare's catch.
mkdir -p artifacts/campaign/guardbounds
for seed in 4070 5070 6070; do
  echo "[$(date +%H:%M:%S)] guard lead_in seed $seed"
  "C:/isaac-sim/python.bat" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 \
      --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
      --grasp_checkpoint "$G" --extract_checkpoint "$E" --insert_checkpoint "$I" \
      --num_envs 8 --seed "$seed" --steps 1900 \
      --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
      --fiducial_guard_bounds lead_in \
      --report "artifacts/campaign/guardbounds/leadin_seed${seed}_report.json" \
      --episode_metrics "artifacts/campaign/guardbounds/leadin_seed${seed}.npz" \
      > "artifacts/campaign/guardbounds/leadin_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)]   exit=$rc"
done
./.venv/Scripts/python.exe scripts/aggregate_evaluation.py \
    --episodes artifacts/campaign/guardbounds/leadin_seed*.npz \
    --output evidence/workflow_robot_carried_vision_leadin_guard_v1_certification.json \
    --title "RGB-D chain with the guarded advance admitting on the entry flare's catch" \
    --scope \
      "Simulation only. No result here was produced on real hardware." \
      "Every module-state channel the policies and the guard consume is camera-derived." \
      "One change from the shipped RGB-D cohort: the guarded advance's admissibility test is the entry flare's catch rather than the deployed estimator's own noise bound. The detection interlock is unchanged and still fails closed." \
      "Success requires 0.70 s supported settling, release of both robot-side supports, then a separate 0.70 s rack-only recheck on the disclosed break-rated Rack-to-module load path." \
    > artifacts/campaign/guardbounds/aggregate.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] guard aggregate exit=$rc"
tail -6 artifacts/campaign/guardbounds/aggregate.log

export GRASP_CKPT="$G" EXTRACT_CKPT="$E" INSERT_CKPT="$I"
for seed in 5070 6070; do
  POINTS="nominal section_120x16 section_140x26" EPISODES=64 ENVS=64 STEPS=6000 \
    OUT="artifacts/robustness64_seed${seed}" SEED="$seed" \
    bash scripts/sweep_chain_robustness.sh
  rc=$?
  echo "[$(date +%H:%M:%S)] section seed $seed exit=$rc"
done
echo "[$(date +%H:%M:%S)] all done"
