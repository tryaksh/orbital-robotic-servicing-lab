#!/usr/bin/env bash
# One ordered evaluation queue, highest paper value first. Each stage runs to
# completion before the next starts, so at most one evaluation process shares
# the GPU with the two training slots.
set -u
cd /d/6axis-space-robotics || exit 1
PY="C:/isaac-sim/python.bat"
CKPT_ROOT="logs/rl_games/zero_g_blade_insertion_contact"
G="$CKPT_ROOT/grapple_grasp_l0_seed70_v7m130/nn/last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
E="$CKPT_ROOT/grapple_extract_l0_seed70_v18pin/nn/last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
I="$CKPT_ROOT/grapple_insert_l0_seed70_v13m130/nn/last_zero_g_blade_insertion_contact_ep_8000_rew_-42.01845.pth"
export GRASP_CKPT="$G" EXTRACT_CKPT="$E" INSERT_CKPT="$I"

echo "[$(date +%H:%M:%S)] waiting for the base_y ladder to finish"
while [ ! -f artifacts/robustness64_baseladder/base_y_+6mm.npz ] \
   && [ ! -f "artifacts/robustness64_baseladder/base_y_+6mm_report.json" ]; do sleep 60; done

# ---------------------------------------------------------------- 1. RGB-D probe
echo "[$(date +%H:%M:%S)] STAGE 1  RGB-D chain probe with the noised extract"
NOISED=$(ls -t "$CKPT_ROOT/grapple_extract_l0_seed70_v19noised/nn/"last_*.pth 2>/dev/null | head -1)
mkdir -p artifacts/campaign/visionprobe
if [ -n "$NOISED" ]; then
  echo "[$(date +%H:%M:%S)]   using $NOISED"
  "$PY" scripts/run_workflow_demo.py --headless \
      --workflow relocate --curriculum_stage 0 \
      --task Isaac-ZeroG-Blade-GrappleVisionTwoSlot-Workflow-v0 \
      --grasp_checkpoint "$G" --extract_checkpoint "$NOISED" --insert_checkpoint "$I" \
      --num_envs 8 --seed 4070 --steps 1900 \
      --robot_rail_on_relocation --latch_on_release --latch_joint_mode fixed \
      --latch_rated_force_n 20000 --latch_rated_torque_nm 1000 \
      --latch_position_stiffness_n_per_m 40000 --latch_rotation_stiffness_nm_per_rad 20000 \
      --destination_channel_relief_m 0.0046125 --mating_mode compliant --mating_force_cap_n 1000 \
      --release_sequence simultaneous --perception_backend fiducial_pnp --rack_retention \
      --report artifacts/campaign/visionprobe/probe_report.json \
      --episode_metrics artifacts/campaign/visionprobe/probe.npz \
      > artifacts/campaign/visionprobe/probe.log 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)]   probe exit=$rc"
fi

# ------------------------------------------------- 2. which channel breaks extract
echo "[$(date +%H:%M:%S)] STAGE 2  channel isolation on the unchanged v18pin checkpoint"
mkdir -p artifacts/campaign/channels
for arm in Extract ExtractNoised ExtractPoseNoised ExtractVelocityNoised; do
  rows=()
  for seed in 1070 2070 3070; do
    out="artifacts/campaign/channels/${arm}_s0_seed${seed}"
    "$PY" scripts/play.py --headless \
        --task "Isaac-ZeroG-Blade-GrapplePin-${arm}-Play-v0" --checkpoint "$E" \
        --num_envs 128 --episodes 512 --curriculum_stage 0 --seed "$seed" \
        --episode_metrics "${out}.npz" > "${out}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)]   $arm seed=$seed exit=$rc"
    rows+=("${out}.npz")
  done
  ./.venv/Scripts/python.exe scripts/aggregate_evaluation.py --episodes "${rows[@]}" \
      --output "evidence/extract_channel_isolation_${arm}.json" \
      --title "Extraction on an unchanged checkpoint: ${arm}" \
      --scope \
        "Simulation only. No result here was produced on real hardware." \
        "The certified v18pin checkpoint, unchanged. Only which observation channels carry the estimator's error differs between these four reports." \
        "Curriculum stage 0 only: that is the station the chain hands extraction over at." \
      > "artifacts/campaign/channels/aggregate_${arm}.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)]   $arm aggregate exit=$rc"
  tail -4 "artifacts/campaign/channels/aggregate_${arm}.log"
done

# ------------------------------------------------------------- 3. guard bounds A/B
echo "[$(date +%H:%M:%S)] STAGE 3  guard-bounds A/B"
mkdir -p artifacts/campaign/guardbounds
for seed in 4070 5070 6070; do
  "$PY" scripts/run_workflow_demo.py --headless \
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
  echo "[$(date +%H:%M:%S)]   guard seed $seed exit=$rc"
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
echo "[$(date +%H:%M:%S)] guard aggregate exit=$rc"; tail -5 artifacts/campaign/guardbounds/aggregate.log

# ------------------------------------------- 4. what squares the module in the bay
echo "[$(date +%H:%M:%S)] STAGE 4  flares removed at 6 mm per side"
POINTS="rack_lat_6mm" EPISODES=64 ENVS=64 STEPS=6000 OUT=artifacts/robustness64_noflare SEED=4070 \
  SWEEP_EXTRA="--rack_clearance_scope channel --remove_entry_flares" \
  bash scripts/sweep_chain_robustness.sh
POINTS="nominal" EPISODES=64 ENVS=64 STEPS=6000 OUT=artifacts/robustness64_noflare_nominal SEED=4070 \
  SWEEP_EXTRA="--remove_entry_flares" bash scripts/sweep_chain_robustness.sh

# ------------------------------------------------------- 5. n on the section axis
echo "[$(date +%H:%M:%S)] STAGE 5  section axis at two further seeds"
for seed in 5070 6070; do
  POINTS="nominal section_120x16 section_140x26" EPISODES=64 ENVS=64 STEPS=6000 \
    OUT="artifacts/robustness64_seed${seed}" SEED="$seed" bash scripts/sweep_chain_robustness.sh
done

# ------------------------------------------------------------- 6. env-count probe
echo "[$(date +%H:%M:%S)] STAGE 6  environment count against episode length"
POINTS="nominal" EPISODES=64 ENVS=64 STEPS=1900 OUT=artifacts/envcount_64x1900 SEED=4070 \
  bash scripts/sweep_chain_robustness.sh
POINTS="nominal" EPISODES=32 ENVS=32 STEPS=6000 OUT=artifacts/envcount_32x6000 SEED=4070 \
  bash scripts/sweep_chain_robustness.sh
echo "[$(date +%H:%M:%S)] master queue done"
